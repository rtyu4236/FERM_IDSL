import pandas as pd
import quantstats as qs
import statsmodels.api as sm
import numpy as np
import os
from config import settings as config
from src.data_processing import manager as data_manager
from src.utils.logger import logger
import matplotlib.pyplot as plt
import traceback
import scipy.stats

OUTPUT_DIR = config.OUTPUT_DIR

def _save_df_as_image(df, filename):
    logger.info(f"[_save_df_as_image] Function entry. Filename: {filename}")
    logger.info(f"[_save_df_as_image] Input df shape={df.shape}, columns={df.columns.tolist()}")
    if df.empty:
        logger.warning(f"DataFrame is empty, skipping image save for {filename}")
        logger.info("[_save_df_as_image] Function exit (empty df).")
        return
    try:
        df = df.round(4)
        fig, ax = plt.subplots(figsize=(max(10, len(df.columns) * 2), max(4, len(df) * 0.5)))
        ax.axis('off')
        ax.axis('tight')
        table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, cellLoc='center', loc='center')
        table.set_fontsize(10)
        table.scale(1.2, 1.2)
        fig.tight_layout()
        path = os.path.join(OUTPUT_DIR, filename)
        fig.savefig(path, bbox_inches='tight', dpi=200)
        plt.close(fig)
        logger.info(f"SUCCESS: Saved table image to {path}")
    except Exception as e:
        logger.error(f"FAILED to save table image {filename}: {e}")
        logger.error(traceback.format_exc())
    logger.info("[_save_df_as_image] Function exit.")

def _calculate_returns_from_cumulative(cumulative_returns):
    logger.info(f"[_calculate_returns_from_cumulative] Function entry. Input cumulative_returns shape={cumulative_returns.shape}")
    returns = cumulative_returns.pct_change().fillna(0)
    logger.info(f"[_calculate_returns_from_cumulative] Output returns shape={returns.shape}")
    logger.info("[_calculate_returns_from_cumulative] Function exit.")
    return returns

def generate_strategy_performance_summary(returns_df, avg_turnover_dict):
    """전략별 상세 성과 요약표 (논문의 Table 2 형식) 생성 (완전판)."""
    logger.info("상세 전략 성과 요약표 생성을 시작합니다.")

    # --- 패널 (a): 월별 수익률 상세 통계 ---
    panel_a_data = {}
    for col in returns_df.columns:
        series = returns_df[col].dropna()
        if series.empty: continue
        t_stat, _ = scipy.stats.ttest_1samp(series, 0)
        stats = {
            'Mean': series.mean(),
            'Standard deviation': series.std(),
            'Standard error': scipy.stats.sem(series),
            't-statistic': t_stat,
            'Min': series.min(),
            '25%': series.quantile(0.25),
            '50%': series.quantile(0.50),
            '75%': series.quantile(0.75),
            'Max': series.max(),
            'Skew': scipy.stats.skew(series),
            'Kurtosis': scipy.stats.kurtosis(series),
        }
        panel_a_data[col] = stats
    if panel_a_data:
        panel_a_df = pd.DataFrame(panel_a_data)
        ordered_index_a = [
            'Mean', 'Standard deviation', 'Standard error', 't-statistic', 'Min',
            '25%', '50%', '75%', 'Max', 'Skew', 'Kurtosis'
        ]
        panel_a_df = panel_a_df.reindex(ordered_index_a)
        _save_df_as_image(panel_a_df, 'table_1_monthly_statistics.png')

    # --- 패널 (b): 연환산 리스크-수익 지표 ---
    panel_b_data = {}
    for col in returns_df.columns:
        series = returns_df[col].dropna()
        if series.empty: continue
        
        yearly_comp = series.resample('Y').apply(qs.stats.comp)
        t_stat, _ = scipy.stats.ttest_1samp(series, 0)

        metrics = {
            'Mean return': qs.stats.cagr(series),
            'Standard deviation': qs.stats.volatility(series),
            'Sharpe ratio': qs.stats.sharpe(series),
            't-statistic': t_stat,
            'Downside deviation': qs.stats.downside_risk(series),
            'Sortino ratio': qs.stats.sortino(series),
            'Gross profit': series[series > 0].sum(),
            'Gross loss': series[series < 0].sum(),
            'Profit factor': qs.stats.profit_factor(series),
            'Profitable years': (yearly_comp > 0).sum(),
            'Unprofitable years': (yearly_comp <= 0).sum(),
            'Maximum drawdown': qs.stats.max_drawdown(series),
            'Calmar ratio': qs.stats.calmar(series),
            'Turnover': avg_turnover_dict.get(col, np.nan) 
        }
        panel_b_data[col] = metrics
    
    if panel_b_data:
        panel_b_df = pd.DataFrame(panel_b_data)
        ordered_index_b = [
            'Mean return', 'Standard deviation', 'Sharpe ratio', 't-statistic',
            'Downside deviation', 'Sortino ratio', 'Gross profit', 'Gross loss',
            'Profit factor', 'Profitable years', 'Unprofitable years', 
            'Maximum drawdown', 'Calmar ratio', 'Turnover'
        ]
        panel_b_df = panel_b_df.reindex(ordered_index_b)
        _save_df_as_image(panel_b_df, 'table_2_annualized_risk_metrics.png')

def generate_sub_period_analysis(returns_df):
    """기간별 성과 분석표 (논문의 Table 6 형식) 생성."""
    logger.info("기간별 성과 분석표 생성을 시작합니다.")
    if returns_df.empty:
        logger.warning("기간별 성과 분석을 위한 데이터가 없습니다.")
        return

    backtest_start_date = returns_df.index.min()
    backtest_end_date = returns_df.index.max()

    potential_periods = {
        '01/1980 - 12/1999': (pd.to_datetime('1980-01-01'), pd.to_datetime('1999-12-31')),
        '01/2000 - 12/2006': (pd.to_datetime('2000-01-01'), pd.to_datetime('2006-12-31')),
        '01/2007 - 12/2009': (pd.to_datetime('2007-01-01'), pd.to_datetime('2009-12-31')),
        '01/2010 - 12/2019': (pd.to_datetime('2010-01-01'), pd.to_datetime('2019-12-31')),
        '01/2020 - 12/2020': (pd.to_datetime('2020-01-01'), pd.to_datetime('2020-12-31'))
    }

    periods_to_analyze = {}
    for name, (period_start, period_end) in potential_periods.items():
        # Check for overlap
        if period_start <= backtest_end_date and period_end >= backtest_start_date:
            actual_start = max(period_start, backtest_start_date)
            actual_end = min(period_end, backtest_end_date)
            new_name = f"{actual_start.strftime('%m/%Y')} - {actual_end.strftime('%m/%Y')}"
            periods_to_analyze[new_name] = (actual_start, actual_end)

    if not periods_to_analyze:
        logger.warning("백테스트 기간과 겹치는 분석 대상 기간이 없습니다.")
        return

    metrics_to_calc = {
        'Mean return': qs.stats.cagr,
        'Sharpe ratio': qs.stats.sharpe,
        'Sortino ratio': qs.stats.sortino,
        'Profit factor': qs.stats.profit_factor,
        'Maximum drawdown': qs.stats.max_drawdown,
        'Calmar ratio': qs.stats.calmar
    }

    all_periods_data = []

    for period_name, (start_date, end_date) in periods_to_analyze.items():
        period_df = returns_df.loc[start_date:end_date]
        if period_df.empty:
            continue

        period_results = {}
        for col in period_df.columns:
            series = period_df[col].dropna()
            if series.empty: continue
            
            calculated_metrics = {metric_name: func(series) for metric_name, func in metrics_to_calc.items()}
            period_results[col] = calculated_metrics

        if period_results:
            period_results_df = pd.DataFrame(period_results)
            period_results_df.index.name = period_name
            all_periods_data.append(period_results_df)

    if all_periods_data:
        final_df = pd.concat(all_periods_data, keys=periods_to_analyze.keys())
        _save_df_as_image(final_df, 'table_6_sub_period_analysis.png')

def generate_performance_tables(strategy_returns, benchmark_returns):
    logger.info("[generate_performance_tables] Function entry.")
    logger.info(f"[generate_performance_tables] Input: strategy_returns shape={strategy_returns.shape}, benchmark_returns shape={benchmark_returns.shape}")
    logger.info("Starting generation of performance analysis tables.")
    if strategy_returns.empty:
        logger.warning("Cannot generate performance analysis tables: input data is empty.")
        logger.info("[generate_performance_tables] Function exit (empty strategy_returns).")
        return
    all_returns = pd.concat([strategy_returns, benchmark_returns], axis=1)
    logger.info(f"[generate_performance_tables] all_returns shape={all_returns.shape}")
    try:
        logger.info("Generating key performance metrics table...")
        metrics_df = qs.reports.metrics(all_returns, mode='full')
        if metrics_df is not None:
            logger.info(f"[generate_performance_tables] metrics_df shape={metrics_df.shape}")
            metrics_path = os.path.join(OUTPUT_DIR, 'table_1_performance_summary.csv')
            metrics_df.to_csv(metrics_path)
            _save_df_as_image(metrics_df, 'table_1_performance_summary.png')
            logger.info(f"Success: Key performance metrics table saved.")
        else:
            logger.error("Failed: Metrics dataframe was not generated.")
    except Exception as e:
        logger.error(f"Failed: Could not generate key performance metrics table: {e}")
        logger.error(traceback.format_exc())
    try:
        logger.info("Generating annual returns table...")
        yearly_returns_df = all_returns.resample('Y').apply(qs.stats.comp).T
        yearly_returns_df.columns = yearly_returns_df.columns.strftime('%Y')
        logger.info(f"[generate_performance_tables] yearly_returns_df shape={yearly_returns_df.shape}")
        yearly_path = os.path.join(OUTPUT_DIR, 'table_2_annual_returns.csv')
        yearly_returns_df.to_csv(yearly_path)
        _save_df_as_image(yearly_returns_df, 'table_2_annual_returns.png')
        logger.info(f"Success: Annual returns table saved.")
    except Exception as e:
        logger.error(f"Failed: Could not generate annual returns table: {e}")
        logger.error(traceback.format_exc())
    try:
        logger.info("Generating worst drawdown periods table...")
        drawdown_df = qs.stats.drawdown_details(strategy_returns)
        if not drawdown_df.empty:
            logger.info(f"[generate_performance_tables] drawdown_df shape={drawdown_df.shape}")
            drawdown_path = os.path.join(OUTPUT_DIR, 'table_3_drawdown_analysis.csv')
            drawdown_df.to_csv(drawdown_path)
            _save_df_as_image(drawdown_df, 'table_3_drawdown_analysis.png')
            logger.info(f"Success: Drawdown analysis table saved.")
        else:
            logger.info("No drawdown data available, skipping table generation.")
    except Exception as e:
        logger.error(f"Failed: Could not generate drawdown analysis table: {e}")
        logger.error(traceback.format_exc())
    logger.info("[generate_performance_tables] Function exit.")


def generate_factor_analysis_table(strategy_returns, ff_df):
    logger.info("[generate_factor_analysis_table] Function entry.")
    logger.info(f"[generate_factor_analysis_table] Input: strategy_returns shape={strategy_returns.shape}, ff_df shape={ff_df.shape}")
    logger.info("Starting Fama-French 3-factor regression analysis.")
    if strategy_returns.empty:
        logger.warning("Cannot perform regression analysis: input data is empty.")
        logger.info("[generate_factor_analysis_table] Function exit (empty strategy_returns).")
        return
    try:
        ff_df = ff_df.set_index('date')
        merged_data = pd.merge(strategy_returns, ff_df, left_index=True, right_index=True, how='inner')
        logger.info(f"[generate_factor_analysis_table] merged_data shape={merged_data.shape}")
        if merged_data.empty or len(merged_data) < 2:
            logger.warning("Stopping regression analysis due to insufficient data.")
            logger.info("[generate_factor_analysis_table] Function exit (insufficient merged_data).")
            return
        y = merged_data[strategy_returns.name] - merged_data['RF']
        X = merged_data[['Mkt-RF', 'SMB', 'HML']]
        X = sm.add_constant(X)
        logger.info(f"[generate_factor_analysis_table] y shape={y.shape}, X shape={X.shape}")
        model = sm.OLS(y, X).fit()
        results_summary = model.summary2().tables[1]
        results_summary.rename(index={'const': 'alpha'}, inplace=True)
        adj_r_squared = model.rsquared_adj
        r_squared_series = pd.Series({'Coef.': adj_r_squared, 'Std.Err.': np.nan, 't': np.nan, 'P>|t|': np.nan}, name='Adj. R-squared')
        results_df = pd.concat([results_summary, r_squared_series.to_frame().T])
        logger.info(f"[generate_factor_analysis_table] results_df shape={results_df.shape}")
        factor_path = os.path.join(OUTPUT_DIR, 'table_4_fama_french_regression.csv')
        results_df.to_csv(factor_path)
        logger.info(f"Success: Fama-French regression results table saved.")
    except Exception as e:
        logger.error(f"Failed: Fama-French regression analysis failed: {e}")
        logger.error(traceback.format_exc())
    logger.info("[generate_factor_analysis_table] Function exit.")

def plot_cumulative_returns(cumulative_df):
    logger.info("[plot_cumulative_returns] Function entry.")
    logger.info(f"[plot_cumulative_returns] Input: cumulative_df shape={cumulative_df.shape}")
    logger.info("Generating cumulative returns plot.")
    if cumulative_df.empty:
        logger.warning("Cannot generate cumulative returns plot: input data is empty.")
        logger.info("[plot_cumulative_returns] Function exit (empty cumulative_df).")
        return
    try:
        strategy_col = cumulative_df.columns[0]
        returns_for_plot = _calculate_returns_from_cumulative(cumulative_df[strategy_col])
        benchmarks_for_plot = _calculate_returns_from_cumulative(cumulative_df.drop(columns=strategy_col))
        logger.info(f"[plot_cumulative_returns] returns_for_plot shape={returns_for_plot.shape}, benchmarks_for_plot shape={benchmarks_for_plot.shape}")
        fig = qs.plots.returns(returns_for_plot, benchmark=benchmarks_for_plot, 
                               savefig=os.path.join(OUTPUT_DIR, 'plot_1_cumulative_returns.png'))
        plt.close(fig)
        logger.info(f"Success: Cumulative returns plot saved.")
    except Exception as e:
        logger.error(f"Failed: Could not generate cumulative returns plot: {e}")
        logger.error(traceback.format_exc())
    logger.info("[plot_cumulative_returns] Function exit.")

def plot_underwater(strategy_returns):
    logger.info("[plot_underwater] Function entry.")
    logger.info(f"[plot_underwater] Input: strategy_returns shape={strategy_returns.shape}")
    logger.info("Generating underwater plot.")
    if strategy_returns.empty:
        logger.warning("Cannot generate underwater plot: input data is empty.")
        logger.info("[plot_underwater] Function exit (empty strategy_returns).")
        return
    try:
        fig = qs.plots.drawdown(strategy_returns, 
                                savefig=os.path.join(OUTPUT_DIR, 'plot_2_underwater.png'))
        plt.close(fig)
        logger.info(f"Success: Underwater plot saved.")
    except Exception as e:
        logger.error(f"Failed: Could not generate underwater plot: {e}")
        logger.error(traceback.format_exc())
    logger.info("[plot_underwater] Function exit.")

def plot_additional_analytics(strategy_returns):
    logger.info("[plot_additional_analytics] Function entry.")
    logger.info(f"[plot_additional_analytics] Input: strategy_returns shape={strategy_returns.shape}")
    logger.info("Starting generation of additional analytics plots (monthly returns, rolling stats).")
    if strategy_returns.empty:
        logger.warning("Cannot generate additional analytics plots: input data is empty.")
        logger.info("[plot_additional_analytics] Function exit (empty strategy_returns).")
        return
    try:
        fig = qs.plots.monthly_returns(strategy_returns, 
                                       savefig=os.path.join(OUTPUT_DIR, 'plot_3_monthly_returns.png'))
        plt.close(fig)
        logger.info(f"Success: Monthly returns plot saved.")
    except Exception as e:
        logger.error(f"Failed: Could not generate monthly returns plot: {e}")
        logger.error(traceback.format_exc())
    
    ROLLING_WINDOW = 12
    logger.info(f"[plot_additional_analytics] strategy_returns length: {len(strategy_returns)}, ROLLING_WINDOW: {ROLLING_WINDOW}")
    if len(strategy_returns) > ROLLING_WINDOW:
        try:
            fig = qs.plots.rolling_volatility(strategy_returns, 
                                              savefig=os.path.join(OUTPUT_DIR, 'plot_4_rolling_volatility.png'))
            plt.close(fig)
            logger.info(f"Success: Rolling volatility plot saved.")
        except Exception as e:
            logger.error(f"Failed: Could not generate rolling volatility plot: {e}")
            logger.error(traceback.format_exc())
        try:
            fig = qs.plots.rolling_sharpe(strategy_returns, 
                                          savefig=os.path.join(OUTPUT_DIR, 'plot_5_rolling_sharpe.png'))
            plt.close(fig)
            logger.info(f"Success: Rolling Sharpe ratio plot saved.")
        except Exception as e:
            logger.error(f"Failed: Could not generate rolling Sharpe ratio plot: {e}")
            logger.error(traceback.format_exc())
    else:
        logger.warning(f"Skipping rolling stats plots: data period ({len(strategy_returns)} months) is shorter than rolling window ({ROLLING_WINDOW} months).")
    logger.info("[plot_additional_analytics] Function exit.")

def plot_monthly_returns_comparison(returns_df):
    logger.info("[plot_monthly_returns_comparison] Function entry.")
    logger.info(f"[plot_monthly_returns_comparison] Input: returns_df shape={returns_df.shape}")
    logger.info("Starting generation of monthly returns comparison plot.")
    if returns_df.empty:
        logger.warning("Cannot generate monthly returns comparison plot: input data is empty.")
        logger.info("[plot_monthly_returns_comparison] Function exit (empty returns_df).")
        return
    try:
        fig, ax = plt.subplots(figsize=(15, 7))
        returns_df.plot(ax=ax, kind='line')
        ax.set_title('Monthly Returns Comparison')
        ax.set_xlabel('Date')
        ax.set_ylabel('Monthly Return')
        ax.axhline(0, color='grey', linestyle='--')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, 'plot_6_monthly_returns_comparison.png')
        fig.savefig(path, dpi=200)
        plt.close(fig)
        logger.info(f"Success: Monthly returns comparison plot saved.")
    except Exception as e:
        logger.error(f"Failed: Could not generate monthly returns comparison plot: {e}")
        logger.error(traceback.format_exc())
    logger.info("[plot_monthly_returns_comparison] Function exit.")

def run_visualization(cumulative_df, ff_df, avg_turnover_dict):
    logger.info("[run_visualization] Function entry.")
    logger.info(f"[run_visualization] Input: cumulative_df shape={cumulative_df.shape}, ff_df shape={ff_df.shape}")
    logger.info("Starting visualization and analysis process.")
    if cumulative_df.empty:
        logger.warning("Stopping visualization process: cumulative returns data is empty.")
        logger.info("[run_visualization] Function exit (empty cumulative_df).")
        return
    returns_df = _calculate_returns_from_cumulative(cumulative_df)
    strategy_col = 'BL_ML_Strategy'
    strategy_returns = returns_df[strategy_col]
    benchmark_returns = returns_df.drop(columns=[strategy_col])
    logger.info(f"[run_visualization] returns_df shape={returns_df.shape}, strategy_returns shape={strategy_returns.shape}, benchmark_returns shape={benchmark_returns.shape}")
    # generate_performance_tables(strategy_returns, benchmark_returns)
    generate_strategy_performance_summary(returns_df, avg_turnover_dict)
    generate_sub_period_analysis(returns_df)
    generate_factor_analysis_table(strategy_returns, ff_df)
    plot_cumulative_returns(cumulative_df)
    plot_underwater(strategy_returns)
    plot_additional_analytics(strategy_returns)
    plot_monthly_returns_comparison(returns_df)
    logger.info("Visualization and analysis process complete.")
    logger.info("[run_visualization] Function exit.")

if __name__ == '__main__':
    # This block is for standalone testing of the visualizer
    try:
        cumulative_returns_path = os.path.join(OUTPUT_DIR, 'cumulative_returns.csv')
        cumulative_df = pd.read_csv(cumulative_returns_path, index_col=0, parse_dates=True)
        _, _, _, ff_df, _ = data_manager.load_raw_data()
        # Dummy data for testing
        avg_turnover_dict = {'BL_ML_Strategy': 0.15} 
        run_visualization(cumulative_df, ff_df, avg_turnover_dict)
    except FileNotFoundError:
        logger.error("To run visualizer.py standalone, you must first run the backtest to generate 'cumulative_returns.csv'.")
    except Exception as e:
        logger.error(f"Error running visualizer.py standalone: {e}")
        logger.error(traceback.format_exc())