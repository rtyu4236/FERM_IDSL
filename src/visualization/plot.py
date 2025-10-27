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

plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = os.path.join(config.OUTPUT_DIR, logger.LOG_NAME)

def _save_df_as_image(df, filename):
    if df.empty:
        logger.warning(f"DataFrame is empty, skipping image save for {filename}")
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

def _calculate_returns_from_cumulative(cumulative_returns):
    returns = cumulative_returns.pct_change().fillna(0)
    return returns

def _calculate_downside_deviation(series, mar=0):
    returns_below_mar = series[series < mar]
    if returns_below_mar.empty:
        return 0.0
    return np.sqrt(np.mean((returns_below_mar - mar)**2))

def _calculate_sortino_ratio(series, mar=0):
    downside_dev = _calculate_downside_deviation(series, mar)
    if downside_dev == 0:
        return np.nan
    return (series.mean() - mar) / downside_dev


def generate_strategy_performance_summary(returns_df, avg_turnover_dict):

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
        panel_a_df.to_csv(os.path.join(OUTPUT_DIR, 'table_1_monthly_statistics.csv'))

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
            'Downside deviation': _calculate_downside_deviation(series),
            'Sortino ratio': _calculate_sortino_ratio(series),
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
        panel_b_df.to_csv(os.path.join(OUTPUT_DIR, 'table_2_annualized_risk_metrics.csv'))


def generate_sub_period_analysis(returns_df, start_year, end_year):
    if returns_df.empty:
        logger.warning("기간별 성과 분석을 위한 데이터가 없습니다.")
        return

    sub_periods = {}
    current_start = start_year
    while current_start <= end_year:
        current_end = min(current_start + 4, end_year)
        period_name = f'{current_start:04d}/01 - {current_end:04d}/12'
        sub_periods[period_name] = (f'{current_start:04d}-01-01', f'{current_end:04d}-12-31')
        current_start = current_end + 1

    metrics_to_calc = {
        'Mean return': qs.stats.cagr,
        'Sharpe ratio': qs.stats.sharpe,
        'Sortino ratio': _calculate_sortino_ratio,
        'Profit factor': qs.stats.profit_factor,
        'Maximum drawdown': qs.stats.max_drawdown,
        'Calmar ratio': qs.stats.calmar
    }

    all_periods_data = []

    for period_name, (start_date_str, end_date_str) in sub_periods.items():
        period_returns = returns_df.loc[start_date_str:end_date_str]
        if period_returns.empty:
            continue

        period_results = {}
        for col in period_returns.columns:
            series = period_returns[col].dropna()
            if series.empty: continue
            
            calculated_metrics = {metric_name: func(series) for metric_name, func in metrics_to_calc.items()}
            period_results[col] = calculated_metrics

        if period_results:
            period_results_df = pd.DataFrame(period_results)
            period_results_df.index.name = period_name
            all_periods_data.append(period_results_df)

    if all_periods_data:
        final_df = pd.concat(all_periods_data, keys=[name for name in sub_periods.keys()])
        _save_df_as_image(final_df, 'table_6_sub_period_analysis.png')
        final_df.to_csv(os.path.join(OUTPUT_DIR, 'table_6_sub_period_analysis.csv'))

        
def generate_performance_tables(strategy_returns, benchmark_returns):
    if strategy_returns.empty:
        logger.warning("Cannot generate performance analysis tables: input data is empty.")
        return
    all_returns = pd.concat([strategy_returns, benchmark_returns], axis=1)
    try:
        metrics_df = qs.reports.metrics(all_returns, mode='full')
        if metrics_df is not None:
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
        yearly_returns_df = all_returns.resample('Y').apply(qs.stats.comp).T
        yearly_returns_df.columns = yearly_returns_df.columns.strftime('%Y')
        yearly_path = os.path.join(OUTPUT_DIR, 'table_2_annual_returns.csv')
        yearly_returns_df.to_csv(yearly_path)
        _save_df_as_image(yearly_returns_df, 'table_2_annual_returns.png')
        logger.info(f"Success: Annual returns table saved.")
    except Exception as e:
        logger.error(f"Failed: Could not generate annual returns table: {e}")
        logger.error(traceback.format_exc())
    try:
        drawdown_df = qs.stats.drawdown_details(strategy_returns)
        if not drawdown_df.empty:
            drawdown_path = os.path.join(OUTPUT_DIR, 'table_3_drawdown_analysis.csv')
            drawdown_df.to_csv(drawdown_path)
            _save_df_as_image(drawdown_df, 'table_3_drawdown_analysis.png')
            logger.info(f"Success: Drawdown analysis table saved.")
        else:
            logger.info("No drawdown data available, skipping table generation.")
    except Exception as e:
        logger.error(f"Failed: Could not generate drawdown analysis table: {e}")
        logger.error(traceback.format_exc())


def generate_factor_analysis_table(strategy_returns, ff_df):
    if strategy_returns.empty:
        logger.warning("Cannot perform regression analysis: input data is empty.")
        return
    try:
        ff_df = ff_df.set_index('date')
        merged_data = pd.merge(strategy_returns, ff_df, left_index=True, right_index=True, how='inner')
        
        merged_data.to_csv(os.path.join(OUTPUT_DIR, 'data_fama_french_regression_input.csv'))
        logger.info(f"Success: Fama-French regression input data saved.")

        if merged_data.empty or len(merged_data) < 2:
            logger.warning("Stopping regression analysis due to insufficient data.")
            return
        y = merged_data[strategy_returns.name] - merged_data['RF']
        X = merged_data[['Mkt-RF', 'SMB', 'HML']]
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        results_summary = model.summary2().tables[1]
        results_summary.rename(index={'const': 'alpha'}, inplace=True)
        adj_r_squared = model.rsquared_adj
        r_squared_series = pd.Series({'Coef.': adj_r_squared, 'Std.Err.': np.nan, 't': np.nan, 'P>|t|': np.nan}, name='Adj. R-squared')
        results_df = pd.concat([results_summary, r_squared_series.to_frame().T])
        factor_path = os.path.join(OUTPUT_DIR, 'table_4_fama_french_regression.csv')
        results_df.to_csv(factor_path)
        logger.info(f"Success: Fama-French regression results table saved.")
    except Exception as e:
        logger.error(f"Failed: Fama-French regression analysis failed: {e}")
        logger.error(traceback.format_exc())


def plot_cumulative_returns(cumulative_df):
    if cumulative_df.empty:
        logger.warning("Cannot generate cumulative returns plot: input data is empty.")
        return
    try:
        strategy_col = cumulative_df.columns[0]
        returns_for_plot = _calculate_returns_from_cumulative(cumulative_df[strategy_col])
        benchmarks_for_plot = _calculate_returns_from_cumulative(cumulative_df.drop(columns=strategy_col))
        fig = qs.plots.returns(returns_for_plot, benchmark=benchmarks_for_plot, 
                               savefig=os.path.join(OUTPUT_DIR, 'plot_1_cumulative_returns.png'))
        plt.close(fig)
        logger.info(f"Success: Cumulative returns plot saved.")
    except Exception as e:
        logger.error(f"Failed: Could not generate cumulative returns plot: {e}")
        logger.error(traceback.format_exc())


def plot_underwater(strategy_returns):
    if strategy_returns.empty:
        logger.warning("Cannot generate underwater plot: input data is empty.")
        return
    try:
        underwater_series = qs.stats.drawdown(strategy_returns)
        underwater_series.to_csv(os.path.join(OUTPUT_DIR, 'data_underwater.csv'))
        logger.info(f"Success: Underwater (drawdown) data saved.")

        fig = qs.plots.drawdown(strategy_returns, 
                                savefig=os.path.join(OUTPUT_DIR, 'plot_2_underwater.png'))
        plt.close(fig)
        logger.info(f"Success: Underwater plot saved.")
    except Exception as e:
        logger.error(f"Failed: Could not generate underwater plot: {e}")
        logger.error(traceback.format_exc())


def plot_additional_analytics(strategy_returns):
    if strategy_returns.empty:
        logger.warning("Cannot generate additional analytics plots: input data is empty.")
        return
    try:
        monthly_returns_table = qs.stats.monthly_returns(strategy_returns)
        monthly_returns_table.to_csv(os.path.join(OUTPUT_DIR, 'data_monthly_returns_heatmap.csv'))
        logger.info(f"Success: Monthly returns heatmap data saved.")

        fig = qs.plots.monthly_returns(strategy_returns, 
                                       savefig=os.path.join(OUTPUT_DIR, 'plot_3_monthly_returns.png'))
        plt.close(fig)
        logger.info(f"Success: Monthly returns plot saved.")
    except Exception as e:
        logger.error(f"Failed: Could not generate monthly returns plot: {e}")
        logger.error(traceback.format_exc())
    
    ROLLING_WINDOW = 12
    if len(strategy_returns) > ROLLING_WINDOW:
        try:
            rolling_vol = qs.stats.rolling_volatility(strategy_returns, window=ROLLING_WINDOW)
            rolling_vol.to_csv(os.path.join(OUTPUT_DIR, 'data_rolling_volatility.csv'))
            logger.info(f"Success: Rolling volatility data saved.")

            fig = qs.plots.rolling_volatility(strategy_returns, 
                                              savefig=os.path.join(OUTPUT_DIR, 'plot_4_rolling_volatility.png'))
            plt.close(fig)
            logger.info(f"Success: Rolling volatility plot saved.")
        except Exception as e:
            logger.error(f"Failed: Could not generate rolling volatility plot: {e}")
            logger.error(traceback.format_exc())
        try:
            rolling_sharpe = qs.stats.rolling_sharpe(strategy_returns, window=ROLLING_WINDOW)
            rolling_sharpe.to_csv(os.path.join(OUTPUT_DIR, 'data_rolling_sharpe.csv'))
            logger.info(f"Success: Rolling Sharpe ratio data saved.")

            fig = qs.plots.rolling_sharpe(strategy_returns, 
                                          savefig=os.path.join(OUTPUT_DIR, 'plot_5_rolling_sharpe.png'))
            plt.close(fig)
            logger.info(f"Success: Rolling Sharpe ratio plot saved.")
        except Exception as e:
            logger.error(f"Failed: Could not generate rolling Sharpe ratio plot: {e}")
            logger.error(traceback.format_exc())
    else:
        logger.warning(f"Skipping rolling stats plots: data period ({len(strategy_returns)} months) is shorter than rolling window ({ROLLING_WINDOW} months).")


def plot_monthly_returns_comparison(returns_df):
    if returns_df.empty:
        logger.warning("Cannot generate monthly returns comparison plot: input data is empty.")
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


def run_visualization(cumulative_df, ff_df, avg_turnover_dict, start_year, end_year):
    if cumulative_df.empty:
        logger.warning("Stopping visualization process: cumulative returns data is empty.")
        return
    returns_df = _calculate_returns_from_cumulative(cumulative_df)
    strategy_col = 'TCN-SVR'
    strategy_returns = returns_df[strategy_col]
    benchmark_returns = returns_df.drop(columns=[strategy_col])
    generate_strategy_performance_summary(returns_df, avg_turnover_dict)
    generate_sub_period_analysis(returns_df, start_year, end_year)
    generate_factor_analysis_table(strategy_returns, ff_df)

    cumulative_df_plot = cumulative_df[cumulative_df.index.year >= 2010]
    returns_df_plot = returns_df[returns_df.index.year >= 2010]
    strategy_returns_plot = returns_df_plot[strategy_col] if strategy_col in returns_df_plot else pd.Series()

    plot_cumulative_returns(cumulative_df_plot)
    plot_underwater(strategy_returns_plot)
    plot_additional_analytics(strategy_returns_plot)
    plot_monthly_returns_comparison(returns_df_plot)
    logger.info("Visualization and analysis process complete.")

if __name__ == '__main__':
    try:
        cumulative_returns_path = os.path.join(OUTPUT_DIR, 'cumulative_returns.csv')
        cumulative_df = pd.read_csv(cumulative_returns_path, index_col=0, parse_dates=True)
        _, _, _, ff_df, _ = data_manager.load_raw_data()
        avg_turnover_dict = {'BL_ML_Strategy': 0.15} 
        run_visualization(cumulative_df, ff_df, avg_turnover_dict)
    except FileNotFoundError:
        logger.error("To run visualizer.py standalone, you must first run the backtest to generate 'cumulative_returns.csv'.")
    except Exception as e:
        logger.error(f"Error running visualizer.py standalone: {e}")
        logger.error(traceback.format_exc())