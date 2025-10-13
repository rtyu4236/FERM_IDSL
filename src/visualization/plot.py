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

def generate_performance_tables(strategy_returns, benchmark_returns):
    logger.info("[generate_performance_tables] Function entry.")
    logger.info(f"[generate_performance_tables] Input: strategy_returns shape={strategy_returns.shape}, benchmark_returns shape={benchmark_returns.shape}")
    logger.info("성과 분석 표 생성을 시작합니다.")
    if strategy_returns.empty:
        logger.warning("성과 분석 표를 생성할 수 없음: 입력 데이터가 비어있음.")
        logger.info("[generate_performance_tables] Function exit (empty strategy_returns).")
        return
    all_returns = pd.concat([strategy_returns, benchmark_returns], axis=1)
    logger.info(f"[generate_performance_tables] all_returns shape={all_returns.shape}")
    try:
        logger.info("주요 성과 지표 표 생성 중...")
        metrics_df = qs.reports.metrics(all_returns, mode='full')
        if metrics_df is not None:
            logger.info(f"[generate_performance_tables] metrics_df shape={metrics_df.shape}")
            metrics_path = os.path.join(OUTPUT_DIR, 'table_1_performance_summary.csv')
            metrics_df.to_csv(metrics_path)
            _save_df_as_image(metrics_df, 'table_1_performance_summary.png')
            logger.info(f"성공: 주요 성과 지표 표 저장 완료.")
        else:
            logger.error("실패: 메트릭 데이터프레임이 생성되지 않았습니다.")
    except Exception as e:
        logger.error(f"실패: 주요 성과 지표 표 생성 실패: {e}")
        logger.error(traceback.format_exc())
    try:
        logger.info("연도별 수익률 표 생성 중...")
        yearly_returns_df = all_returns.resample('Y').apply(qs.stats.comp).T
        yearly_returns_df.columns = yearly_returns_df.columns.strftime('%Y')
        logger.info(f"[generate_performance_tables] yearly_returns_df shape={yearly_returns_df.shape}")
        yearly_path = os.path.join(OUTPUT_DIR, 'table_2_annual_returns.csv')
        yearly_returns_df.to_csv(yearly_path)
        _save_df_as_image(yearly_returns_df, 'table_2_annual_returns.png')
        logger.info(f"성공: 연도별 수익률 표 저장 완료.")
    except Exception as e:
        logger.error(f"실패: 연도별 수익률 표 생성 실패: {e}")
        logger.error(traceback.format_exc())
    try:
        logger.info("최악 낙폭 기간 표 생성 중...")
        drawdown_df = qs.stats.drawdown_details(strategy_returns)
        if not drawdown_df.empty:
            logger.info(f"[generate_performance_tables] drawdown_df shape={drawdown_df.shape}")
            drawdown_path = os.path.join(OUTPUT_DIR, 'table_3_drawdown_analysis.csv')
            drawdown_df.to_csv(drawdown_path)
            _save_df_as_image(drawdown_df, 'table_3_drawdown_analysis.png')
            logger.info(f"성공: 최악 낙폭 기간 표 저장 완료.")
        else:
            logger.info("낙폭 기간 데이터가 없어 표를 생성하지 않음.")
    except Exception as e:
        logger.error(f"실패: 최악 낙폭 기간 표 생성 실패: {e}")
        logger.error(traceback.format_exc())
    logger.info("[generate_performance_tables] Function exit.")


def generate_factor_analysis_table(strategy_returns, ff_df):
    logger.info("[generate_factor_analysis_table] Function entry.")
    logger.info(f"[generate_factor_analysis_table] Input: strategy_returns shape={strategy_returns.shape}, ff_df shape={ff_df.shape}")
    logger.info("파마-프렌치 3요인 회귀분석을 시작합니다.")
    if strategy_returns.empty:
        logger.warning("회귀분석을 할 수 없음: 입력 데이터가 비어있음.")
        logger.info("[generate_factor_analysis_table] Function exit (empty strategy_returns).")
        return
    try:
        ff_df = ff_df.set_index('date')
        merged_data = pd.merge(strategy_returns, ff_df, left_index=True, right_index=True, how='inner')
        logger.info(f"[generate_factor_analysis_table] merged_data shape={merged_data.shape}")
        if merged_data.empty or len(merged_data) < 2:
            logger.warning("회귀분석 데이터 부족으로 실행 중단.")
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
        logger.info(f"성공: 파마-프렌치 회귀분석 결과 표 저장 완료.")
    except Exception as e:
        logger.error(f"실패: 파마-프렌치 회귀분석 실패: {e}")
        logger.error(traceback.format_exc())
    logger.info("[generate_factor_analysis_table] Function exit.")

def plot_cumulative_returns(cumulative_df):
    logger.info("[plot_cumulative_returns] Function entry.")
    logger.info(f"[plot_cumulative_returns] Input: cumulative_df shape={cumulative_df.shape}")
    logger.info("누적 수익률 그래프를 생성합니다.")
    if cumulative_df.empty:
        logger.warning("누적 수익률 그래프를 생성할 수 없음: 입력 데이터가 비어있음.")
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
        logger.info(f"성공: 누적 수익률 그래프 저장 완료.")
    except Exception as e:
        logger.error(f"실패: 누적 수익률 그래프 생성 실패: {e}")
        logger.error(traceback.format_exc())
    logger.info("[plot_cumulative_returns] Function exit.")

def plot_underwater(strategy_returns):
    logger.info("[plot_underwater] Function entry.")
    logger.info(f"[plot_underwater] Input: strategy_returns shape={strategy_returns.shape}")
    logger.info("수중 그래프(Underwater Plot)를 생성합니다.")
    if strategy_returns.empty:
        logger.warning("수중 그래프를 생성할 수 없음: 입력 데이터가 비어있음.")
        logger.info("[plot_underwater] Function exit (empty strategy_returns).")
        return
    try:
        fig = qs.plots.drawdown(strategy_returns, 
                                savefig=os.path.join(OUTPUT_DIR, 'plot_2_underwater.png'))
        plt.close(fig)
        logger.info(f"성공: 수중 그래프 저장 완료.")
    except Exception as e:
        logger.error(f"실패: 수중 그래프 생성 실패: {e}")
        logger.error(traceback.format_exc())
    logger.info("[plot_underwater] Function exit.")

def plot_additional_analytics(strategy_returns):
    logger.info("[plot_additional_analytics] Function entry.")
    logger.info(f"[plot_additional_analytics] Input: strategy_returns shape={strategy_returns.shape}")
    logger.info("추가 분석 그래프(월별 수익률, 롤링 통계) 생성을 시작합니다.")
    if strategy_returns.empty:
        logger.warning("추가 분석 그래프를 생성할 수 없음: 입력 데이터가 비어있음.")
        logger.info("[plot_additional_analytics] Function exit (empty strategy_returns).")
        return
    try:
        fig = qs.plots.monthly_returns(strategy_returns, 
                                       savefig=os.path.join(OUTPUT_DIR, 'plot_3_monthly_returns.png'))
        plt.close(fig)
        logger.info(f"성공: 월별 수익률 그래프 저장 완료.")
    except Exception as e:
        logger.error(f"실패: 월별 수익률 그래프 생성 실패: {e}")
        logger.error(traceback.format_exc())
    
    ROLLING_WINDOW = 12
    logger.info(f"[plot_additional_analytics] strategy_returns length: {len(strategy_returns)}, ROLLING_WINDOW: {ROLLING_WINDOW}")
    if len(strategy_returns) > ROLLING_WINDOW:
        try:
            fig = qs.plots.rolling_volatility(strategy_returns, 
                                              savefig=os.path.join(OUTPUT_DIR, 'plot_4_rolling_volatility.png'))
            plt.close(fig)
            logger.info(f"성공: 롤링 변동성 그래프 저장 완료.")
        except Exception as e:
            logger.error(f"실패: 롤링 변동성 그래프 생성 실패: {e}")
            logger.error(traceback.format_exc())
        try:
            fig = qs.plots.rolling_sharpe(strategy_returns, 
                                          savefig=os.path.join(OUTPUT_DIR, 'plot_5_rolling_sharpe.png'))
            plt.close(fig)
            logger.info(f"성공: 롤링 샤프 지수 그래프 저장 완료.")
        except Exception as e:
            logger.error(f"실패: 롤링 샤프 지수 그래프 생성 실패: {e}")
            logger.error(traceback.format_exc())
    else:
        logger.warning(f"롤링 통계 그래프 생성 중단: 데이터 기간({len(strategy_returns)}개월)이 롤링 기간({ROLLING_WINDOW}개월)보다 짧습니다.")
    logger.info("[plot_additional_analytics] Function exit.")

def plot_monthly_returns_comparison(returns_df):
    logger.info("[plot_monthly_returns_comparison] Function entry.")
    logger.info(f"[plot_monthly_returns_comparison] Input: returns_df shape={returns_df.shape}")
    logger.info("월별 수익률 비교 그래프 생성을 시작합니다.")
    if returns_df.empty:
        logger.warning("월별 수익률 비교 그래프를 생성할 수 없음: 입력 데이터가 비어있음.")
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
        logger.info(f"성공: 월별 수익률 비교 그래프 저장 완료.")
    except Exception as e:
        logger.error(f"실패: 월별 수익률 비교 그래프 생성 실패: {e}")
        logger.error(traceback.format_exc())
    logger.info("[plot_monthly_returns_comparison] Function exit.")

def run_visualization(cumulative_df, ff_df):
    logger.info("[run_visualization] Function entry.")
    logger.info(f"[run_visualization] Input: cumulative_df shape={cumulative_df.shape}, ff_df shape={ff_df.shape}")
    logger.info("시각화 및 분석 프로세스 시작")
    if cumulative_df.empty:
        logger.warning("시각화 프로세스 중단: 누적 수익률 데이터가 비어있음.")
        logger.info("[run_visualization] Function exit (empty cumulative_df).")
        return
    returns_df = _calculate_returns_from_cumulative(cumulative_df)
    strategy_col = 'BL_ML_Strategy'
    strategy_returns = returns_df[strategy_col]
    benchmark_returns = returns_df.drop(columns=[strategy_col])
    logger.info(f"[run_visualization] returns_df shape={returns_df.shape}, strategy_returns shape={strategy_returns.shape}, benchmark_returns shape={benchmark_returns.shape}")
    generate_performance_tables(strategy_returns, benchmark_returns)
    generate_factor_analysis_table(strategy_returns, ff_df)
    plot_cumulative_returns(cumulative_df)
    plot_underwater(strategy_returns)
    plot_additional_analytics(strategy_returns)
    plot_monthly_returns_comparison(returns_df)
    logger.info("시각화 및 분석 프로세스 완료")
    logger.info("[run_visualization] Function exit.")

if __name__ == '__main__':
    # This block is for standalone testing of the visualizer
    try:
        cumulative_returns_path = os.path.join(OUTPUT_DIR, 'cumulative_returns.csv')
        cumulative_df = pd.read_csv(cumulative_returns_path, index_col=0, parse_dates=True)
        _, _, _, ff_df, _ = data_manager.load_raw_data()
        run_visualization(cumulative_df, ff_df)
    except FileNotFoundError:
        logger.error("visualizer.py를 단독으로 실행하려면 먼저 백테스트를 실행하여 'cumulative_returns.csv'를 생성해야 합니다.")
    except Exception as e:
        logger.error(f"visualizer.py 단독 실행 중 오류 발생: {e}")
        logger.error(traceback.format_exc())