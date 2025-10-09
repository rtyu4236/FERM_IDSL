import pandas as pd
import quantstats as qs
import statsmodels.api as sm
import numpy as np
import os
import config
import data_manager
from logger_setup import logger
import matplotlib.pyplot as plt
import traceback

OUTPUT_DIR = config.OUTPUT_DIR

def _save_df_as_image(df, filename):
    """Pandas DataFrame을 이미지 파일로 저장."""
    logger.info(f"Attempting to save dataframe to {filename}")
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
    """누적 수익률 시리즈에서 월별 수익률 계산."""
    return cumulative_returns.pct_change().fillna(0)

def generate_performance_tables(strategy_returns, benchmark_returns):
    """주요 성과 지표, 연도별 수익률, 최악의 낙폭 기간 표 생성 및 저장."""
    logger.info("성과 분석 표 생성을 시작합니다.")
    if strategy_returns.empty:
        logger.warning("성과 분석 표를 생성할 수 없음: 입력 데이터가 비어있음.")
        return
    all_returns = pd.concat([strategy_returns, benchmark_returns], axis=1)
    try:
        logger.info("주요 성과 지표 표 생성 중...")
        metrics_df = qs.reports.metrics(all_returns, mode='full')
        if metrics_df is not None:
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
        # 수동으로 연도별 수익률 계산 (라이브러리 버전 문제 우회)
        yearly_returns_df = all_returns.resample('Y').apply(qs.stats.comp).T
        yearly_returns_df.columns = yearly_returns_df.columns.strftime('%Y')
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
            drawdown_path = os.path.join(OUTPUT_DIR, 'table_3_drawdown_analysis.csv')
            drawdown_df.to_csv(drawdown_path)
            _save_df_as_image(drawdown_df, 'table_3_drawdown_analysis.png')
            logger.info(f"성공: 최악 낙폭 기간 표 저장 완료.")
        else:
            logger.info("낙폭 기간 데이터가 없어 표를 생성하지 않음.")
    except Exception as e:
        logger.error(f"실패: 최악 낙폭 기간 표 생성 실패: {e}")
        logger.error(traceback.format_exc())


def generate_factor_analysis_table(strategy_returns, ff_df):
    """파마-프렌치 3요인 모델 회귀분석 수행 및 결과 표 저장."""
    logger.info("파마-프렌치 3요인 회귀분석을 시작합니다.")
    if strategy_returns.empty:
        logger.warning("회귀분석을 할 수 없음: 입력 데이터가 비어있음.")
        return
    try:
        ff_df = ff_df.set_index('date')
        merged_data = pd.merge(strategy_returns, ff_df, left_index=True, right_index=True, how='inner')
        if merged_data.empty or len(merged_data) < 2:
            logger.warning("회귀분석 데이터 부족으로 실행 중단.")
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
        #_save_df_as_image(results_df, 'table_4_fama_french_regression.png')
        logger.info(f"성공: 파마-프렌치 회귀분석 결과 표 저장 완료.")
    except Exception as e:
        logger.error(f"실패: 파마-프렌치 회귀분석 실패: {e}")
        logger.error(traceback.format_exc())

def plot_cumulative_returns(cumulative_df):
    """누적 수익률 그래프 생성 및 저장."""
    logger.info("누적 수익률 그래프를 생성합니다.")
    if cumulative_df.empty:
        logger.warning("누적 수익률 그래프를 생성할 수 없음: 입력 데이터가 비어있음.")
        return
    try:
        strategy_col = cumulative_df.columns[0]
        returns_for_plot = _calculate_returns_from_cumulative(cumulative_df[strategy_col])
        benchmarks_for_plot = _calculate_returns_from_cumulative(cumulative_df.drop(columns=strategy_col))
        fig = qs.plots.returns(returns_for_plot, benchmark=benchmarks_for_plot, 
                               savefig=os.path.join(OUTPUT_DIR, 'plot_1_cumulative_returns.png'))
        plt.close(fig)
        logger.info(f"성공: 누적 수익률 그래프 저장 완료.")
    except Exception as e:
        logger.error(f"실패: 누적 수익률 그래프 생성 실패: {e}")
        logger.error(traceback.format_exc())

def plot_underwater(strategy_returns):
    """수중 그래프(Underwater Plot) 생성 및 저장."""
    logger.info("수중 그래프(Underwater Plot)를 생성합니다.")
    if strategy_returns.empty:
        logger.warning("수중 그래프를 생성할 수 없음: 입력 데이터가 비어있음.")
        return
    try:
        fig = qs.plots.drawdown(strategy_returns, 
                                savefig=os.path.join(OUTPUT_DIR, 'plot_2_underwater.png'))
        plt.close(fig)
        logger.info(f"성공: 수중 그래프 저장 완료.")
    except Exception as e:
        logger.error(f"실패: 수중 그래프 생성 실패: {e}")
        logger.error(traceback.format_exc())

def plot_additional_analytics(strategy_returns):
    """월별 수익률, 롤링 변동성, 롤링 샤프 지수 등 추가 분석 그래프 생성 및 저장."""
    logger.info("추가 분석 그래프(월별 수익률, 롤링 통계) 생성을 시작합니다.")
    if strategy_returns.empty:
        logger.warning("추가 분석 그래프를 생성할 수 없음: 입력 데이터가 비어있음.")
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

def plot_monthly_returns_comparison(returns_df):
    """전략과 벤치마크의 월별 수익률 비교 라인 그래프 생성."""
    logger.info("월별 수익률 비교 그래프 생성을 시작합니다.")
    if returns_df.empty:
        logger.warning("월별 수익률 비교 그래프를 생성할 수 없음: 입력 데이터가 비어있음.")
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

def run_visualization(cumulative_df, ff_df):
    """전체 시각화 및 분석 프로세스를 실행하는 메인 함수."""
    logger.info("시각화 및 분석 프로세스 시작")
    if cumulative_df.empty:
        logger.warning("시각화 프로세스 중단: 누적 수익률 데이터가 비어있음.")
        return
    returns_df = _calculate_returns_from_cumulative(cumulative_df)
    strategy_col = 'BL_ML_Strategy'
    strategy_returns = returns_df[strategy_col]
    benchmark_returns = returns_df.drop(columns=[strategy_col])
    generate_performance_tables(strategy_returns, benchmark_returns)
    generate_factor_analysis_table(strategy_returns, ff_df)
    plot_cumulative_returns(cumulative_df)
    plot_underwater(strategy_returns)
    plot_additional_analytics(strategy_returns)
    plot_monthly_returns_comparison(returns_df)
    logger.info("시각화 및 분석 프로세스 완료")

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