from src.backtesting import backtester
import warnings
from config import settings as config
from src.utils import logger as logger_setup 
warnings.simplefilter(action='ignore', category=FutureWarning)

if __name__ == '__main__':
    """
    메인 진입점(Entry Point).

    - 전체 백테스팅 프로세스를 시작하고 조율하는 최상위 컨트롤러.
    - `config.py`의 모든 설정을 불러와 백테스터에 전달.
    """
    # 1. 백테스팅 실행
    logger_setup.logger.info("백테스트 실행 시작")
    ff_df = backtester.run_backtest(
        start_year=config.START_YEAR, 
        end_year=config.END_YEAR,
        etf_costs=config.ETF_COSTS,
        asset_groups=config.ASSET_GROUPS,
        group_constraints=config.GROUP_CONSTRAINTS,
        model_params=config.MODEL_PARAMS,
        benchmark_tickers=config.BENCHMARK_TICKERS,
        use_etf_ranking=config.USE_ETF_RANKING,
        top_n=config.TOP_N_ETFS
    )
    logger_setup.logger.info("백테스트 실행 완료")

    # 2. 결과 시각화
    logger_setup.logger.info("\n결과 시각화 시작")
    try:
        import os
        import pandas as pd
        cumulative_returns_path = os.path.join(config.OUTPUT_DIR, 'cumulative_returns.csv')
        cumulative_df = pd.read_csv(cumulative_returns_path, index_col=0, parse_dates=True)
        from src.visualization.plot import run_visualization
        run_visualization(cumulative_df, ff_df)
    except FileNotFoundError:
        logger_setup.logger.error(f"시각화를 위한 'cumulative_returns.csv' 파일을 찾을 수 없습니다.")
    except Exception as e:
        logger_setup.logger.error(f"시각화 실행 중 오류 발생: {e}")
    logger_setup.logger.info("결과 시각화 완료")
    