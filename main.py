import argparse
import json
import os
import warnings
import pandas as pd

from src.backtesting import backtester
from config import settings as config
from src.utils import logger as logger_setup
from src.data_processing import manager as data_manager
from src.models.etf_quant_ranker import ETFQuantRanker, create_etf_universe_from_daily

warnings.simplefilter(action='ignore', category=FutureWarning)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the backtesting and visualization process.")
    # 튜닝 비활성화 (--no-tune) 플래그, 기본적으로 튜닝 실행
    parser.add_argument('--no-tune', dest='tune', action='store_false', help="Skip the monthly rolling hyperparameter tuning.")
    # 랭킹 비활성화 (--no-ranking) 플래그, 기본적으로 랭킹 실행
    parser.add_argument('--no-ranking', dest='ranking', action='store_false', help="Skip the monthly ETF ranking and use all tickers.")
    parser.add_argument('--trials', type=int, default=50, help="Number of trials for hyperparameter tuning.")
    args = parser.parse_args()

    logger_setup.logger.info("main.py execution started.")
    logger_setup.logger.info(f"Arguments: tune={args.tune}, ranking={args.ranking}, trials={args.trials}")

    # 1. 전체 데이터 로드
    daily_df, monthly_df, vix_df, ff_df, all_tickers = data_manager.load_raw_data()

    # 2. 투자 유니버스 사전 필터링 (유동성 기준)
    if args.ranking:
        initial_universe = data_manager.filter_liquid_universe(
            daily_df=daily_df,
            all_tickers=all_tickers,
            start_year=config.START_YEAR
        )
    else:
        initial_universe = all_tickers

    # 3. 백테스터에 모든 데이터와 설정을 전달하여 실행
    # 랭킹 및 튜닝 로직은 이제 백테스터 내부에서 처리됨
    ff_df_from_backtest = backtester.run_backtest(
        # Dataframes
        daily_df=daily_df,
        monthly_df=monthly_df,
        vix_df=vix_df,
        ff_df=ff_df,
        all_tickers=initial_universe, # 수정: 필터링된 유니버스 전달
        # Configurations
        start_year=config.START_YEAR,
        end_year=config.END_YEAR,
        etf_costs=config.ETF_COSTS,
        asset_groups=config.ASSET_GROUPS,
        group_constraints=config.GROUP_CONSTRAINTS,
        model_params=config.MODEL_PARAMS,
        benchmark_tickers=config.BENCHMARK_TICKERS,
        use_etf_ranking=args.ranking,
        top_n=config.TOP_N_ETFS,
        run_rolling_tune=args.tune,
        tune_trials=args.trials
    )
    logger_setup.logger.info("backtester.run_backtest completed.")

    # 3. 결과 시각화
    logger_setup.logger.info("\nResult visualization started.")
    try:
        cumulative_returns_path = os.path.join(config.OUTPUT_DIR, 'cumulative_returns.csv')
        cumulative_df = pd.read_csv(cumulative_returns_path, index_col=0, parse_dates=True)
        from src.visualization.plot import run_visualization
        run_visualization(cumulative_df, ff_df_from_backtest)
    except FileNotFoundError:
        logger_setup.logger.error(f"Visualization error: 'cumulative_returns.csv' file not found.")
    except Exception as e:
        logger_setup.logger.error(f"Visualization error: {e}")
    logger_setup.logger.info("Result visualization completed.")
    logger_setup.logger.info("main.py execution finished.")
    