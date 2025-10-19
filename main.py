import argparse
import json
import os
import warnings
import pandas as pd

from src.backtesting import backtester
from config import settings as config
from src.utils import logger as logger_setup
from src.data_processing import manager as data_manager

warnings.simplefilter(action='ignore', category=FutureWarning)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the backtesting and visualization process.")
    parser.add_argument('--no-tune', dest='tune', action='store_false', help="Skip the monthly rolling hyperparameter tuning.")
    parser.add_argument('--no-ranking', dest='ranking', action='store_false', help="Skip the monthly ETF ranking and use all permnos.")
    args = parser.parse_args() # No --trials argument anymore

    logger_setup.logger.info("main.py execution started.")
    logger_setup.logger.info(f"Arguments: tune={args.tune}, ranking={args.ranking}, tune_trials_per_month={config.MODEL_PARAMS['tcn_svr_params']['tune_trials_per_month']}")

    # 1. Load all data
    daily_df, monthly_df, vix_df, ff_df, all_permnos = data_manager.load_raw_data()

    # 2. 월별 동적 유동성 필터 적용
    liquid_universe_dict = None
    initial_universe_permnos = all_permnos
    if args.ranking:
        # 월별 리밸런싱 날짜 생성 (월말)
        monthly_dates = pd.date_range(
            start=f"{config.START_YEAR}-01-01", end=f"{config.END_YEAR}-12-31", freq='M'
        )
        # 월별 유동성 필터 적용
        liquid_universe_dict = data_manager.filter_liquid_universe(
            daily_df=daily_df,
            all_permnos=all_permnos,
            monthly_dates=monthly_dates,
            min_avg_value=config.MODEL_PARAMS.get('min_avg_value', 1_000_000)
        )
        # 5년 이상 거래 이력 필터 (각 월별로 적용)
        logger_setup.logger.info("Filtering universe by minimum history (5 years) for each month...")
        MIN_MONTHS = 5 * 12
        for date in monthly_dates:
            filter_date = pd.to_datetime(date)
            permnos = liquid_universe_dict[filter_date]
            history_counts = monthly_df[
                (monthly_df['permno'].isin(permnos)) &
                (monthly_df['date'] < filter_date)
            ].groupby('permno').size()
            history_filtered_permnos = history_counts[history_counts >= MIN_MONTHS].index.tolist()
            liquid_universe_dict[filter_date] = history_filtered_permnos
        # 최초 universe는 첫 월의 permnos로 설정 (비상시)
        initial_universe_permnos = liquid_universe_dict[monthly_dates[0]]

    # 3. Pass all data and settings to the backtester and run
    ff_df_from_backtest, avg_turnover, start_year_backtest, end_year_backtest = backtester.run_backtest(
        daily_df=daily_df,
        monthly_df=monthly_df,
        vix_df=vix_df,
        ff_df=ff_df,
        all_permnos=initial_universe_permnos,
        start_year=config.START_YEAR,
        end_year=config.END_YEAR,
        etf_costs=config.ETF_COSTS,
        model_params=config.MODEL_PARAMS,
        benchmark_permnos=config.BENCHMARK_PERMNOS,
        use_etf_ranking=args.ranking,
        top_n=config.TOP_N_ETFS,
        run_rolling_tune=args.tune,
        liquid_universe_dict=liquid_universe_dict
    )
    logger_setup.logger.info("backtester.run_backtest completed.")

    # 4. Visualize results
    logger_setup.logger.info("\nResult visualization started.")
    try:
        cumulative_returns_path = os.path.join(config.OUTPUT_DIR, logger_setup.logger.LOG_NAME, 'cumulative_returns.csv')
        cumulative_df = pd.read_csv(cumulative_returns_path, index_col=0, parse_dates=True)
        from src.visualization.plot import run_visualization
        avg_turnover_dict = {'TCN-SVR': avg_turnover}

        logger_setup.logger.info("--- [main.py] Data for Visualization ---")
        logger_setup.logger.info("1. cumulative_df:")
        logger_setup.logger.info(cumulative_df.to_string())
        logger_setup.logger.info("2. ff_df_from_backtest:")
        logger_setup.logger.info(ff_df_from_backtest.to_string())
        logger_setup.logger.info("3. avg_turnover_dict:")
        logger_setup.logger.info(avg_turnover_dict)
        logger_setup.logger.info("--- End of Data for Visualization ---")

        run_visualization(cumulative_df, ff_df_from_backtest, avg_turnover_dict, start_year_backtest, end_year_backtest)
    except FileNotFoundError:
        logger_setup.logger.error(f"Visualization error: 'cumulative_returns.csv' file not found.")
    except Exception as e:
        logger_setup.logger.error(f"Visualization error: {e}")
    logger_setup.logger.info("Result visualization completed.")
    logger_setup.logger.info("main.py execution finished.")