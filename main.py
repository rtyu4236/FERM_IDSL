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
    parser.add_argument('--trials', type=int, default=2, help="Number of trials for hyperparameter tuning.")
    args = parser.parse_args()

    logger_setup.logger.info("main.py execution started.")
    logger_setup.logger.info(f"Arguments: tune={args.tune}, ranking={args.ranking}, trials={args.trials}")

    # 1. Load all data
    daily_df, monthly_df, vix_df, ff_df, all_permnos = data_manager.load_raw_data()

    # 2. Pre-filter the investment universe
    initial_universe_permnos = all_permnos
    if args.ranking: # Only apply filtering if ranking is enabled
        # 2.1. Filter by liquidity
        logger_setup.logger.info("Starting universe pre-filtering based on liquidity...")
        liquid_permnos = data_manager.filter_liquid_universe(
            daily_df=daily_df,
            all_permnos=all_permnos,
            start_year=config.START_YEAR
        )
        
        # 2.2. Filter by minimum trading history (5 years)
        logger_setup.logger.info("Filtering universe by minimum history (5 years)...")
        filter_date = pd.to_datetime(f"{config.START_YEAR}-01-01")
        MIN_MONTHS = 5 * 12

        history_counts = monthly_df[
            (monthly_df['permno'].isin(liquid_permnos)) &
            (monthly_df['date'] < filter_date)
        ].groupby('permno').size()

        history_filtered_permnos = history_counts[history_counts >= MIN_MONTHS].index.tolist()
        logger_setup.logger.info(f"History filtering complete. {len(liquid_permnos)} permnos -> {len(history_filtered_permnos)} permnos.")
        
        initial_universe_permnos = history_filtered_permnos

    # 3. Pass all data and settings to the backtester and run
    ff_df_from_backtest, avg_turnover = backtester.run_backtest(
        daily_df=daily_df,
        monthly_df=monthly_df,
        vix_df=vix_df,
        ff_df=ff_df,
        all_permnos=initial_universe_permnos, # Pass the final filtered universe
        start_year=config.START_YEAR,
        end_year=config.END_YEAR,
        etf_costs=config.ETF_COSTS,
        model_params=config.MODEL_PARAMS,
        benchmark_permnos=config.BENCHMARK_PERMNOS,
        use_etf_ranking=args.ranking,
        top_n=config.TOP_N_ETFS,
        run_rolling_tune=args.tune,
        tune_trials=args.trials
    )
    logger_setup.logger.info("backtester.run_backtest completed.")

    # 4. Visualize results
    logger_setup.logger.info("\nResult visualization started.")
    try:
        cumulative_returns_path = os.path.join(config.OUTPUT_DIR, logger_setup.logger.LOG_NAME, 'cumulative_returns.csv')
        cumulative_df = pd.read_csv(cumulative_returns_path, index_col=0, parse_dates=True)
        from src.visualization.plot import run_visualization
        avg_turnover_dict = {'BL_ML_Strategy': avg_turnover}

        logger_setup.logger.info("--- [main.py] Data for Visualization ---")
        logger_setup.logger.info("1. cumulative_df:")
        logger_setup.logger.info(cumulative_df.to_string())
        logger_setup.logger.info("2. ff_df_from_backtest:")
        logger_setup.logger.info(ff_df_from_backtest.to_string())
        logger_setup.logger.info("3. avg_turnover_dict:")
        logger_setup.logger.info(avg_turnover_dict)
        logger_setup.logger.info("--- End of Data for Visualization ---")

        run_visualization(cumulative_df, ff_df_from_backtest, avg_turnover_dict)
    except FileNotFoundError:
        logger_setup.logger.error(f"Visualization error: 'cumulative_returns.csv' file not found.")
    except Exception as e:
        logger_setup.logger.error(f"Visualization error: {e}")
    logger_setup.logger.info("Result visualization completed.")
    logger_setup.logger.info("main.py execution finished.")