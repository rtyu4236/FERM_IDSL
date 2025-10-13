import argparse
import json
import os
import warnings
from src.backtesting import backtester
from config import settings as config
from src.utils import logger as logger_setup
warnings.simplefilter(action='ignore', category=FutureWarning)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the backtesting and visualization process.")
    parser.add_argument('--no-tune', dest='tune', action='store_false', help="Skip the yearly rolling hyperparameter tuning.")
    parser.add_argument('--trials', type=int, default=50, help="Number of trials for hyperparameter tuning.")
    args = parser.parse_args()

    logger_setup.logger.info("main.py execution started.")
    logger_setup.logger.info(f"Arguments: tune={args.tune}, trials={args.trials}")

    # Initial model parameters are loaded here. If tuning is enabled, 
    # the backtester will overwrite them each year.
    logger_setup.logger.info("Loading initial model parameters...")
    model_params = config.get_model_params(use_tuned_params=False) # Start with defaults
    
    logger_setup.logger.info("Calling backtester.run_backtest...")
    ff_df = backtester.run_backtest(
        start_year=config.START_YEAR,
        end_year=config.END_YEAR,
        etf_costs=config.ETF_COSTS,
        asset_groups=config.ASSET_GROUPS,
        group_constraints=config.GROUP_CONSTRAINTS,
        model_params=model_params,
        benchmark_tickers=config.BENCHMARK_TICKERS,
        run_rolling_tune=args.tune,
        tune_trials=args.trials
    )
    logger_setup.logger.info("backtester.run_backtest completed. ff_df type: %s, shape: %s", type(ff_df), ff_df.shape if hasattr(ff_df, 'shape') else 'N/A')

    logger_setup.logger.info("\nResult visualization started.")
    try:
        import pandas as pd
        cumulative_returns_path = os.path.join(config.OUTPUT_DIR, 'cumulative_returns.csv')
        cumulative_df = pd.read_csv(cumulative_returns_path, index_col=0, parse_dates=True)
        from src.visualization.plot import run_visualization
        run_visualization(cumulative_df, ff_df)
    except FileNotFoundError:
        logger_setup.logger.error(f"Visualization error: 'cumulative_returns.csv' file not found.")
    except Exception as e:
        logger_setup.logger.error(f"Visualization error: {e}")
    logger_setup.logger.info("Result visualization completed.")
    logger_setup.logger.info("main.py execution finished.")
