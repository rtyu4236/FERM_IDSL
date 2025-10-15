import pandas as pd
import numpy as np
import os
import json
from src.data_processing import manager as data_manager
from src.models import view_generator as ml_view_generator
from src.models.black_litterman import BlackLittermanPortfolio
import quantstats as qs
from config import settings as config
from src.utils.logger import logger
from src.tuning import tcn_svr_tuner
from src.models.etf_quant_ranker import ETFQuantRanker, create_etf_universe_from_daily

def _filter_config_by_tickers(all_available_tickers, etf_costs, asset_groups, group_constraints, benchmark_tickers):
    logger.info(f"[_filter_config_by_tickers] Filtering configs for {len(all_available_tickers)} tickers.")
    available_set = set(all_available_tickers)
    filtered_costs = {t: c for t, c in etf_costs.items() if t in available_set}
    for ticker in available_set:
        if ticker not in filtered_costs:
            logger.warning(f"Cost information for '{ticker}' not found in config.ETF_COSTS. Using default value 0.")
            filtered_costs[ticker] = {'expense_ratio': 0.0, 'trading_cost_spread': 0.0}
    
    filtered_asset_groups = {}
    for group, tickers in asset_groups.items():
        filtered_tickers = [t for t in tickers if t in available_set]
        if filtered_tickers:
            filtered_asset_groups[group] = filtered_tickers

    filtered_benchmark_tickers = [t for t in benchmark_tickers if t is None or t in available_set]
    return filtered_costs, filtered_asset_groups, group_constraints, filtered_benchmark_tickers

def create_one_over_n_benchmark_investable(monthly_df, target_dates, investable_tickers):
    try:
        investable_data = monthly_df[monthly_df['ticker'].isin(investable_tickers)]
        if investable_data.empty:
            return None
        returns_pivot = investable_data.pivot_table(index='date', columns='ticker', values='retx')
        available_dates = returns_pivot.index.intersection(target_dates)
        returns_pivot = returns_pivot.loc[available_dates]
        one_over_n_returns = [(returns_pivot.loc[date].dropna() * (1.0 / len(returns_pivot.loc[date].dropna()))).sum() if date in returns_pivot.index and not returns_pivot.loc[date].dropna().empty else 0.0 for date in target_dates]
        return pd.Series(one_over_n_returns, index=target_dates, name='1/N Portfolio')
    except Exception as e:
        logger.error(f"[create_one_over_n_benchmark_investable] Failed to create 1/N portfolio: {e}")
        return None

def run_backtest(daily_df, monthly_df, vix_df, ff_df, all_tickers, start_year, end_year, etf_costs, asset_groups, group_constraints, model_params, benchmark_tickers, use_etf_ranking, top_n, run_rolling_tune, tune_trials):
    logger.info("[run_backtest] Function entry.")
    qs.extend_pandas()

    # 1. Create feature dataset (changed to daily dataset)
    ml_features_df = data_manager.create_daily_feature_dataset_for_tcn(daily_df, vix_df, ff_df)
    logger.info(f"[run_backtest] ML features created: ml_features_df shape={ml_features_df.shape}")

    # 2. Initialize ranking model (if necessary, only once)
    ranker = None
    if use_etf_ranking:
        logger.info("Initializing ETFQuantRanker...")
        ranker = ETFQuantRanker()
        logger.info("ETFQuantRanker initialized.")

    # Generate monthly rebalancing dates
    all_dates_in_df = pd.to_datetime(ml_features_df['date'].unique())
    backtest_dates_in_range = all_dates_in_df[
        (all_dates_in_df.year >= start_year) & (all_dates_in_df.year <= end_year)
    ]
    # Regenerate rebalancing dates based on month-end dates
    backtest_dates = pd.to_datetime(pd.Series(backtest_dates_in_range).dt.to_period('M').unique().to_timestamp(how='end'))
    
    logger.info(f"[run_backtest] Backtest dates range: {backtest_dates.min()} to {backtest_dates.max()}, total {len(backtest_dates)} dates.")
    
    bl_returns = []
    previous_weights = pd.Series(dtype=float)

    # 3. Monthly rebalancing loop
    for analysis_date in backtest_dates[:-1]:
        logger.info(f"\n--- Processing {analysis_date.strftime('%Y-%m')} ---")

        # 3.1. Select investment universe (monthly)
        if use_etf_ranking and ranker:
            logger.info(f"Ranking ETFs for {analysis_date.strftime('%Y-%m')}...")
            universe_for_month = ranker.get_top_tickers(str(analysis_date.date()), daily_df, all_tickers, top_n=top_n)
        else:
            universe_for_month = all_tickers # Use the pre-filtered liquid universe if ranking is off
        logger.info(f"Universe for {analysis_date.strftime('%Y-%m')}: {len(universe_for_month)} tickers")

        if not universe_for_month:
            logger.warning(f"Universe for {analysis_date.strftime('%Y-%m')} is empty. Skipping month.")
            bl_returns.append(pd.Series([0.0], index=[analysis_date + pd.offsets.MonthEnd(1)]))
            continue

        # 3.2. Hyperparameter tuning (monthly)
        if run_rolling_tune and model_params.get('use_tcn_svr', False):
            logger.info(f"Running hyperparameter tuning for {analysis_date.strftime('%Y-%m')}...")
            tcn_svr_tuner.run_tuning(ml_features_df, n_trials=tune_trials, end_date=analysis_date)
            current_model_params = config.get_model_params()
            active_model_params = current_model_params['tcn_svr_params']
            logger.info(f"Tuned parameters for {analysis_date.strftime('%Y-%m')}:\n{json.dumps(current_model_params, indent=2)}")
        else:
            current_model_params = model_params
            active_model_params = current_model_params.get('tcn_svr_params')

        # 3.3. Initialize Black-Litterman model (monthly)
        filtered_costs, filtered_asset_groups, filtered_group_constraints, _ = _filter_config_by_tickers(
            universe_for_month, etf_costs, asset_groups, group_constraints, benchmark_tickers
        )
        monthly_df_filtered = monthly_df[monthly_df['ticker'].isin(universe_for_month)]

        bl_portfolio_model = BlackLittermanPortfolio(
            all_returns_df=monthly_df_filtered,
            ff_df=ff_df,
            expense_ratios=filtered_costs,
            lookback_months=active_model_params.get('lookback_window', 24),
            tau=active_model_params.get('tau'),
            market_proxy_ticker=current_model_params['market_proxy_ticker'],
            asset_groups=filtered_asset_groups,
            group_constraints=filtered_group_constraints
        )

        # 3.4. Generate views and construct portfolio
        current_tickers, returns_pivot = bl_portfolio_model._get_current_universe(analysis_date)
        if not current_tickers:
            logger.warning(f"No viable tickers for {analysis_date.strftime('%Y-%m-%d')}. Skipping.")
            weights = pd.Series(dtype=float)
        else:
            Sigma, delta, W_mkt = bl_portfolio_model._calculate_inputs(returns_pivot, analysis_date)
            if Sigma is None:
                weights = pd.Series(np.ones(len(current_tickers)) / len(current_tickers), index=current_tickers)
            else:
                P, Q, Omega = ml_view_generator.generate_tcn_svr_views(
                    analysis_date=analysis_date, 
                    tickers=current_tickers, 
                    full_feature_df=ml_features_df[ml_features_df['ticker'].isin(current_tickers)],
                    model_params=active_model_params
                )
                weights, _ = bl_portfolio_model.get_black_litterman_portfolio(
                    analysis_date=analysis_date, P=P, Q=Q, Omega=Omega, 
                    pre_calculated_inputs=(Sigma, delta, W_mkt), max_weight=current_model_params['max_weight'],
                    previous_weights=previous_weights
                )
        
        # 3.5. Calculate returns
        next_month_date = analysis_date + pd.offsets.MonthEnd(1)
        next_month_returns = monthly_df[monthly_df['date'] == next_month_date]
        net_bl_return = 0.0
        if not next_month_returns.empty and weights is not None and not weights.empty:
            merged_bl = pd.merge(weights.to_frame('weight'), next_month_returns, left_index=True, right_on='ticker')
            raw_bl_return = (merged_bl['weight'] * merged_bl['retx']).sum()
            holding_costs = (merged_bl['weight'] * merged_bl['ticker'].map(lambda t: filtered_costs.get(t, {}).get('expense_ratio', 0) / 12)).sum()
            aligned_prev, aligned_new = previous_weights.align(weights, join='outer', fill_value=0)
            trade_cost = (np.abs(aligned_new - aligned_prev) * aligned_new.index.map(lambda t: filtered_costs.get(t, {}).get('trading_cost_spread', 0.0001))).sum()
            net_bl_return = raw_bl_return - holding_costs - trade_cost

        bl_returns.append(pd.Series([net_bl_return], index=[next_month_date]))
        previous_weights = weights

    # 4. Aggregate and save final results
    logger.info("\nGenerating backtest analysis and report.")
    bl_returns_series = pd.concat(bl_returns).sort_index().squeeze().rename("BL_ML_Strategy")
    
    _, _, _, filtered_benchmarks = _filter_config_by_tickers(all_tickers, etf_costs, asset_groups, group_constraints, benchmark_tickers)
    benchmark_returns_dict = {}
    valid_benchmark_tickers = [t for t in filtered_benchmarks if t is not None]
    for ticker in valid_benchmark_tickers:
        ticker_returns = monthly_df[monthly_df['ticker'] == ticker].set_index('date')['retx']
        benchmark_returns_dict[ticker] = ticker_returns.rename(ticker)
    
    if None in filtered_benchmarks:
        one_over_n_returns = create_one_over_n_benchmark_investable(monthly_df, bl_returns_series.index, all_tickers)
        if one_over_n_returns is not None:
            benchmark_returns_dict['1/N Portfolio'] = one_over_n_returns
    
    results_df = pd.DataFrame({'BL_ML_Strategy': bl_returns_series, **benchmark_returns_dict})
    cumulative_results_df = (1 + results_df).cumprod()
    cum_results_path = os.path.join(config.OUTPUT_DIR, 'cumulative_returns.csv')
    cumulative_results_df.to_csv(cum_results_path)
    logger.info(f"Cumulative returns data saved successfully: {cum_results_path}")
    
    final_returns = cumulative_results_df.iloc[-1] - 1
    logger.info("Final cumulative returns: " + ', '.join([f'{idx} {val:.4f}' for idx, val in final_returns.items()]))
    logger.info("[run_backtest] Function exit.")
    return ff_df