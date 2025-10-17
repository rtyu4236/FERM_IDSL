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

def _filter_config_by_permnos(all_available_permnos, etf_costs, benchmark_permnos):
    logger.info(f"[_filter_config_by_permnos] Filtering configs for {len(all_available_permnos)} permnos.")
    available_set = set(all_available_permnos)
    etf_costs_int_keys = {int(k): v for k, v in etf_costs.items()}
    filtered_costs = {p: c for p, c in etf_costs_int_keys.items() if p in available_set}
    
    for permno in available_set:
        if permno not in filtered_costs:
            logger.warning(f"Cost information for PERMNO '{permno}' not found in config.ETF_COSTS. Using default value 0.")
            filtered_costs[permno] = {'expense_ratio': 0.0, 'trading_cost_spread': 0.0}
    
    filtered_benchmark_permnos = [p for p in benchmark_permnos if p is None or p in available_set]
    return filtered_costs, filtered_benchmark_permnos

def create_one_over_n_benchmark_investable(monthly_df, target_dates, investable_permnos):
    try:
        investable_data = monthly_df[monthly_df['permno'].isin(investable_permnos)]
        if investable_data.empty:
            return None
        returns_pivot = investable_data.pivot_table(index='date', columns='permno', values='total_return')
        available_dates = returns_pivot.index.intersection(target_dates)
        returns_pivot = returns_pivot.loc[available_dates]
        one_over_n_returns = [(returns_pivot.loc[date].dropna() * (1.0 / len(returns_pivot.loc[date].dropna()))).sum() if date in returns_pivot.index and not returns_pivot.loc[date].dropna().empty else 0.0 for date in target_dates]
        return pd.Series(one_over_n_returns, index=target_dates, name='1/N Portfolio')
    except Exception as e:
        logger.error(f"[create_one_over_n_benchmark_investable] Failed to create 1/N portfolio: {e}")
        return None

def run_backtest(daily_df, monthly_df, vix_df, ff_df, all_permnos, start_year, end_year, etf_costs, model_params, benchmark_permnos, use_etf_ranking, top_n, run_rolling_tune, tune_trials):
    logger.info("[run_backtest] Function entry.")
    qs.extend_pandas()

    ml_features_df = data_manager.create_daily_feature_dataset_for_tcn(daily_df, vix_df, ff_df)
    logger.info(f"[run_backtest] ML features created: ml_features_df shape={ml_features_df.shape}")

    ranker = None
    if use_etf_ranking:
        logger.info("Initializing ETFQuantRanker...")
        ranker = ETFQuantRanker()
        logger.info("ETFQuantRanker initialized.")

    all_dates_in_df = pd.to_datetime(ml_features_df['date'].unique()).normalize()
    backtest_dates_in_range = all_dates_in_df[
        (all_dates_in_df.year >= start_year) & (all_dates_in_df.year <= end_year)
    ]
    calendar_month_ends = backtest_dates_in_range.to_period('M').unique().to_timestamp(how='end')
    backtest_dates = []
    all_dates_set = set(all_dates_in_df)
    for date in calendar_month_ends:
        current_date = date.normalize()
        # 월말이 데이터에 없을 경우, 가장 가까운 과거의 날짜를 찾음
        loop_count = 0
        while current_date not in all_dates_set:
            current_date -= pd.DateOffset(days=1)
            loop_count += 1
            if loop_count > 365:  # 무한 루프 방지
                logger.warning(f"Could not find a valid date for month end {date.strftime('%Y-%m')}. Skipping this month.")
                current_date = None
                break
        if current_date:
            backtest_dates.append(current_date)
    backtest_dates = pd.DatetimeIndex(backtest_dates)
    
    logger.info(f"[run_backtest] Backtest dates range: {backtest_dates.min()} to {backtest_dates.max()}, total {len(backtest_dates)} dates.")
    
    bl_returns = []
    previous_weights = pd.Series(dtype=float)
    monthly_turnovers = [] # 월별 회전율을 저장할 리스트 추가

    for analysis_date in backtest_dates[:-1]:
        logger.info(f"\n--- Processing {analysis_date.strftime('%Y-%m')} ---")

        if use_etf_ranking and ranker:
            logger.info(f"Ranking ETFs for {analysis_date.strftime('%Y-%m')}...")
            universe_for_month = ranker.get_top_permnos(str(analysis_date.date()), daily_df, all_permnos, top_n=top_n)
        else:
            universe_for_month = all_permnos
        logger.info(f"Universe for {analysis_date.strftime('%Y-%m')}: {len(universe_for_month)} permnos")

        if not universe_for_month:
            logger.warning(f"Universe for {analysis_date.strftime('%Y-%m')} is empty. Skipping month.")
            bl_returns.append(pd.Series([0.0], index=[analysis_date + pd.offsets.MonthEnd(1)]))
            continue

        if run_rolling_tune and model_params.get('use_tcn_svr', False):
            logger.info(f"Running hyperparameter tuning for {analysis_date.strftime('%Y-%m')}...")
            tcn_svr_tuner.run_tuning(ml_features_df, n_trials=tune_trials, end_date=analysis_date)
            current_model_params = config.get_model_params()
            active_model_params = current_model_params['tcn_svr_params']
            logger.info(f"Tuned parameters for {analysis_date.strftime('%Y-%m')}:\n{json.dumps(current_model_params, indent=2)}")
        else:
            current_model_params = model_params
            active_model_params = current_model_params.get('tcn_svr_params')

        filtered_costs, _ = _filter_config_by_permnos(
            universe_for_month, etf_costs, benchmark_permnos
        )
        monthly_df_filtered = monthly_df[monthly_df['permno'].isin(universe_for_month)]

        bl_portfolio_model = BlackLittermanPortfolio(
            all_returns_df=monthly_df_filtered,
            ff_df=ff_df,
            expense_ratios=filtered_costs,
            lookback_months=active_model_params.get('lookback_window', 24),
            tau=active_model_params.get('tau'),
            market_proxy_permno=current_model_params['market_proxy_permno']
        )

        current_permnos, returns_pivot = bl_portfolio_model._get_current_universe(analysis_date)
        if not current_permnos:
            logger.warning(f"No viable permnos for {analysis_date.strftime('%Y-%m-%d')}. Skipping.")
            weights = pd.Series(dtype=float)
        else:
            Sigma, delta, W_mkt = bl_portfolio_model._calculate_inputs(returns_pivot, analysis_date)
            if Sigma is None:
                weights = pd.Series(np.ones(len(current_permnos)) / len(current_permnos), index=current_permnos)
            else:
                P, Q, Omega = ml_view_generator.generate_tcn_svr_views(
                    analysis_date=analysis_date, 
                    permnos=current_permnos, 
                    full_feature_df=ml_features_df[ml_features_df['permno'].isin(current_permnos)],
                    model_params=active_model_params
                )
                weights, _ = bl_portfolio_model.get_black_litterman_portfolio(
                    analysis_date=analysis_date, P=P, Q=Q, Omega=Omega, 
                    pre_calculated_inputs=(Sigma, delta, W_mkt), max_weight=current_model_params['max_weight'],
                    previous_weights=previous_weights
                )
        
        next_month_date = analysis_date + pd.offsets.MonthEnd(1)
        next_month_returns = monthly_df[monthly_df['date'] == next_month_date]
        net_bl_return = 0.0
        if not next_month_returns.empty and weights is not None and not weights.empty:
            merged_bl = pd.merge(weights.to_frame('weight'), next_month_returns, left_index=True, right_on='permno')
            raw_bl_return = (merged_bl['weight'] * merged_bl['total_return']).sum()
            holding_costs = (merged_bl['weight'] * merged_bl['permno'].map(lambda p: filtered_costs.get(p, {}).get('expense_ratio', 0) / 12)).sum()
            aligned_prev, aligned_new = previous_weights.align(weights, join='outer', fill_value=0)
            trade_cost = (np.abs(aligned_new - aligned_prev) * aligned_new.index.map(lambda p: filtered_costs.get(p, {}).get('trading_cost_spread', 0.0001))).sum()
            net_bl_return = raw_bl_return - holding_costs - trade_cost

            turnover = np.abs(aligned_new - aligned_prev).sum() / 2
            monthly_turnovers.append(turnover)

        bl_returns.append(pd.Series([net_bl_return], index=[next_month_date]))
        previous_weights = weights

    avg_turnover = np.mean(monthly_turnovers) if monthly_turnovers else 0
    logger.info(f"Average Monthly Turnover: {avg_turnover:.2%}")

    logger.info("\nGenerating backtest analysis and report.")
    bl_returns_series = pd.concat(bl_returns).sort_index().squeeze().rename("BL_ML_Strategy")
    
    _, filtered_benchmarks = _filter_config_by_permnos(all_permnos, etf_costs, benchmark_permnos)
    benchmark_returns_dict = {}
    valid_benchmark_permnos = [p for p in filtered_benchmarks if p is not None]
    for permno in valid_benchmark_permnos:
        permno_returns_df = monthly_df[monthly_df['permno'] == permno]
        # 동일 날짜에 중복된 데이터가 있을 경우 첫 번째 값을 사용
        permno_returns_df = permno_returns_df.drop_duplicates(subset='date', keep='first')
        permno_returns = permno_returns_df.set_index('date')['total_return']
        benchmark_returns_dict[permno] = permno_returns.rename(permno)
    
    if None in filtered_benchmarks:
        one_over_n_returns = create_one_over_n_benchmark_investable(monthly_df, bl_returns_series.index, all_permnos)
        if one_over_n_returns is not None:
            benchmark_returns_dict['1/N Portfolio'] = one_over_n_returns
    
    results_df = pd.DataFrame({'BL_ML_Strategy': bl_returns_series, **benchmark_returns_dict})

    logger.info("--- Monthly Returns for All Strategies (results_df) ---")
    logger.info(results_df.to_string())
    logger.info("--- End of Monthly Returns ---")

    cumulative_results_df = (1 + results_df).cumprod()
    cum_results_path = os.path.join(config.OUTPUT_DIR, logger.LOG_NAME, 'cumulative_returns.csv')
    cumulative_results_df.to_csv(cum_results_path)
    logger.info(f"Cumulative returns data saved successfully: {cum_results_path}")
    
    final_returns = cumulative_results_df.iloc[-1] - 1
    logger.info("Final cumulative returns: " + ', '.join([f'{idx} {val:.4f}' for idx, val in final_returns.items()]))
    logger.info("[run_backtest] Function exit.")
    return ff_df, avg_turnover
