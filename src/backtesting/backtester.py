import pandas as pd
import numpy as np
import os
import json
import traceback
from src.data_processing import manager as data_manager
from src.models import view_generator_new as ml_view_generator
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
    
    # Do not filter out benchmarks based on investable universe; keep as provided (None allowed)
    filtered_benchmark_permnos = benchmark_permnos
    return filtered_costs, filtered_benchmark_permnos

def create_one_over_n_benchmark_investable(monthly_df, target_dates, investable_permnos):
    logger.info(f"[create_one_over_n_benchmark_investable] Function entry. target_dates length: {len(target_dates)}")
    try:
        # Ensure datetime dtype
        if 'date' in monthly_df.columns and not pd.api.types.is_datetime64_any_dtype(monthly_df['date']):
            monthly_df = monthly_df.copy()
            monthly_df['date'] = pd.to_datetime(monthly_df['date'])

        investable_data = monthly_df[monthly_df['permno'].isin(investable_permnos)]
        if investable_data.empty:
            logger.warning("[create_one_over_n_benchmark_investable] Investable data is empty. Returning zero series.")
            return pd.Series(0.0, index=target_dates, name='1/N Portfolio')

        # Pre-sort for "last within month" selection
        investable_data = investable_data.sort_values(['permno', 'date'])

        one_over_n_returns = []
        for dt in target_dates:
            # Exact date first
            exact = investable_data[investable_data['date'] == dt]
            candidate = exact
            if exact.empty:
                # Fallback: same calendar month period, last available per permno
                target_period = pd.Timestamp(dt).to_period('M')
                period_mask = investable_data['date'].dt.to_period('M') == target_period
                period_df = investable_data[period_mask]
                if not period_df.empty:
                    candidate = period_df.groupby('permno', as_index=False).last()
            if candidate is not None and not candidate.empty:
                ret = candidate['total_return'].mean()
            else:
                ret = 0.0
            one_over_n_returns.append(ret)

        logger.info("[create_one_over_n_benchmark_investable] Successfully created 1/N portfolio series (with period fallback).")
        return pd.Series(one_over_n_returns, index=target_dates, name='1/N Portfolio')
    except Exception as e:
        logger.error(f"[create_one_over_n_benchmark_investable] Failed to create 1/N portfolio: {e}")
        logger.error(traceback.format_exc())
        return pd.Series(0.0, index=target_dates, name='1/N Portfolio')

def run_backtest(daily_df, monthly_df, vix_df, ff_df, all_permnos, start_year, end_year, etf_costs, model_params, benchmark_permnos, use_etf_ranking, top_n, run_rolling_tune, liquid_universe_dict=None):
    logger.info("[run_backtest] Function entry.")
    qs.extend_pandas()

    # Clear TCN warm start cache at the beginning of each backtest run
    if model_params.get('use_tcn_svr', False) and model_params.get('tcn_svr_params', {}).get('warm_start', False):
        logger.info("Clearing TCN warm start cache to ensure clean state...")
        ml_view_generator.clear_tcn_cache()

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
    # Ensure chronological order of backtest dates
    backtest_dates = pd.DatetimeIndex(backtest_dates).unique().sort_values()
    
    logger.info(f"[run_backtest] Backtest dates range: {backtest_dates.min()} to {backtest_dates.max()}, total {len(backtest_dates)} dates.")
    
    bl_returns = []
    topn_equal_weight_returns = [] if use_etf_ranking else None
    previous_weights = pd.Series(dtype=float)
    avg_turnover_dict = {'BL_ML_Strategy': 0.0} # Initialize with a default value
    monthly_turnovers = [] # 월별 회전율을 저장할 리스트 추가

    # For detailed exports
    weights_records = []  # [{'as_of': date, 'next_month': date, 'permno': int, 'weight': float}]
    month_stats_records = []  # [{'as_of': date, 'next_month': date, 'raw_bl_return': float, 'holding_costs': float, 'trade_cost': float, 'net_bl_return': float, 'turnover': float}]
    universe_per_month = {}  # {date_str: [permnos]}
    liquidity_log_records = []  # [{'as_of': date, 'before_count': int, 'after_count': int}]
    ranking_selected_per_month = {} if use_etf_ranking else None

    # liquid_universe_dict 인자 지원
    import inspect
    sig = inspect.signature(run_backtest)
    use_liquid_dict = 'liquid_universe_dict' in sig.parameters

    for idx, analysis_date in enumerate(backtest_dates[:-1]):
        logger.info(f"\n--- Processing {analysis_date.strftime('%Y-%m')} ---")

        # 1) 유동성 필터 전/후 개수 로깅 및 후보군 생성
        pre_liq_count = len(all_permnos)
        if liquid_universe_dict is not None:
            # Align analysis_date to its month-end to match dict keys and handle edge cases safely
            month_end_key = (pd.Timestamp(analysis_date) + pd.offsets.MonthEnd(0)).normalize()
            available_keys = sorted(liquid_universe_dict.keys())
            if month_end_key in liquid_universe_dict:
                key = month_end_key
            else:
                # Find the latest available key not after month_end_key; if none, fall back to earliest key
                prior_keys = [d for d in available_keys if d <= month_end_key]
                if prior_keys:
                    key = prior_keys[-1]
                else:
                    # analysis_date precedes the first available liquidity snapshot; use the earliest
                    key = available_keys[0]
            candidate_permnos = liquid_universe_dict.get(key, all_permnos)
        else:
            candidate_permnos = all_permnos
        after_liq_count = len(candidate_permnos)
        logger.info(f"[Rebalance {analysis_date.strftime('%Y-%m')}] Liquidity filter: before={pre_liq_count}, after={after_liq_count}")
        liquidity_log_records.append({
            'as_of': analysis_date.normalize(),
            'before_count': int(pre_liq_count),
            'after_count': int(after_liq_count)
        })

        # 2) 랭킹 적용 및 리스트 로깅
        if use_etf_ranking and ranker:
            if not candidate_permnos:
                ranked_permnos = []
            else:
                ranked_permnos = ranker.get_top_permnos(str(analysis_date.date()), daily_df, candidate_permnos, top_n=top_n)
            logger.info(f"[Rebalance {analysis_date.strftime('%Y-%m')}] Rank selected ({len(ranked_permnos)}): {ranked_permnos}")
            universe_for_month = ranked_permnos
            # save monthly ranking selection
            try:
                ranking_selected_per_month[analysis_date.strftime('%Y-%m-%d')] = list(map(int, ranked_permnos))
            except Exception:
                ranking_selected_per_month[analysis_date.strftime('%Y-%m-%d')] = ranked_permnos
        else:
            universe_for_month = candidate_permnos
            logger.info(f"[Rebalance {analysis_date.strftime('%Y-%m')}] Ranking disabled. Using candidate universe ({len(universe_for_month)}).")
        # Save selected universe for this month (as_of analysis_date)
        try:
            universe_per_month[analysis_date.strftime('%Y-%m-%d')] = list(map(int, universe_for_month))
        except Exception:
            universe_per_month[analysis_date.strftime('%Y-%m-%d')] = universe_for_month


        if not universe_for_month:
            logger.warning(f"Universe for {analysis_date.strftime('%Y-%m')} is empty. Skipping month.")
            bl_returns.append(pd.Series([0.0], index=[analysis_date + pd.offsets.MonthEnd(1)]))
            continue

       

        # Check tuning cadence (every K months)
        tune_every_k = model_params.get('tcn_svr_params', {}).get('tune_every_k_months', 1)
        month_index = (analysis_date.year - backtest_dates.min().year) * 12 + (analysis_date.month - backtest_dates.min().month)
        should_tune_this_month = (month_index % tune_every_k == 0)

        if run_rolling_tune and model_params.get('use_tcn_svr', False) and should_tune_this_month:
            logger.info(f"Running hyperparameter tuning for {analysis_date.strftime('%Y-%m')}...")
            tcn_svr_tuner.run_tuning(ml_features_df, n_trials=config.MODEL_PARAMS['tcn_svr_params']['tune_trials_per_month'], end_date=analysis_date)
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
                # Fallback: equal-weight across optimization universe if Sigma cannot be estimated
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
                # BL 최적화 실패 시 분석 대상 유니버스 대상 1/N 투자로 대체
                if weights is None or weights.empty or not np.isfinite(weights).all() or abs(weights.sum()) < 1e-8:
                    logger.warning("BL optimization failed or produced invalid weights; using equal-weight fallback across optimization universe.")
                    weights = pd.Series(np.ones(len(current_permnos)) / len(current_permnos), index=current_permnos)
        
        # Use the next backtest date as the next-month evaluation date to better align with available data
        next_month_date = backtest_dates[idx + 1]
        next_month_returns = monthly_df[monthly_df['date'] == next_month_date]
        # If exact match fails (calendar vs trading month-end mismatch), fallback to same calendar month period
        if next_month_returns.empty and 'date' in monthly_df.columns and pd.api.types.is_datetime64_any_dtype(monthly_df['date']):
            try:
                target_period = pd.Timestamp(next_month_date).to_period('M')
                period_mask = monthly_df['date'].dt.to_period('M') == target_period
                candidate = monthly_df[period_mask]
                if not candidate.empty:
                    # Use the last available row within the month per permno
                    candidate = candidate.sort_values('date').groupby('permno', as_index=False).last()
                    next_month_returns = candidate
                    logger.info(f"[Fallback] Using period-based monthly returns for {str(target_period)} due to exact date mismatch.")
            except Exception:
                pass
        net_bl_return = 0.0
        raw_bl_return = 0.0
        holding_costs = 0.0
        trade_cost = 0.0
        turnover = 0.0
        if not next_month_returns.empty and weights is not None and not weights.empty:
            logger.debug(f"[DEBUG] next_month_returns head:\n{next_month_returns.head().to_string()}")
            logger.debug(f"[DEBUG] weights head:\n{weights.head().to_string()}")
            merged_bl = pd.merge(weights.to_frame('weight'), next_month_returns, left_index=True, right_on='permno')
            logger.debug(f"[DEBUG] merged_bl head:\n{merged_bl.head().to_string()}")
            raw_bl_return = (merged_bl['weight'] * merged_bl['total_return']).sum()
            holding_costs = (merged_bl['weight'] * merged_bl['permno'].map(lambda p: filtered_costs.get(p, {}).get('expense_ratio', 0) / 12)).sum()
            aligned_prev, aligned_new = previous_weights.align(weights, join='outer', fill_value=0)
            trade_cost = (np.abs(aligned_new - aligned_prev) * aligned_new.index.map(lambda p: filtered_costs.get(p, {}).get('trading_cost_spread', 0.0001))).sum()
            net_bl_return = raw_bl_return - holding_costs - trade_cost
            turnover = np.abs(aligned_new - aligned_prev).sum() / 2
            monthly_turnovers.append(turnover)
            logger.info(f"[Rebalance {analysis_date.strftime('%Y-%m')}] Raw Return: {raw_bl_return:.4f}, Holding Costs: {holding_costs:.4f}, Trade Costs: {trade_cost:.4f}, Net Return: {net_bl_return:.4f}, Turnover: {turnover:.4f}")

            # Save weights detailed records
            for permno_idx, w in aligned_new.items():
                weights_records.append({
                    'as_of': analysis_date.normalize(),
                    'next_month': next_month_date.normalize(),
                    'permno': int(permno_idx) if isinstance(permno_idx, (int, np.integer)) else permno_idx,
                    'weight': float(w)
                })

        # Save per-month stats regardless
        month_stats_records.append({
            'as_of': analysis_date.normalize(),
            'next_month': next_month_date.normalize(),
            'raw_bl_return': float(raw_bl_return) if pd.notnull(raw_bl_return) else 0.0,
            'holding_costs': float(holding_costs) if pd.notnull(holding_costs) else 0.0,
            'trade_cost': float(trade_cost) if pd.notnull(trade_cost) else 0.0,
            'net_bl_return': float(net_bl_return) if pd.notnull(net_bl_return) else 0.0,
            'turnover': float(turnover) if pd.notnull(turnover) else 0.0
        })

        bl_returns.append(pd.Series([net_bl_return], index=[next_month_date]))
        previous_weights = weights

        # Compute TopN 1/N benchmark (equal weight on the universe_for_month), if ranking is enabled
        if use_etf_ranking and topn_equal_weight_returns is not None:
            if not next_month_returns.empty and universe_for_month:
                topn_next = next_month_returns[next_month_returns['permno'].isin(universe_for_month)]
                if not topn_next.empty:
                    ew_ret = topn_next['total_return'].mean()
                else:
                    ew_ret = 0.0
            else:
                ew_ret = 0.0
            topn_equal_weight_returns.append(pd.Series([ew_ret], index=[next_month_date]))

    avg_turnover = np.mean(monthly_turnovers) if monthly_turnovers else 0
    logger.info(f"Average Monthly Turnover: {avg_turnover:.2%}")

    logger.info("\nGenerating backtest analysis and report.")
    bl_returns_series = pd.concat(bl_returns).sort_index().squeeze().rename("BL_ML_Strategy")
    # Ensure unique index for bl_returns_series
    if not bl_returns_series.index.is_unique:
        logger.warning("Duplicate dates found in bl_returns_series index. Dropping duplicates.")
        bl_returns_series = bl_returns_series[~bl_returns_series.index.duplicated(keep='first')]
    
    _, filtered_benchmarks = _filter_config_by_permnos(all_permnos, etf_costs, benchmark_permnos)
    benchmark_returns_dict = {}
    valid_benchmark_permnos = [p for p in filtered_benchmarks if p is not None]
    for permno in valid_benchmark_permnos:
        permno_returns_df = monthly_df[monthly_df['permno'] == permno]
        if permno_returns_df.empty:
            logger.warning(f"Benchmark permno {permno} has no data; skipping.")
            continue
        # Ensure datetime dtype
        if not pd.api.types.is_datetime64_any_dtype(permno_returns_df['date']):
            permno_returns_df = permno_returns_df.copy()
            permno_returns_df['date'] = pd.to_datetime(permno_returns_df['date'])

        # Sort for last-within-month
        permno_returns_df = permno_returns_df.sort_values('date')

        # Build series aligned to strategy dates using exact-date-else-period fallback
        values = []
        for dt in bl_returns_series.index:
            exact = permno_returns_df[permno_returns_df['date'] == dt]
            if exact.empty:
                target_period = pd.Timestamp(dt).to_period('M')
                period_mask = permno_returns_df['date'].dt.to_period('M') == target_period
                period_df = permno_returns_df[period_mask]
                if not period_df.empty:
                    # take the last observation in that month
                    val = period_df.iloc[-1]['total_return']
                else:
                    val = np.nan
            else:
                val = exact.iloc[-1]['total_return']
            values.append(val)

        bench_series = pd.Series(values, index=bl_returns_series.index, name=str(permno))
        benchmark_returns_dict[permno] = bench_series
    
    # Overall 1/N benchmark across all investable permnos (unconditional)
    one_over_n_returns = create_one_over_n_benchmark_investable(monthly_df, bl_returns_series.index, all_permnos)
    if one_over_n_returns is not None:
        # Ensure unique index for 1/N Portfolio series
        if not one_over_n_returns.index.is_unique:
            logger.warning("Duplicate dates found in '1/N Portfolio' series index. Dropping duplicates.")
            one_over_n_returns = one_over_n_returns[~one_over_n_returns.index.duplicated(keep='first')]
        benchmark_returns_dict['1/N Portfolio'] = one_over_n_returns

    # Include TopN 1/N benchmark if created
    if use_etf_ranking and topn_equal_weight_returns is not None and len(topn_equal_weight_returns) > 0:
        topn_series = pd.concat(topn_equal_weight_returns).sort_index().squeeze().rename('TopN 1/N')
        # Align to strategy dates just in case
        topn_series = topn_series.reindex(bl_returns_series.index, fill_value=0.0)
        benchmark_returns_dict['TopN 1/N'] = topn_series
    
    results_df = pd.DataFrame({'BL_ML_Strategy': bl_returns_series, **benchmark_returns_dict})

    # Rename columns for clarity in output
    column_rename_map = {
        'BL_ML_Strategy': 'TCN-SVR',
        84398: 'SPY',  # Assuming 84398 is SPY
        88320: 'QQQ'   # Assuming 88320 is QQQ
    }
    results_df = results_df.rename(columns=column_rename_map)

    logger.info("--- Monthly Returns for All Strategies (results_df) ---")
    logger.info(results_df.to_string())
    logger.info("--- End of Monthly Returns ---")

    cumulative_results_df = (1 + results_df).cumprod()
    cum_results_path = os.path.join(config.OUTPUT_DIR, logger.LOG_NAME, 'cumulative_returns.csv')
    cumulative_results_df.to_csv(cum_results_path)
    logger.info(f"Cumulative returns data saved successfully: {cum_results_path}")

    # Save additional artifacts for robust downstream usage
    out_dir = os.path.join(config.OUTPUT_DIR, logger.LOG_NAME)
    try:
        # Monthly returns for all strategies
        monthly_returns_path = os.path.join(out_dir, 'monthly_returns.csv')
        results_df.to_csv(monthly_returns_path)
        logger.info(f"Monthly returns saved: {monthly_returns_path}")

        # Weights by month and permno
        if weights_records:
            weights_df = pd.DataFrame(weights_records)
            weights_df.to_csv(os.path.join(out_dir, 'weights.csv'), index=False)
            logger.info("Weights by month saved.")
        else:
            logger.info("No weights records to save (empty portfolio months).")

        # Costs and turnover per month
        stats_df = pd.DataFrame(month_stats_records)
        stats_df.to_csv(os.path.join(out_dir, 'per_month_stats.csv'), index=False)
        logger.info("Per-month stats saved.")

        # Universe per month
        with open(os.path.join(out_dir, 'universe_per_month.json'), 'w') as f:
            json.dump(universe_per_month, f)
        logger.info("Universe per month saved.")

        # Liquidity before/after per month
        pd.DataFrame(liquidity_log_records).to_csv(os.path.join(out_dir, 'liquidity_log.csv'), index=False)
        logger.info("Liquidity log saved.")

        # Ranking selection per month (if applicable)
        if ranking_selected_per_month is not None:
            with open(os.path.join(out_dir, 'ranking_selected_per_month.json'), 'w') as f:
                json.dump(ranking_selected_per_month, f)
            logger.info("Ranking selections per month saved.")

        # Save FF factors used for plotting
        ff_out = ff_df.copy()
        ff_out.to_csv(os.path.join(out_dir, 'fama_french_factors.csv'), index=False)

        # Avg turnover and metadata
        with open(os.path.join(out_dir, 'avg_turnover.json'), 'w') as f:
            json.dump({'BL_ML_Strategy': avg_turnover}, f)
        meta = {
            'start_year': start_year,
            'end_year': end_year,
            'use_etf_ranking': use_etf_ranking,
            'top_n': top_n,
            'log_name': logger.LOG_NAME
        }
        with open(os.path.join(out_dir, 'metadata.json'), 'w') as f:
            json.dump(meta, f)
        logger.info("Metadata saved.")
    except Exception as e:
        logger.error(f"Failed to save detailed artifacts: {e}")
    
    final_returns = cumulative_results_df.iloc[-1] - 1
    logger.info("Final cumulative returns: " + ', '.join([f'{idx} {val:.4f}' for idx, val in final_returns.items()]))
    logger.info("[run_backtest] Function exit.")
    return ff_df, avg_turnover, start_year, end_year
