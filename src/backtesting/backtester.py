import pandas as pd
import numpy as np
import os
from src.data_processing import manager as data_manager
from src.models import view_generator as ml_view_generator
from src.models.black_litterman import BlackLittermanPortfolio
import quantstats as qs
from config import settings as config
from src.utils.logger import logger
from src.tuning import tcn_svr_tuner

def _filter_config_by_tickers(all_available_tickers, etf_costs, asset_groups, group_constraints, benchmark_tickers):
    logger.info("[_filter_config_by_tickers] Function entry.")
    logger.info(f"[_filter_config_by_tickers] Input: all_available_tickers len={len(all_available_tickers)}, etf_costs len={len(etf_costs)}, asset_groups len={len(asset_groups)}, group_constraints len={len(group_constraints)}, benchmark_tickers len={len(benchmark_tickers)}")
    available_set = set(all_available_tickers)
    logger.info(f"설정 필터링 기준 티커 수: {len(available_set)}")

    filtered_costs = {t: c for t, c in etf_costs.items() if t in available_set}
    for ticker in available_set:
        if ticker not in filtered_costs:
            logger.warning(f"'{ticker}'에 대한 비용 정보가 config.ETF_COSTS에 없습니다. 기본값 0을 사용합니다.")
            filtered_costs[ticker] = {'expense_ratio': 0.0, 'trading_cost_spread': 0.0}
    
    filtered_asset_groups = {}
    for group, tickers in asset_groups.items():
        filtered_tickers = [t for t in tickers if t in available_set]
        if filtered_tickers:
            filtered_asset_groups[group] = filtered_tickers

    filtered_benchmark_tickers = [t for t in benchmark_tickers if t is None or t in available_set]
    
    logger.info("설정 필터링 완료")
    logger.info(f"[_filter_config_by_tickers] Output: filtered_costs len={len(filtered_costs)}, filtered_asset_groups len={len(filtered_asset_groups)}, filtered_benchmark_tickers len={len(filtered_benchmark_tickers)}")
    logger.info("[_filter_config_by_tickers] Function exit.")
    return filtered_costs, filtered_asset_groups, group_constraints, filtered_benchmark_tickers

def create_one_over_n_benchmark_investable(monthly_df, target_dates, investable_tickers):
    logger.info("[create_one_over_n_benchmark_investable] Function entry.")
    logger.info(f"[create_one_over_n_benchmark_investable] Input: monthly_df shape={monthly_df.shape}, target_dates len={len(target_dates)}, investable_tickers len={len(investable_tickers)}")
    try:
        logger.info(f"1/N 포트폴리오 투자 유니버스: {len(investable_tickers)}개 자산 {investable_tickers}")
        
        investable_data = monthly_df[monthly_df['ticker'].isin(investable_tickers)]
        logger.info(f"[create_one_over_n_benchmark_investable] investable_data shape: {investable_data.shape}")
        
        if investable_data.empty:
            logger.warning("투자 가능한 자산 데이터가 없습니다.")
            logger.info("[create_one_over_n_benchmark_investable] Function exit (empty data).")
            return None
        
        returns_pivot = investable_data.pivot_table(index='date', columns='ticker', values='retx')
        logger.info(f"[create_one_over_n_benchmark_investable] returns_pivot shape: {returns_pivot.shape}")
        available_dates = returns_pivot.index.intersection(target_dates)
        returns_pivot = returns_pivot.loc[available_dates]
        logger.info(f"[create_one_over_n_benchmark_investable] returns_pivot after date filter shape: {returns_pivot.shape}")
        
        one_over_n_returns = []
        for date in target_dates:
            if date in returns_pivot.index:
                date_returns = returns_pivot.loc[date].dropna()
                if len(date_returns) > 0:
                    n_assets = len(date_returns)
                    weight = 1.0 / n_assets
                    portfolio_return = (date_returns * weight).sum()
                    one_over_n_returns.append(portfolio_return)
                else:
                    one_over_n_returns.append(0.0)
            else:
                one_over_n_returns.append(0.0)
        
        one_over_n_series = pd.Series(one_over_n_returns, index=target_dates, name='1/N Portfolio')
        logger.info(f"1/N 포트폴리오 벤치마크 생성 완료: {len(one_over_n_series)}개 시점")
        logger.info(f"[create_one_over_n_benchmark_investable] one_over_n_series shape: {one_over_n_series.shape}")
        logger.info("[create_one_over_n_benchmark_investable] Function exit.")
        return one_over_n_series
        
    except Exception as e:
        logger.error(f"[create_one_over_n_benchmark_investable] 1/N 포트폴리오 생성 실패: {e}")
        logger.info("[create_one_over_n_benchmark_investable] Function exit with error.")
        return None

def run_backtest(start_year, end_year, etf_costs, asset_groups, group_constraints, model_params, benchmark_tickers, run_rolling_tune=True, tune_trials=50):
    import json
    logger.info("[run_backtest] Function entry.")
    logger.debug(f"[run_backtest] Received initial model_params:\n{json.dumps(model_params, indent=4)}")
    logger.info(f"[run_backtest] Input: start_year={start_year}, end_year={end_year}, run_rolling_tune={run_rolling_tune}")
    logger.info("백테스트 프로세스 시작")
    qs.extend_pandas()

    daily_df, monthly_df, vix_df, ff_df, all_tickers = data_manager.load_raw_data()
    logger.info(f"[run_backtest] Raw data loaded: daily_df shape={daily_df.shape}, monthly_df shape={monthly_df.shape}, vix_df shape={vix_df.shape}, ff_df shape={ff_df.shape}, all_tickers len={len(all_tickers)}")
    
    filtered_costs, filtered_asset_groups, filtered_group_constraints, filtered_benchmarks = _filter_config_by_tickers(
        all_tickers, etf_costs, asset_groups, group_constraints, benchmark_tickers
    )
    logger.info(f"[run_backtest] Config filtered: filtered_costs len={len(filtered_costs)}, filtered_asset_groups len={len(filtered_asset_groups)}, filtered_benchmarks len={len(filtered_benchmarks)}")
    
    ml_features_df = data_manager.create_feature_dataset(daily_df, monthly_df, vix_df, ff_df)
    logger.info(f"[run_backtest] ML features created: ml_features_df shape={ml_features_df.shape}, columns={ml_features_df.columns.tolist()}")

    active_model_params = model_params.get('tcn_svr_params') if model_params.get('use_tcn_svr') else model_params.get('arima_params')
    lookback_months = active_model_params.get('lookback_window') if model_params.get('use_tcn_svr') else active_model_params.get('lookback_months')
    tau = active_model_params.get('tau')

    expense_ratios_for_bl = {k: v['expense_ratio'] for k, v in filtered_costs.items()}
    bl_portfolio_model = BlackLittermanPortfolio(
        all_returns_df=monthly_df,
        ff_df=ff_df,
        expense_ratios=expense_ratios_for_bl,
        lookback_months=lookback_months,
        tau=tau,
        market_proxy_ticker=model_params['market_proxy_ticker'],
        asset_groups=filtered_asset_groups,
        group_constraints=filtered_group_constraints
    )
    logger.info("[run_backtest] BlackLittermanPortfolio initialized.")

    backtest_dates = pd.to_datetime(ml_features_df['date'].unique())
    backtest_dates = backtest_dates[(backtest_dates.year >= start_year) & (backtest_dates.year <= end_year)]
    logger.info(f"[run_backtest] Backtest dates range: {backtest_dates.min()} to {backtest_dates.max()}, total {len(backtest_dates)} dates.")
    
    bl_returns, ew_returns = [], []
    previous_weights = pd.Series(dtype=float)
    current_year = 0

    for analysis_date in backtest_dates[:-1]:
        if analysis_date.year != current_year:
            current_year = analysis_date.year
            if run_rolling_tune and model_params.get('use_tcn_svr', False):
                logger.info(f"\nStarting new backtest year {current_year}. Running hyperparameter tuning...")
                tcn_svr_tuner.run_tuning(n_trials=tune_trials, end_date=analysis_date)
                
                model_params = config.get_model_params()
                active_model_params = model_params['tcn_svr_params']
                logger.info("Tuning complete for the year. Using newly tuned parameters.")
                logger.info(f"Tuned parameters for year {current_year}:\n{json.dumps(model_params, indent=4)}")

        logger.info(f"\n[run_backtest] {analysis_date.strftime('%Y-%m')}월 백테스트 처리 시작")
        current_tickers, returns_pivot = bl_portfolio_model._get_current_universe(analysis_date)
        logger.info(f"[run_backtest] 투자 유니버스 ({len(current_tickers)}개): {current_tickers}")
        logger.info(f"[run_backtest] returns_pivot shape: {returns_pivot.shape}")
        
        if not current_tickers:
            logger.warning(f"[run_backtest] {analysis_date.strftime('%Y-%m-%d')} 건너뛰기, 유니버스 구성 자산 수 부족")
            weights = pd.Series(dtype=float)
        else:
            Sigma, delta, W_mkt = bl_portfolio_model._calculate_inputs(returns_pivot, analysis_date)
            logger.info(f"[run_backtest] BL inputs: Sigma shape={Sigma.shape if Sigma is not None else 'None'}, delta={delta}, W_mkt shape={W_mkt.shape if W_mkt is not None else 'None'}")
            if Sigma is None:
                weights = pd.Series(np.ones(len(current_tickers)) / len(current_tickers), index=current_tickers)
                logger.warning("[run_backtest] Sigma is None, using equal weights.")
            else:
                if model_params.get('use_tcn_svr', False):
                    logger.info("[run_backtest] Calling generate_tcn_svr_views.")
                    P, Q, Omega = ml_view_generator.generate_tcn_svr_views(
                        analysis_date=analysis_date, 
                        tickers=current_tickers, 
                        full_feature_df=ml_features_df,
                        model_params=active_model_params
                    )
                else:
                    logger.info("[run_backtest] Calling generate_ml_views.")
                    P, Q, Omega = ml_view_generator.generate_ml_views(
                        analysis_date=analysis_date, tickers=current_tickers, full_feature_df=ml_features_df, 
                        Sigma=Sigma, tau=tau, benchmark_ticker=model_params['market_proxy_ticker'],
                        view_outperformance=active_model_params['view_outperformance']
                    )
                logger.info(f"[run_backtest] Views generated: P shape={P.shape}, Q shape={Q.shape}, Omega shape={Omega.shape}")
                weights, _ = bl_portfolio_model.get_black_litterman_portfolio(
                    analysis_date=analysis_date, P=P, Q=Q, Omega=Omega, 
                    pre_calculated_inputs=(Sigma, delta, W_mkt), max_weight=model_params['max_weight'],
                    previous_weights=previous_weights
                )
                logger.info(f"[run_backtest] Portfolio weights calculated: weights shape={weights.shape}")
        
        next_month_date = analysis_date + pd.offsets.MonthEnd(1)
        next_month_returns = monthly_df[monthly_df['date'] == next_month_date]
        logger.info(f"[run_backtest] next_month_returns shape: {next_month_returns.shape}")

        if next_month_returns.empty:
            logger.warning(f"\n{next_month_date.strftime('%Y-%m-%d')} 수익률 데이터 없음, 해당 월 수익률을 0으로 처리")
            net_bl_return = 0.0
        else:
            if weights is not None and not weights.empty:
                merged_bl = pd.merge(weights.to_frame('weight'), next_month_returns, left_index=True, right_on='ticker')
                raw_bl_return = (merged_bl['weight'] * merged_bl['retx']).sum()
                
                holding_costs = (merged_bl['weight'] * merged_bl['ticker'].map(lambda t: filtered_costs.get(t, {}).get('expense_ratio', 0) / 12)).sum()
                
                aligned_prev, aligned_new = previous_weights.align(weights, join='outer', fill_value=0)
                trade_cost = (np.abs(aligned_new - aligned_prev) * aligned_new.index.map(lambda t: filtered_costs.get(t, {}).get('trading_cost_spread', 0.0001))).sum()

                net_bl_return = raw_bl_return - holding_costs - trade_cost
            else:
                net_bl_return = 0
        logger.info(f"[run_backtest] Net BL return for {next_month_date.strftime('%Y-%m')}: {net_bl_return:.4f}")

        bl_returns.append(pd.Series([net_bl_return], index=[next_month_date]))
        previous_weights = weights

    logger.info("\n백테스트 분석 및 리포트 생성")
    bl_returns_series = pd.concat(bl_returns).sort_index().squeeze().rename("BL_ML_Strategy")
    logger.info(f"[run_backtest] bl_returns_series shape: {bl_returns_series.shape}")
    
    benchmark_returns_dict = {}
    valid_benchmark_tickers = [t for t in filtered_benchmarks if t is not None]
    
    for ticker in valid_benchmark_tickers:
        ticker_returns = monthly_df[monthly_df['ticker'] == ticker].set_index('date')['retx']
        ticker_returns = ticker_returns[~ticker_returns.index.duplicated(keep='first')]
        benchmark_returns_dict[ticker] = ticker_returns.rename(ticker)
    logger.info(f"[run_backtest] benchmark_returns_dict keys: {benchmark_returns_dict.keys()}")
    
    if None in filtered_benchmarks:
        logger.info("1/N 포트폴리오 벤치마크 생성 중...")
        investable_tickers = list(filtered_costs.keys())
        one_over_n_returns = create_one_over_n_benchmark_investable(monthly_df, bl_returns_series.index, investable_tickers)
        if one_over_n_returns is not None:
            benchmark_returns_dict['1/N Portfolio'] = one_over_n_returns
    
    results_dict = {'BL_ML_Strategy': bl_returns_series}
    if benchmark_returns_dict:
        for ticker, returns in benchmark_returns_dict.items():
            results_dict[ticker] = returns.reindex(bl_returns_series.index)
    
    results_df = pd.DataFrame(results_dict)
    cumulative_results_df = (1 + results_df).cumprod()
    logger.info(f"[run_backtest] cumulative_results_df shape: {cumulative_results_df.shape}")
    cum_results_path = os.path.join(config.OUTPUT_DIR, 'cumulative_returns.csv')
    cumulative_results_df.to_csv(cum_results_path)
    logger.info(f"누적 수익률 데이터 저장 완료: {cum_results_path}")
    
    final_returns = cumulative_results_df.iloc[-1] - 1
    logger.info("최종 누적 수익률: " + ', '.join([f'{idx} {val:.4f}' for idx, val in final_returns.items()]))
    logger.info("[run_backtest] Function exit.")
    return ff_df