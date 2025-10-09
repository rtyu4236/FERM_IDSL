import pandas as pd
import numpy as np
import os
import data_manager
import ml_view_generator
from black_litterman import BlackLittermanPortfolio
import quantstats as qs
import config
from logger_setup import logger

def _filter_config_by_tickers(all_available_tickers, etf_costs, asset_groups, group_constraints, benchmark_tickers):
    """
    데이터에 존재하는 티커 목록을 기반으로 config 설정들을 필터링합니다.
    """
    available_set = set(all_available_tickers)
    logger.info(f"설정 필터링 기준 티커 수: {len(available_set)}")

    # ETF 비용 필터링
    filtered_costs = {t: c for t, c in etf_costs.items() if t in available_set}
    for ticker in available_set:
        if ticker not in filtered_costs:
            logger.warning(f"'{ticker}'에 대한 비용 정보가 config.ETF_COSTS에 없습니다. 기본값 0을 사용합니다.")
            filtered_costs[ticker] = {'expense_ratio': 0.0, 'trading_cost_spread': 0.0}
    
    # 자산 그룹 필터링
    filtered_asset_groups = {}
    for group, tickers in asset_groups.items():
        filtered_tickers = [t for t in tickers if t in available_set]
        if filtered_tickers:
            filtered_asset_groups[group] = filtered_tickers

    # 벤치마크 티커 필터링
    filtered_benchmark_tickers = [t for t in benchmark_tickers if t is None or t in available_set]
    
    logger.info("설정 필터링 완료")
    return filtered_costs, filtered_asset_groups, group_constraints, filtered_benchmark_tickers

def create_one_over_n_benchmark_investable(monthly_df, target_dates, investable_tickers):
    """
    투자 가능한 자산들로만 구성된 1/N 포트폴리오 벤치마크 생성
    """
    try:
        logger.info(f"1/N 포트폴리오 투자 유니버스: {len(investable_tickers)}개 자산 {investable_tickers}")
        
        investable_data = monthly_df[monthly_df['TICKER'].isin(investable_tickers)]
        
        if investable_data.empty:
            logger.warning("투자 가능한 자산 데이터가 없습니다.")
            return None
        
        returns_pivot = investable_data.pivot_table(index='date', columns='TICKER', values='retx')
        available_dates = returns_pivot.index.intersection(target_dates)
        returns_pivot = returns_pivot.loc[available_dates]
        
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
        return one_over_n_series
        
    except Exception as e:
        logger.error(f"1/N 포트폴리오 생성 실패: {e}")
        return None

def run_backtest(start_year, end_year, etf_costs, asset_groups, group_constraints, model_params, benchmark_tickers):
    """
    Black-Litterman 전략과 머신러닝 뷰 기반 전체 백테스트 실행
    """
    logger.info("백테스트 프로세스 시작")
    qs.extend_pandas()

    # 데이터 준비 및 동적 설정 필터링
    daily_df, monthly_df, vix_df, ff_df, all_tickers = data_manager.load_raw_data()
    
    filtered_costs, filtered_asset_groups, filtered_group_constraints, filtered_benchmarks = _filter_config_by_tickers(
        all_tickers, etf_costs, asset_groups, group_constraints, benchmark_tickers
    )
    
    ml_features_df = data_manager.create_feature_dataset(daily_df, monthly_df, vix_df, ff_df)

    # 모델 초기화
    expense_ratios_for_bl = {k: v['expense_ratio'] for k, v in filtered_costs.items()}
    bl_portfolio_model = BlackLittermanPortfolio(
        all_returns_df=monthly_df,
        ff_df=ff_df,
        expense_ratios=expense_ratios_for_bl,
        lookback_months=model_params['lookback_months'],
        tau=model_params['tau'],
        market_proxy_ticker=model_params['market_proxy_ticker'],
        asset_groups=filtered_asset_groups,
        group_constraints=filtered_group_constraints
    )

    # 백테스팅 루프
    backtest_dates = pd.to_datetime(ml_features_df['date'].unique())
    backtest_dates = backtest_dates[(backtest_dates.year >= start_year) & (backtest_dates.year <= end_year)]
    
    bl_returns, ew_returns = [], []
    previous_weights = pd.Series(dtype=float)

    for analysis_date in backtest_dates[:-1]:
        logger.info(f"\n{analysis_date.strftime('%Y-%m')}월 백테스트 처리 시작")
        current_tickers, returns_pivot = bl_portfolio_model._get_current_universe(analysis_date)
        logger.info(f"투자 유니버스 ({len(current_tickers)}개): {current_tickers}")
        
        if not current_tickers:
            logger.warning(f"{analysis_date.strftime('%Y-%m-%d')} 건너뛰기, 유니버스 구성 자산 수 부족")
            weights = pd.Series(dtype=float)
        else:
            Sigma, delta, W_mkt = bl_portfolio_model._calculate_inputs(returns_pivot, analysis_date)
            if Sigma is None:
                weights = pd.Series(np.ones(len(current_tickers)) / len(current_tickers), index=current_tickers)
            else:
                P, Q, Omega = ml_view_generator.generate_ml_views(
                    analysis_date=analysis_date, tickers=current_tickers, full_feature_df=ml_features_df, 
                    Sigma=Sigma, tau=model_params['tau'], benchmark_ticker=model_params['market_proxy_ticker'],
                    view_outperformance=model_params['view_outperformance']
                )
                weights, _ = bl_portfolio_model.get_black_litterman_portfolio(
                    analysis_date=analysis_date, P=P, Q=Q, Omega=Omega, 
                    pre_calculated_inputs=(Sigma, delta, W_mkt), max_weight=model_params['max_weight'],
                    previous_weights=previous_weights
                )
        
        # 수익률 계산
        next_month_date = analysis_date + pd.offsets.MonthEnd(1)
        next_month_returns = monthly_df[monthly_df['date'] == next_month_date]

        if next_month_returns.empty:
            logger.warning(f"\n{next_month_date.strftime('%Y-%m-%d')} 수익률 데이터 없음, 해당 월 수익률을 0으로 처리")
            net_bl_return = 0.0
        else:
            if weights is not None and not weights.empty:
                merged_bl = pd.merge(weights.to_frame('weight'), next_month_returns, left_index=True, right_on='TICKER')
                raw_bl_return = (merged_bl['weight'] * merged_bl['retx']).sum()
                
                holding_costs = (merged_bl['weight'] * merged_bl['TICKER'].map(lambda t: filtered_costs.get(t, {}).get('expense_ratio', 0) / 12)).sum()
                
                aligned_prev, aligned_new = previous_weights.align(weights, join='outer', fill_value=0)
                trade_cost = (np.abs(aligned_new - aligned_prev) * aligned_new.index.map(lambda t: filtered_costs.get(t, {}).get('trading_cost_spread', 0.0001))).sum()

                net_bl_return = raw_bl_return - holding_costs - trade_cost
            else:
                net_bl_return = 0

        bl_returns.append(pd.Series([net_bl_return], index=[next_month_date]))
        previous_weights = weights

    # 성과 분석 및 리포트 생성
    logger.info("\n백테스트 분석 및 리포트 생성")
    bl_returns_series = pd.concat(bl_returns).sort_index().squeeze().rename("BL_ML_Strategy")
    
    benchmark_returns_dict = {}
    valid_benchmark_tickers = [t for t in filtered_benchmarks if t is not None]
    
    for ticker in valid_benchmark_tickers:
        ticker_returns = monthly_df[monthly_df['TICKER'] == ticker].set_index('date')['retx']
        ticker_returns = ticker_returns[~ticker_returns.index.duplicated(keep='first')]
        benchmark_returns_dict[ticker] = ticker_returns.rename(ticker)
    
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
    cum_results_path = os.path.join(config.OUTPUT_DIR, 'cumulative_returns.csv')
    cumulative_results_df.to_csv(cum_results_path)
    logger.info(f"누적 수익률 데이터 저장 완료: {cum_results_path}")
    
    final_returns = cumulative_results_df.iloc[-1] - 1
    logger.info("최종 누적 수익률: " + ', '.join([f'{idx} {val:.4f}' for idx, val in final_returns.items()]))
    
    return ff_df