import pandas as pd
import numpy as np
import os
import data_manager
import ml_view_generator
from black_litterman import BlackLittermanPortfolio
import quantstats as qs
import config
from logger_setup import logger

def run_backtest(start_year, end_year, etf_costs, model_params, benchmark_ticker):
    """
    Black-Litterman 전략과 머신러닝 뷰 기반 전체 백테스트 실행

    `main.py`에서 `config.py`를 통해 전달된 설정값으로 백테스팅 전 과정 조율
    최종적으로 `quantstats`를 사용해 상세 성과 분석 리포트 생성

    Args:
        start_year (int): 백테스트 시작 연도
        end_year (int): 백테스트 종료 연도
        etf_costs (dict): ETF별 운용 보수 및 거래 비용 정보
        model_params (dict): Black-Litterman 모델 및 뷰 생성 파라미터
        benchmark_ticker (str): 비교 벤치마크 티커
    """
    logger.info("백테스트 프로세스 시작")
    qs.extend_pandas()

    # 데이터 준비
    daily_df, monthly_df, vix_df, ff_df = data_manager.load_raw_data()
    ml_features_df = data_manager.create_feature_dataset(daily_df, monthly_df)
    
    # 모델 초기화
    expense_ratios_for_bl = {k: v['expense_ratio'] for k, v in etf_costs.items()}
    bl_portfolio_model = BlackLittermanPortfolio(
        all_returns_df=monthly_df,
        ff_df=ff_df,
        expense_ratios=expense_ratios_for_bl,
        lookback_months=model_params['lookback_months'],
        tau=model_params['tau'],
        market_proxy_ticker=model_params['market_proxy_ticker']
    )

    # 백테스팅 루프
    backtest_dates = pd.to_datetime(ml_features_df['date'].unique())
    backtest_dates = backtest_dates[(backtest_dates.year >= start_year) & (backtest_dates.year <= end_year)]
    
    bl_returns = []
    ew_returns = [] # 동일 가중 포트폴리오 수익률
    previous_weights = pd.Series(dtype=float)

    for analysis_date in backtest_dates[:-1]:
        logger.info(f"\n{analysis_date.strftime('%Y-%m')}월 백테스트 처리 시작")
        current_tickers, returns_pivot = bl_portfolio_model._get_current_universe(analysis_date)
        
        if not current_tickers:
            logger.warning(f"{analysis_date.strftime('%Y-%m-%d')} 건너뛰기, 유니버스 구성 자산 수 부족")
            weights = pd.Series(dtype=float)
            ew_weights = pd.Series(dtype=float)
        else:
            Sigma, delta, W_mkt = bl_portfolio_model._calculate_inputs(returns_pivot, analysis_date)
            if Sigma is None:
                weights = pd.Series(np.ones(len(current_tickers)) / len(current_tickers), index=current_tickers)
            else:
                P, Q, Omega = ml_view_generator.generate_ml_views(
                    analysis_date=analysis_date, 
                    tickers=current_tickers, 
                    full_feature_df=ml_features_df, 
                    Sigma=Sigma, 
                    tau=model_params['tau'],
                    benchmark_ticker=model_params['market_proxy_ticker'],
                    view_outperformance=model_params['view_outperformance']
                )
                weights, _ = bl_portfolio_model.get_black_litterman_portfolio(
                    analysis_date=analysis_date, 
                    P=P, 
                    Q=Q, 
                    Omega=Omega, 
                    pre_calculated_inputs=(Sigma, delta, W_mkt),
                    max_weight=model_params['max_weight'],
                    previous_weights=previous_weights
                )
            ew_weights = pd.Series(1/len(current_tickers), index=current_tickers)

        # 수익률 계산
        next_month_date = analysis_date + pd.offsets.MonthEnd(1)
        next_month_returns = monthly_df[monthly_df['date'] == next_month_date]

        if next_month_returns.empty:
            logger.warning(f"\n{next_month_date.strftime('%Y-%m-%d')} 수익률 데이터 없음, 해당 월 백테스트 건너뜀")
            continue

        if weights is not None and not weights.empty:
            merged_bl = pd.merge(weights.to_frame('weight'), next_month_returns, left_index=True, right_on='TICKER')
            raw_bl_return = (merged_bl['weight'] * merged_bl['retx']).sum()
            
            holding_costs = (merged_bl['weight'] * merged_bl['TICKER'].map(lambda t: etf_costs.get(t, {}).get('expense_ratio', 0) / 12)).sum()
            
            aligned_prev, aligned_new = previous_weights.align(weights, join='outer', fill_value=0)
            trade_cost = (np.abs(aligned_new - aligned_prev) * aligned_new.index.map(lambda t: etf_costs.get(t, {}).get('trading_cost_spread', 0.0001))).sum()

            net_bl_return = raw_bl_return - holding_costs - trade_cost
        else:
            net_bl_return = 0

        if ew_weights is not None and not ew_weights.empty:
            merged_ew = pd.merge(ew_weights.to_frame('weight'), next_month_returns, left_index=True, right_on='TICKER')
            net_ew_return = (merged_ew['weight'] * merged_ew['retx']).sum()
        else:
            net_ew_return = 0

        if pd.isna(net_bl_return):
            rf_return_series = ff_df[ff_df['date'].dt.to_period('M') == next_month_date.to_period('M')]['RF']
            net_bl_return = rf_return_series.iloc[0] if not rf_return_series.empty else 0

        bl_returns.append(pd.Series([net_bl_return], index=[next_month_date]))
        ew_returns.append(pd.Series([net_ew_return], index=[next_month_date]))
        previous_weights = weights

    # 성과 분석 및 리포트 생성 
    logger.info("\n백테스트 분석 및 리포트 생성")
    
    bl_returns_series = pd.concat(bl_returns).sort_index().squeeze().rename("BL_ML_Strategy")
    ew_returns_series = pd.concat(ew_returns).sort_index().squeeze().rename("Equal_Weight")
    
    # 벤치마크 수익률 준비 및 중복 인덱스 제거
    benchmark_returns = monthly_df[monthly_df['TICKER'] == benchmark_ticker]
    benchmark_returns = benchmark_returns.set_index('date')['retx']
    benchmark_returns = benchmark_returns[~benchmark_returns.index.duplicated(keep='first')]
    benchmark_returns = benchmark_returns.rename(benchmark_ticker)
    
    results_df = pd.DataFrame({
        'BL_ML_Strategy': bl_returns_series,
        'Equal_Weight': ew_returns_series,
    })
    # results_path = os.path.join(config.OUTPUT_DIR, 'backtest_returns.csv')
    # results_df.to_csv(results_path)
    # logger.info(f"백테스트 수익률 데이터 저장 완료: {results_path}")

    # 누적 수익률 계산 및 저장
    logger.info("\n누적 수익률 계산 시작")
    cumulative_results_df = (1 + results_df).cumprod()
    cum_results_path = os.path.join(config.OUTPUT_DIR, 'cumulative_returns.csv')
    cumulative_results_df.to_csv(cum_results_path)
    logger.info(f"누적 수익률 데이터 저장 완료: {cum_results_path}")
    logger.info("최종 누적 수익률")
    final_returns = cumulative_results_df.iloc[-1] - 1
    log_str = ', '.join([f'{idx} {val:.6f}' for idx, val in final_returns.items()])
    logger.info(log_str)

    # QuantStats 리포트 생성
    # report_path = os.path.join(config.OUTPUT_DIR, 'BL_ML_Strategy_report.html')
    # qs.reports.html(bl_returns_series, 
    #                 benchmark=benchmark_returns, 
    #                 output=report_path, 
    #                 title='Black-Litterman with ML Views Strategy Analysis')
    # logger.info(f"전략 분석 리포트 저장 완료: {report_path}")

    # # 동일 가중 벤치마크 리포트 생성
    # ew_report_path = os.path.join(config.OUTPUT_DIR, 'Equal_Weight_report.html')
    # qs.reports.html(ew_returns_series, 
    #                 benchmark=benchmark_returns, 
    #                 output=ew_report_path, 
    #                 title='Equal Weight Benchmark Analysis')
    # logger.info(f"동일 가중 벤치마크 리포트 저장 완료: {ew_report_path}")