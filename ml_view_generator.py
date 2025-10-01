import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
import config
from logger_setup import logger

def train_volatility_model(ticker, train_df):
    """
    단일 티커의 변동성 예측을 위한 ARIMAX 모델 훈련

    - `pmdarima.auto_arima`로 최적 ARIMA 차수(p,d,q) 자동 탐색
    - 외생 변수(exogenous variables)를 포함하여 모델 훈련
    - 시계열 패턴과 외부 요인 영향을 모두 고려

    Args:
        ticker (str): 모델 훈련 대상 티커명
        train_df (pd.DataFrame): 해당 티커의 훈련 데이터

    Returns:
        statsmodels.tsa.arima.model.ARIMAResultsWrapper: 훈련된 ARIMAX 모델 객체
    """
    y_train = train_df['realized_vol']
    X_train = train_df.drop(columns=['TICKER', 'target_return', 'realized_vol'])
    
    # auto_arima로 최적 (p,d,q) 탐색
    model = pm.auto_arima(y_train, 
                          exogenous=X_train,
                          start_p=0, start_q=0,
                          max_p=3, max_q=3, m=1, d=None,
                          seasonal=False, 
                          stepwise=True, trace=False, 
                          error_action='ignore', suppress_warnings=True)
    
    # 최적 차수로 statsmodels ARIMA 모델 정의 및 훈련
    sm_model = ARIMA(y_train, exog=X_train, order=model.order)
    fitted_model = sm_model.fit()
    
    return fitted_model

def generate_ml_views(analysis_date, tickers, full_feature_df, Sigma, tau, benchmark_ticker, view_outperformance):
    """
    ARIMAX 변동성 예측 기반의 Black-Litterman 뷰 생성

    - 핵심 가정: '저변동성 이상 현상(Low-volatility anomaly)'
    - 변동성이 낮을 것으로 예측되는 자산이 벤치마크 대비 초과 성과를 낼 것이라는 뷰 생성
    - `config.USE_DYNAMIC_OMEGA`가 True일 경우, 모델 잔차 분산으로 뷰의 불확실성(Omega) 동적 조정

    Args:
        analysis_date (pd.Timestamp): 뷰 생성 기준 날짜
        tickers (list): 현재 투자 유니버스 티커 리스트
        full_feature_df (pd.DataFrame): 모든 티커의 피처 데이터
        Sigma (np.array): 현재 유니버스의 연율화된 공분산 행렬
        tau (float): Black-Litterman 모델의 불확실성 파라미터
        benchmark_ticker (str): 성과 비교 기준 티커
        view_outperformance (float): 뷰의 월간 기대 초과 수익률

    Returns:
        tuple: Black-Litterman 모델의 P, Q, Omega 행렬
    """
    logger.info("ML 뷰 생성 시작")
    
    train_df = full_feature_df[full_feature_df['date'] < analysis_date]
    features_for_prediction = full_feature_df[full_feature_df['date'] == analysis_date]

    if features_for_prediction.empty:
        logger.warning("예측 날짜 피처 없음, 뷰 생성 불가")
        return np.array([]), np.array([]), np.array([])

    predictions = {}
    residual_variances = {}
    
    for ticker in tickers:
        ticker_train_df = train_df[train_df['TICKER'] == ticker].set_index('date')
        ticker_features = features_for_prediction[features_for_prediction['TICKER'] == ticker]
        
        if ticker_train_df.empty or ticker_features.empty:
            continue

        model = train_volatility_model(ticker, ticker_train_df)
        
        X_pred = ticker_features.drop(columns=['TICKER', 'target_return', 'realized_vol', 'date'])
        train_cols = train_df.drop(columns=['TICKER', 'target_return', 'realized_vol', 'date']).columns
        X_pred = X_pred[train_cols]

        pred_vol = model.forecast(steps=1, exog=X_pred).iloc[0]
        predictions[ticker] = pred_vol
        
        residual_variances[ticker] = model.resid.var()

    if benchmark_ticker not in predictions:
        logger.warning(f"벤치마크 티커 {benchmark_ticker} 예측 없음, 뷰 생성 불가")
        return np.array([]), np.array([]), np.array([])

    # 변동성 예측치를 기준으로 자산을 정렬
    sorted_predictions = sorted(predictions.items(), key=lambda item: item[1])
    
    # 뷰 후보군에서 벤치마크 자산은 제외
    non_benchmark_predictions = [p for p in sorted_predictions if p[0] != benchmark_ticker]

    if not non_benchmark_predictions:
        logger.info("벤치마크 외 자산 예측 없음, 뷰 생성 불가")
        return np.array([]), np.array([]), np.array([])

    # 가장 변동성이 낮은 자산과 높은 자산을 선택
    best_performer_ticker, best_performer_vol = non_benchmark_predictions[0]
    worst_performer_ticker, worst_performer_vol = non_benchmark_predictions[-1]
    benchmark_vol = predictions[benchmark_ticker]

    view_definitions = []
    # 뷰 1: Best Performer vs. Benchmark
    if best_performer_vol < benchmark_vol:
        view_definitions.append({
            'winner': best_performer_ticker,
            'loser': benchmark_ticker,
            'confidence': view_outperformance
        })

    # 뷰 2: Benchmark vs. Worst Performer
    if worst_performer_vol > benchmark_vol:
        view_definitions.append({
            'winner': benchmark_ticker,
            'loser': worst_performer_ticker,
            'confidence': view_outperformance
        })

    if not view_definitions:
        logger.info("생성된 확신 있는 뷰 없음")
        return np.array([]), np.array([]), np.array([])

    num_views = len(view_definitions)
    num_assets = len(tickers)
    
    P = np.zeros((num_views, num_assets))
    Q = np.zeros((num_views, 1))
    
    # He-Litterman 공식 기반 기본 Omega 계산
    # P 행렬이 먼저 채워져야 정확한 계산 가능, 임시 계산 후 아래에서 재계산
    temp_P = P.copy()
    for i, view in enumerate(view_definitions):
        winner_idx = tickers.index(view['winner'])
        loser_idx = tickers.index(view['loser'])
        temp_P[i, winner_idx] = 1
        temp_P[i, loser_idx] = -1
    omega_diag_vector = np.diag(temp_P @ (tau * Sigma) @ temp_P.T).copy()

    for i, view in enumerate(view_definitions):
        winner_idx = tickers.index(view['winner'])
        loser_idx = tickers.index(view['loser'])
        P[i, winner_idx] = 1
        P[i, loser_idx] = -1
        Q[i] = view['confidence']
        logger.info(f"{view['winner']}가 {view['loser']} 능가 예상")

        # 동적 오메가 계산
        if config.USE_DYNAMIC_OMEGA:
            winner_resid_var = residual_variances.get(view['winner'], 0)
            loser_resid_var = residual_variances.get(view['loser'], 0)
            avg_resid_var = (winner_resid_var + loser_resid_var) / 2
            
            # 기본 불확실성에 모델 예측 오차(잔차 분산)를 더해 오메가 조정
            omega_diag_vector[i] += avg_resid_var
            logger.info(f"동적 오메가 적용, 기본 불확실성에 {avg_resid_var:.6f} 추가")

    Omega = np.diag(omega_diag_vector)

    return P, Q, Omega