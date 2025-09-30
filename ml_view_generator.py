import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
import config
from logger_setup import logger

def train_volatility_model(ticker, train_df):
    """
    단일 티커의 변동성 예측을 위한 ARIMAX 모델 훈련.

    - `pmdarima.auto_arima`로 최적 ARIMA 차수(p,d,q) 자동 탐색.
    - 외생 변수(exogenous variables)를 포함하여 모델 훈련.
    - 시계열 패턴과 외부 요인 영향을 모두 고려.

    Args:
        ticker (str): 모델 훈련 대상 티커명.
        train_df (pd.DataFrame): 해당 티커의 훈련 데이터.

    Returns:
        statsmodels.tsa.arima.model.ARIMAResultsWrapper: 훈련된 ARIMAX 모델 객체.
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
    ARIMAX 변동성 예측 기반의 Black-Litterman 뷰 생성.

    - 핵심 가정: '저변동성 이상 현상(Low-volatility anomaly)'.
    - 변동성이 낮을 것으로 예측되는 자산이 벤치마크 대비 초과 성과를 낼 것이라는 뷰 생성.
    - `config.USE_DYNAMIC_OMEGA`가 True일 경우, 모델 잔차 분산으로 뷰의 불확실성(Omega) 동적 조정.

    Args:
        analysis_date (pd.Timestamp): 뷰 생성 기준 날짜.
        tickers (list): 현재 투자 유니버스 티커 리스트.
        full_feature_df (pd.DataFrame): 모든 티커의 피처 데이터.
        Sigma (np.array): 현재 유니버스의 연율화된 공분산 행렬.
        tau (float): Black-Litterman 모델의 불확실성 파라미터.
        benchmark_ticker (str): 성과 비교 기준 티커.
        view_outperformance (float): 뷰의 월간 기대 초과 수익률.

    Returns:
        tuple: Black-Litterman 모델의 P, Q, Omega 행렬.
    """
    logger.info(f"[{analysis_date.strftime('%Y-%m-%d')}] ML 뷰 생성 시작...")
    
    train_df = full_feature_df[full_feature_df['date'] < analysis_date]
    features_for_prediction = full_feature_df[full_feature_df['date'] == analysis_date]

    if features_for_prediction.empty:
        logger.warning("예측 날짜에 사용할 피처가 없어 뷰를 생성할 수 없음.")
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
        logger.warning(f"벤치마크 티커 {benchmark_ticker} 예측이 없어 뷰를 생성할 수 없음.")
        return np.array([]), np.array([]), np.array([])

    benchmark_vol = predictions[benchmark_ticker]
    
    view_pairs = []
    for ticker, pred_vol in predictions.items():
        if ticker == benchmark_ticker:
            continue
        if pred_vol < benchmark_vol:
            view_pairs.append((ticker, benchmark_ticker))

    if not view_pairs:
        logger.info("생성된 확신 있는 뷰가 없음.")
        return np.array([]), np.array([]), np.array([])

    num_views = len(view_pairs)
    num_assets = len(tickers)
    
    P = np.zeros((num_views, num_assets))
    Q = np.full((num_views, 1), view_outperformance)
    
    # He-Litterman 공식 기반 기본 Omega 계산
    omega_diag_vector = np.diag(P @ (tau * Sigma) @ P.T).copy()

    for i, (winner, loser) in enumerate(view_pairs):
        winner_idx = tickers.index(winner)
        loser_idx = tickers.index(loser)
        P[i, winner_idx] = 1
        P[i, loser_idx] = -1
        logger.info(f"  - 뷰 생성: {winner}가 {loser}를 능가할 것으로 예상")

        # 동적 오메가 계산
        if config.USE_DYNAMIC_OMEGA:
            winner_resid_var = residual_variances.get(winner, 0)
            loser_resid_var = residual_variances.get(loser, 0)
            avg_resid_var = (winner_resid_var + loser_resid_var) / 2
            
            # 기본 불확실성에 모델 예측 오차(잔차 분산)를 더해 오메가 조정
            omega_diag_vector[i] += avg_resid_var
            logger.info(f"    - 동적 오메가 적용: 기본 불확실성에 {avg_resid_var:.6f} 추가")

    Omega = np.diag(omega_diag_vector)

    return P, Q, Omega