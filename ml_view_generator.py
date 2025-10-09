import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
import config
from logger_setup import logger
import traceback

def train_volatility_model(ticker, train_df):
    """단일 티커의 변동성 예측을 위한 ARIMAX 모델 훈련.

    pmdarima.auto_arima로 최적 ARIMA 차수를 자동 탐색하고,
    외생 변수(exogenous variables)를 포함하여 시계열 패턴과 외부 요인 영향을 모두 고려.
    """
    y_train = train_df['realized_vol'].values
    X_train = train_df.drop(columns=['TICKER', 'target_return', 'realized_vol']).values

    # 입력 데이터가 1차원 배열인 경우 2차원으로 변환
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    
    logger.error(f"DEBUG: Training model with y_train shape={y_train.shape}, X_train shape={X_train.shape}")

    # auto_arima로 최적 모델 탐색 및 학습
    model = pm.auto_arima(y_train, 
                          exogenous=X_train,
                          start_p=0, start_q=0,
                          max_p=3, max_q=3, m=1, d=None,
                          seasonal=False, 
                          stepwise=True, trace=False, 
                          error_action='ignore', suppress_warnings=True)
    
    return model

def generate_ml_views(analysis_date, tickers, full_feature_df, Sigma, tau, benchmark_ticker, view_outperformance):
    """ARIMAX 변동성 예측 기반의 Black-Litterman 뷰(view) 생성.

    '저변동성 이상 현상(Low-volatility anomaly)'에 근거하여, 변동성이 낮을 것으로
    예측되는 자산이 벤치마크 대비 초과 성과를 낼 것이라는 뷰를 구성.
    config.USE_DYNAMIC_OMEGA 설정 시, 모델 잔차 분산을 이용해 뷰의 불확실성(Omega)을 동적으로 조정.
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
        
        # 모델 학습에 필요한 최소 데이터 길이 확인
        MIN_TRAIN_SAMPLES = 15
        if len(ticker_train_df) < MIN_TRAIN_SAMPLES:
            logger.warning(f"Skipping {ticker} due to insufficient training data: {len(ticker_train_df)} samples < {MIN_TRAIN_SAMPLES}")
            continue

        if ticker_train_df.empty or ticker_features.empty:
            continue

        try:
            model = train_volatility_model(ticker, ticker_train_df)
            
            # 학습에 사용된 컬럼 순서를 보장
            train_cols = ticker_train_df.drop(columns=['TICKER', 'target_return', 'realized_vol']).columns
            X_pred_df = ticker_features[train_cols]

            # 예측용 데이터를 numpy 배열로 변환
            X_pred = np.array(X_pred_df)
            if X_pred.ndim == 1:
                X_pred = X_pred.reshape(-1, 1)

            # pmdarima 모델의 predict API 사용 및 반환 타입에 따른 처리
            prediction_result = model.predict(n_periods=1, X=X_pred)
            if isinstance(prediction_result, pd.Series):
                pred_vol = prediction_result.iloc[0]
            else:
                pred_vol = prediction_result[0]
            
            predictions[ticker] = pred_vol
            
            # pmdarima 모델의 resid() 메소드 사용
            residual_variances[ticker] = np.var(model.resid())
        except Exception as e:
            logger.error(f"--- ERROR processing ticker: {ticker} ---")
            logger.error(f"Exception Type: {type(e)}")
            logger.error(f"Exception Message: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            continue

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

    logger.error(f"DEBUG: Returning P shape={P.shape}, Q shape={Q.shape}, Omega shape={Omega.shape}")
    return P, Q, Omega