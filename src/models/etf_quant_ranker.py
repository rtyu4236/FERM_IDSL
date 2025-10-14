import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from src.utils.logger import logger
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

RANDOM_SEED = 42

def create_etf_universe_from_daily(daily_df, etf_tickers=None):
    """
    CRSP daily_df에서 실제 OHLCV 값을 사용하여 ETF 유니버스 데이터를 생성하는 수정된 함수.
    Args:
        daily_df (pd.DataFrame): 일별 CRSP 데이터 (컬럼은 소문자로 가정)
        etf_tickers (list): ETF 티커 리스트. None이면 모든 티커 사용
    Returns:
        pd.DataFrame: ETF 유니버스 데이터 (MultiIndex: date, ticker)
    """
    required_cols = ['date', 'ticker', 'openprc', 'askhi', 'bidlo', 'prc', 'vol']
    missing_cols = [col for col in required_cols if col not in daily_df.columns]
    if missing_cols:
        raise ValueError(f"필수 컬럼이 누락되었습니다: {missing_cols}")
    
    df_filtered = daily_df.copy()
    if etf_tickers is not None:
        df_filtered = df_filtered[df_filtered['ticker'].isin(etf_tickers)]
    
    ohlcv_cols = ['openprc', 'askhi', 'bidlo', 'prc', 'vol']
    df_clean = df_filtered.dropna(subset=ohlcv_cols)
    df_clean = df_clean[(df_clean[ohlcv_cols] > 0).all(axis=1)]

    df_clean['open'] = df_clean['openprc']
    df_clean['high'] = df_clean['askhi']
    df_clean['low'] = df_clean['bidlo']
    df_clean['close'] = df_clean['prc']
    df_clean['volume'] = df_clean['vol']
    
    df_clean['date'] = pd.to_datetime(df_clean['date'])
    etf_universe = df_clean.set_index(['date', 'ticker'])[['open', 'high', 'low', 'close', 'volume']]
    return etf_universe

class ETFQuantRanker:
    def __init__(self, ml_training_window_months: int = 36):
        self.ml_training_window_months = ml_training_window_months
        logger.info(f"ETFQuantRanker initialized with training window: {self.ml_training_window_months} months.")

    def _calculate_rsi(self, series: pd.Series, length: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -1 * delta.clip(upper=0)
        ema_gain = gain.ewm(com=length - 1, adjust=False).mean()
        ema_loss = loss.ewm(com=length - 1, adjust=False).mean()
        rs = ema_gain / ema_loss
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(100)
        return rsi

    def _calculate_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Calculating factors...")
        grouped = df.groupby(level='ticker')['close']
        df['mom3m'] = grouped.pct_change(63)
        df['mom6m'] = grouped.pct_change(126)
        df['mom12m'] = grouped.pct_change(252)
        df['volatility'] = grouped.pct_change().rolling(21).std()
        df['rsi14'] = grouped.transform(self._calculate_rsi, length=14)
        df['target'] = (grouped.pct_change(21).shift(-21) > 0).astype(float)
        factors_df = df.dropna(subset=['mom3m', 'mom6m', 'mom12m', 'rsi14', 'volatility', 'target'])
        logger.info(f"Factor calculation complete. Valid data size: {factors_df.shape}")
        return factors_df

    def get_top_tickers(self, analysis_date_str: str, daily_df: pd.DataFrame, all_tickers: list, top_n: int = 100):
        analysis_date = pd.to_datetime(analysis_date_str)
        logger.info(f"{analysis_date_str} 기준 상위 {top_n}개 ETF 선정 시작...")

        # 1. 현재 시점까지의 데이터만 사용
        data_for_ranking = daily_df[daily_df['date'] <= analysis_date].copy()
        etf_universe_df = create_etf_universe_from_daily(data_for_ranking, etf_tickers=all_tickers)

        # 2. 현재 시점까지의 데이터로 팩터 계산
        factors_df = self._calculate_factors(etf_universe_df)

        features = ['mom3m', 'mom6m', 'mom12m', 'rsi14', 'volatility']
        
        # 3. ML 모델 학습 및 예측
        train_end_date = analysis_date - pd.DateOffset(days=1)
        train_start_date = train_end_date - pd.DateOffset(months=self.ml_training_window_months)
        
        train_df = factors_df.loc[pd.IndexSlice[train_start_date:train_end_date, :]]
        predict_df = factors_df.loc[pd.IndexSlice[analysis_date, :]].copy()

        if train_df.empty or len(train_df) < 500 or predict_df.empty:
            logger.warning("학습 또는 예측 데이터 부족. ML 점수 없이 모멘텀 랭킹만 진행합니다.")
            ml_scores = pd.Series(0, index=predict_df.index)
        else:
            X_train = train_df[features]
            y_train = train_df['target']
            X_predict = predict_df[features]

            # 데이터 정제 (NaN, inf 처리)
            X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
            X_predict.replace([np.inf, -np.inf], np.nan, inplace=True)
            imputer = SimpleImputer(strategy='median')
            X_train = imputer.fit_transform(X_train)
            X_predict = imputer.transform(X_predict)

            scaler = StandardScaler().fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            X_predict_scaled = scaler.transform(X_predict)
            
            models = {
                'lr': LogisticRegression(random_state=RANDOM_SEED, max_iter=1000, n_jobs=-1),
                'rf': RandomForestClassifier(random_state=RANDOM_SEED, n_estimators=50, max_depth=15, min_samples_leaf=10, n_jobs=-1)
            }
            
            ml_scores_df = pd.DataFrame(index=predict_df.index)
            for name, model in models.items():
                try:
                    logger.info(f"{name} 모델 학습 중...")
                    model.fit(X_train_scaled, y_train)
                    ml_scores_df[f'ml_score_{name}'] = model.predict_proba(X_predict_scaled)[:, 1]
                except Exception as e:
                    logger.warning(f"{name} 모델 처리 실패: {e}")
                    ml_scores_df[f'ml_score_{name}'] = 0

            ml_scores = ml_scores_df.mean(axis=1)

        # 4. 최종 점수 계산 및 랭킹
        rebal_df = factors_df.loc[pd.IndexSlice[analysis_date, :]].copy()
        rebal_df['ml_score'] = ml_scores
        rebal_df.dropna(subset=['ml_score'], inplace=True)

        if rebal_df.empty:
            logger.warning(f"{analysis_date_str}에 랭킹할 최종 데이터가 없습니다.")
            return []

        for factor in features:
            rebal_df[f'{factor}_rank'] = rebal_df[factor].rank(pct=True)
        rebal_df['ml_score_rank'] = rebal_df['ml_score'].rank(pct=True)

        momentum_rank_cols = [f'{f}_rank' for f in features]
        rebal_df['momentum_score'] = rebal_df[momentum_rank_cols].mean(axis=1)
        rebal_df['final_score'] = 0.5 * rebal_df['momentum_score'] + 0.5 * rebal_df['ml_score_rank']

        top_tickers = rebal_df.sort_values('final_score', ascending=False).head(top_n).index.get_level_values('ticker').tolist()
        
        logger.info(f"선정 완료: {len(top_tickers)}개 ETF")
        return top_tickers