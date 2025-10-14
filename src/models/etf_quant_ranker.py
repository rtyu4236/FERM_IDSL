import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from src.utils.logger import logger
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

RANDOM_SEED = 42

def create_etf_universe_from_daily(daily_df, etf_tickers=None):
    """
    CRSP daily_df에서 실제 OHLCV 값을 사용하여 ETF 유니버스 데이터를 생성하는 수정된 함수.
    
    Args:
        daily_df (pd.DataFrame): CRSP daily 데이터
        etf_tickers (list): ETF 티커 리스트. None이면 모든 티커 사용
        
    Returns:
        pd.DataFrame: ETF 유니버스 데이터 (MultiIndex: date, ticker)
    """

    # PRC 대신 OPENPRC, ASKHI, BIDLO 등 실제 값 사용
    required_cols = ['date', 'TICKER', 'OPENPRC', 'ASKHI', 'BIDLO', 'PRC', 'VOL']
    missing_cols = [col for col in required_cols if col not in daily_df.columns]
    if missing_cols:
        raise ValueError(f"필수 컬럼이 누락되었습니다: {missing_cols}")
    
    # ETF 티커 필터링
    if etf_tickers is not None:
        daily_df = daily_df[daily_df['TICKER'].isin(etf_tickers)]
    
    # 데이터 정제
    df_clean = daily_df.copy()

    # 모든 OHLCV 컬럼에 대해 확인
    ohlcv_cols = ['OPENPRC', 'ASKHI', 'BIDLO', 'PRC', 'VOL']
    df_clean = df_clean.dropna(subset=ohlcv_cols)
    df_clean = df_clean[(df_clean[ohlcv_cols] > 0).all(axis=1)]

    # 근사치 생성 로직을 실제 데이터 매핑으로 변경
    df_clean['open'] = df_clean['OPENPRC']
    df_clean['high'] = df_clean['ASKHI']
    df_clean['low'] = df_clean['BIDLO']
    df_clean['close'] = df_clean['PRC']
    df_clean['volume'] = df_clean['VOL']
    
    # 날짜 형식 변환
    df_clean['date'] = pd.to_datetime(df_clean['date'])

    # TICKER를 ticker로 변경
    df_clean['ticker'] = df_clean['TICKER']
    df_clean.drop(columns=['TICKER'], inplace=True)

    # MultiIndex 생성
    etf_universe = df_clean.set_index(['date', 'ticker'])[['open', 'high', 'low', 'close', 'volume']]

    return etf_universe


class ETFQuantRanker:
    """
    (최종 최적화 버전) 대규모 ETF 유니버스에 대해 퀀트 랭킹을 효율적으로 계산하는 클래스.
    SVM, 랜덤 포레스트, 로지스틱 회귀 모델을 모두 포함.
    """
    def __init__(self, etf_universe_df: pd.DataFrame, ml_training_window_months: int = 36):
        # 데이터 정리 및 중복 제거
        if not isinstance(etf_universe_df.index, pd.MultiIndex):
            etf_universe_df['date'] = pd.to_datetime(etf_universe_df['date'])
            self.df = etf_universe_df.set_index(['date', 'ticker'])
        else:
            self.df = etf_universe_df.copy()
        
        # 중복된 인덱스 제거 (마지막 값 유지)
        self.df = self.df[~self.df.index.duplicated(keep='last')]
        
        # 인덱스 정렬
        self.df = self.df.sort_index()
        
        # 데이터 품질 검증
        self.df = self.df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
        self.df = self.df[(self.df[['open', 'high', 'low', 'close', 'volume']] > 0).all(axis=1)]
        
        self.ml_training_window_months = ml_training_window_months
        self.factors_df = None
        
        logger.info(f"ETFQuantRanker 초기화 완료. 데이터 크기: {self.df.shape}")
        logger.info(f"티커 수: {len(self.df.index.get_level_values('ticker').unique())}")
        logger.info(f"날짜 범위: {self.df.index.get_level_values('date').min()} ~ {self.df.index.get_level_values('date').max()}")
        
        self._calculate_all_factors()

    def _calculate_rsi(self, series: pd.Series, length: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -1 * delta.clip(upper=0)

        # Wilder's Smoothing 방식 사용 (RSI 표준)
        ema_gain = gain.ewm(com=length - 1, adjust=False).mean()
        ema_loss = loss.ewm(com=length - 1, adjust=False).mean()

        rs = ema_gain / ema_loss
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(100)
        
        return rsi

    def _calculate_all_factors(self):
        logger.info("모멘텀 팩터 계산 시작 (벡터화 방식)...")
        grouped = self.df.groupby(level='ticker')['close']
        self.df['mom3m'] = grouped.pct_change(63)
        self.df['mom6m'] = grouped.pct_change(126)
        self.df['mom12m'] = grouped.pct_change(252)
        self.df['volatility'] = grouped.pct_change().rolling(21).std()
        self.df['rsi14'] = grouped.transform(self._calculate_rsi, length=14)
        self.df['target'] = (grouped.pct_change(21).shift(-21) > 0).astype(float)
        
        self.factors_df = self.df.dropna(subset=['mom3m', 'mom6m', 'mom12m', 'rsi14', 'volatility', 'target'])
        logger.info(f"모멘텀 팩터 계산 완료. 유효 데이터 크기: {self.factors_df.shape}")

    def get_top_tickers(self, analysis_date_str: str, top_n: int = 100):
        analysis_date = pd.to_datetime(analysis_date_str)
        logger.info(f"{analysis_date_str} 기준 상위 {top_n}개 ETF 선정 시작...")

        features = ['mom3m', 'mom6m', 'mom12m', 'rsi14', 'volatility']
        
        train_end_date = analysis_date - pd.DateOffset(days=1)
        train_start_date = train_end_date - pd.DateOffset(months=self.ml_training_window_months)
        
        train_df = self.factors_df.loc[pd.IndexSlice[train_start_date:train_end_date, :]]
        
        predict_df = self.factors_df.loc[pd.IndexSlice[analysis_date, :]].copy()

        if train_df.empty or len(train_df) < 500 or predict_df.empty:
            logger.warning("학습 또는 예측 데이터 부족. ML 점수 없이 모멘텀 랭킹만 진행합니다.")
            ml_scores = pd.Series(0, index=predict_df.index)
        else:
            X_train = train_df[features]
            y_train = train_df['target']
            X_predict = predict_df[features]

            scaler = StandardScaler().fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            X_predict_scaled = scaler.transform(X_predict)
            
            models = {
                'lr': LogisticRegression(random_state=RANDOM_SEED, max_iter=1000, n_jobs=-1),
                # 'svm': SVC(kernel='linear',probability=True, random_state=RANDOM_SEED),
                'rf': RandomForestClassifier(
                    random_state=RANDOM_SEED, 
                    n_estimators=50,     
                    max_depth=15,         
                    min_samples_leaf=10, 
                    n_jobs=-1
                )
            }
            
            ml_scores_df = pd.DataFrame(index=X_predict.index)

            for name, model in models.items():
                try:
                    logger.info(f"{name} 모델 학습 중...")
                    model.fit(X_train_scaled, y_train)
                    ml_scores_df[f'ml_score_{name}'] = model.predict_proba(X_predict_scaled)[:, 1]
                except Exception as e:
                    logger.warning(f"{name} 모델 처리 실패: {e}")
                    ml_scores_df[f'ml_score_{name}'] = 0 # 실패 시 0점 부여

            # 각 모델의 점수를 평균하여 최종 ML 점수 생성
            ml_scores = ml_scores_df.mean(axis=1)

        rebal_df = self.factors_df.loc[pd.IndexSlice[analysis_date, :]].copy()
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
