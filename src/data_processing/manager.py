import pandas as pd
import numpy as np
import os
import pandas_ta as ta
from config import settings as config
from statsmodels.tsa.stattools import adfuller
from src.utils.logger import logger

def calculate_hurst_exponent(time_series, max_lag=100):
    """Calculates the Hurst Exponent of a time series."""
    if len(time_series) < max_lag:
        return 0.5 # Return neutral value if series is too short

    lags = range(2, max_lag)
    tau = [np.std(time_series[lag:] - time_series[:-lag]) for lag in lags]
    
    # Filter out zero or negative tau values for log compatibility
    tau = [v for v in tau if v > 0]
    if len(tau) < 2:
        return 0.5 # Not enough data points to fit

    lags = range(2, len(tau) + 2)
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0]

def load_raw_data():
    """
    모든 원시 데이터 파일을 로드하고 초기 정제 작업 수행
    `config.py`에 정의된 `DATA_DIR`에서 원본 CSV 파일 로드
    기본 전처리(날짜 변환, 컬럼명 표준화 등) 수행
    """
    logger.info("원시 데이터 파일 로드 시작")
    
    daily_df = pd.read_csv(os.path.join(config.DATA_DIR, 'crsp_daily.csv'), low_memory=False)
    daily_df.columns = [col.lower() for col in daily_df.columns]
    daily_df['date'] = pd.to_datetime(daily_df['date'], format='%Y%m%d')
    
    monthly_df = pd.read_csv(os.path.join(config.DATA_DIR, 'crsp_monthly.csv'))
    monthly_df.columns = [col.lower() for col in monthly_df.columns]
    monthly_df['date'] = pd.to_datetime(monthly_df['date'], format='%Y%m%d')
    monthly_df = monthly_df[monthly_df['date'] >= '1990-01-01'].copy()
    monthly_df = monthly_df.dropna(subset=['ticker'])
    monthly_df = monthly_df[['date', 'ticker', 'vwretd']].copy()
    monthly_df.rename(columns={'vwretd': 'retx'}, inplace=True)

    all_tickers = monthly_df['ticker'].unique().tolist()
    logger.info(f"데이터에서 총 {len(all_tickers)}개의 고유 티커 발견")

    vix_df = pd.read_csv(os.path.join(config.DATA_DIR, 'vix_index.csv'))
    vix_df.rename(columns={'Date': 'date'}, inplace=True)
    vix_df['date'] = pd.to_datetime(vix_df['date'])

    ff_df = pd.read_csv(os.path.join(config.DATA_DIR, 'F-F_Research_Data_Factors_monthly.csv'))
    ff_df.rename(columns={'Date': 'date'}, inplace=True)
    ff_df = ff_df[pd.to_numeric(ff_df['date'], errors='coerce').notna()]
    ff_df['date'] = pd.to_datetime(ff_df['date'], format='%Y%m')
    ff_df['date'] = ff_df['date'] + pd.offsets.MonthEnd(0)
    ff_df[['Mkt-RF', 'SMB', 'HML', 'RF']] = ff_df[['Mkt-RF', 'SMB', 'HML', 'RF']].astype(float) / 100
    
    logger.info("원시 데이터 로딩 완료")
    return daily_df, monthly_df, vix_df, ff_df, all_tickers

def create_feature_dataset(daily_df, monthly_df, vix_df, ff_df):
    """머신러닝 모델을 위한 통합 피처(feature) 데이터셋 생성"""
    cache_path = os.path.join(config.CACHE_DIR, f'feature_dataset_stationary_{config.CHECK_STATIONARITY}.feather')
    if config.USE_CACHING and os.path.exists(cache_path):
        logger.info(f"캐시된 피처 데이터셋 로드: {cache_path}")
        return pd.read_feather(cache_path)

    logger.info("ML 모델용 피처 데이터셋 신규 생성 시작")
    logger.info(f"[DEBUG] Initial monthly_df columns: {monthly_df.columns.tolist()}")
    logger.info(f"[DEBUG] Initial daily_df columns: {daily_df.columns.tolist()}")

    monthly_returns = monthly_df.pivot_table(index='date', columns='ticker', values='retx')
    target_returns = monthly_returns.shift(-1)
    logger.info(f"[DEBUG] target_returns shape: {target_returns.shape}, dtypes: {target_returns.dtypes.tolist()}")

    def intra_month_mdd(x):
        cumulative_returns = (1 + x).cumprod()
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        return drawdown.min() if not drawdown.empty else 0

    features = daily_df.groupby(['ticker', pd.Grouper(key='date', freq='ME')])['vwretd'].agg(
        realized_vol='std',
        intra_month_mdd=intra_month_mdd
    ).reset_index()
    features['date'] = features['date'] + pd.offsets.MonthEnd(0)
    logger.info(f"[DEBUG] Features (realized_vol, mdd) shape: {features.shape}, columns: {features.columns.tolist()}")
    logger.info(f"[DEBUG] Features head:\n{features.head()}")
    
    vix_monthly = vix_df.groupby(pd.Grouper(key='date', freq='ME')).agg(
        avg_vix=('^VIX', 'mean'),
        vol_of_vix=('^VIX', 'std')
    ).reset_index()
    vix_monthly['date'] = vix_monthly['date'] + pd.offsets.MonthEnd(0)
    
    macro_df = pd.merge(vix_monthly, ff_df, on='date', how='left')
    logger.info(f"[DEBUG] Macro_df shape: {macro_df.shape}, columns: {macro_df.columns.tolist()}")
    logger.info(f"[DEBUG] Macro_df head:\n{macro_df.head()}")

    target_returns_long = target_returns.stack().reset_index(name='target_return')
    final_df = pd.merge(features, target_returns_long, on=['date', 'ticker'], how='left')
    final_df = pd.merge(final_df, macro_df, on='date', how='left')
    final_df = final_df.sort_values(by=['ticker', 'date']).reset_index(drop=True)
    logger.info(f"[DEBUG] final_df after initial merges shape: {final_df.shape}, columns: {final_df.columns.tolist()}")
    logger.info(f"[DEBUG] final_df head:\n{final_df.head()}")

    logger.info("기술적 분석 지표 생성 시작")
    if 'close' not in daily_df.columns and 'vwretd' in daily_df.columns:
        logger.warning("'close' 컬럼이 없어 'vwretd'를 기반으로 가상 종가를 생성합니다.")
        daily_df['close'] = daily_df.groupby('ticker')['vwretd'].transform(lambda x: (1 + x).cumprod())
    if 'high' not in daily_df.columns:
        daily_df['high'] = daily_df['close']
    if 'low' not in daily_df.columns:
        daily_df['low'] = daily_df['close']
    if 'open' not in daily_df.columns:
        daily_df['open'] = daily_df.groupby('ticker')['close'].shift(1)
    if 'volume' not in daily_df.columns:
        daily_df['volume'] = 0
    daily_df.fillna(method='ffill', inplace=True)
    logger.info(f"[DEBUG] daily_df after OHLCV generation shape: {daily_df.shape}, columns: {daily_df.columns.tolist()}")
    logger.info(f"[DEBUG] daily_df head:\n{daily_df.head()}")

    def apply_ta(df):
        logger.info(f"[DEBUG] apply_ta: Processing ticker {df['ticker'].iloc[0]} with shape {df.shape}")
        logger.info(f"[DEBUG] apply_ta: df columns before TA: {df.columns.tolist()}")
        df.ta.atr(append=True)
        df.ta.adx(append=True)
        df.ta.ema(length=20, append=True)
        df.ta.macd(append=True)
        df.ta.sma(length=50, append=True)
        df['HURST'] = calculate_hurst_exponent(df['close'].values)
        df.ta.rsi(append=True)
        logger.info(f"[DEBUG] apply_ta: df columns after TA: {df.columns.tolist()}")
        return df

    daily_with_ta = daily_df.groupby('ticker', group_keys=False).apply(apply_ta)
    logger.info(f"[DEBUG] daily_with_ta shape: {daily_with_ta.shape}, columns: {daily_with_ta.columns.tolist()}")
    logger.info(f"[DEBUG] daily_with_ta head:\n{daily_with_ta.head()}")

    monthly_ta = daily_with_ta.groupby('ticker').apply(
        lambda df: df.set_index('date').resample('M').last()
    ).drop(columns='ticker', errors='ignore').reset_index()
    monthly_ta['date'] = monthly_ta['date'] + pd.offsets.MonthEnd(0)
    logger.info(f"[DEBUG] monthly_ta after resampling shape: {monthly_ta.shape}, columns: {monthly_ta.columns.tolist()}")
    logger.info(f"[DEBUG] monthly_ta head:\n{monthly_ta.head()}")

    ta_feature_columns = ['ATRr_14', 'ADX_14', 'DMP_14', 'DMN_14', 'EMA_20', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'SMA_50', 'HURST', 'RSI_14']

    logger.info(f"[DEBUG] Columns of final_df before merge: {final_df.columns.tolist()}")
    logger.info(f"[DEBUG] Columns of monthly_ta: {monthly_ta.columns.tolist()}")
    logger.info(f"[DEBUG] Data types of merge keys in final_df: ticker={final_df['ticker'].dtype}, date={final_df['date'].dtype}")
    logger.info(f"[DEBUG] Data types of merge keys in monthly_ta: ticker={monthly_ta['ticker'].dtype}, date={monthly_ta['date'].dtype}")
    logger.info(f"[DEBUG] Head of monthly_ta:\n{monthly_ta.head()}")

    final_df = pd.merge(final_df, monthly_ta[['ticker', 'date'] + ta_feature_columns], on=['ticker', 'date'], how='left')

    logger.info(f"[DEBUG] Columns of final_df after merge: {final_df.columns.tolist()}")
    logger.info(f"[DEBUG] final_df head after TA merge:\n{final_df.head()}")

    if config.CHECK_STATIONARITY:
        logger.info("피처 정상성 검증 및 처리 시작")
        features_to_check = [
            'realized_vol', 'intra_month_mdd', 'avg_vix', 'vol_of_vix', 
            'Mkt-RF', 'SMB', 'HML', 'RF',
            'ATRr_14', 'ADX_14', 'DMP_14', 'DMN_14', 'EMA_20',
            'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
            'SMA_50', 'HURST', 'RSI_14'
        ]
        for col in features_to_check:
            if col not in final_df.columns or final_df[col].isnull().all():
                logger.warning(f"[DEBUG] Skipping stationarity check for {col}: not in final_df or all null.")
                continue
            p_value = adfuller(final_df[col].dropna())[1]
            if p_value > config.STATIONARITY_SIGNIFICANCE_LEVEL:
                logger.info(f"  - '{col}' 피처 비정상성, 1차 차분 수행 (p-value {p_value:.4f})")
                final_df[col] = final_df.groupby('ticker')[col].diff()
            else:
                logger.info(f"  - '{col}' 피처 정상성 만족 (p-value {p_value:.4f})")

    final_df = final_df.dropna(subset=['target_return'])
    logger.info(f"[DEBUG] final_df shape after dropping NaNs for target_return: {final_df.shape}")

    for lag in [1, 2, 3, 12]:
        final_df[f'realized_vol_lag_{lag}'] = final_df.groupby('ticker')['realized_vol'].shift(lag)
    logger.info(f"[DEBUG] final_df columns after lag feature creation: {final_df.columns.tolist()}")

    final_df = final_df.dropna().reset_index(drop=True)
    logger.info(f"[DEBUG] final_df final shape: {final_df.shape}")
    logger.info(f"[DEBUG] final_df final head:\n{final_df.head()}")
    
    if config.USE_CACHING:
        os.makedirs(config.CACHE_DIR, exist_ok=True)
        final_df.to_feather(cache_path)
        logger.info(f"피처 데이터셋 캐시 저장 완료: {cache_path}")

    logger.info("피처 데이터셋 생성 완료")
    return final_df

if __name__ == '__main__':
    daily, monthly, vix, ff, _ = load_raw_data()
    feature_dataset = create_feature_dataset(daily, monthly, vix, ff)
    logger.info("\n피처 데이터셋 샘플")
    logger.info(feature_dataset.head())
    logger.info("\n피처 데이터셋 정보")
    feature_dataset.info()