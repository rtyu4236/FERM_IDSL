import pandas as pd
import numpy as np
import os
import config
from statsmodels.tsa.stattools import adfuller
from logger_setup import logger

def load_raw_data():
    """
    모든 원시 데이터 파일을 로드하고 초기 정제 작업 수행.

    `config.py`에 정의된 `DATA_DIR`에서 원본 CSV 파일 로드.
    기본 전처리(날짜 변환, 컬럼명 표준화 등) 수행.

    Returns:
        tuple: (daily_df, monthly_df, vix_df, ff_df) 데이터프레임 튜플.
    """
    logger.info("원시 데이터 파일 로드 시작...")
    
    daily_df = pd.read_csv(os.path.join(config.DATA_DIR, 'crsp_daily.csv'), low_memory=False)
    daily_df['date'] = pd.to_datetime(daily_df['date'], format='%Y%m%d')
    
    monthly_df = pd.read_csv(os.path.join(config.DATA_DIR, 'crsp_monthly.csv'))
    monthly_df['date'] = pd.to_datetime(monthly_df['date'], format='%Y%m%d')
    monthly_df = monthly_df[monthly_df['date'] >= '1990-01-01'].copy()
    monthly_df = monthly_df.dropna(subset=['TICKER'])
    monthly_df = monthly_df[['date', 'TICKER', 'vwretd']].copy()
    monthly_df.rename(columns={'vwretd': 'retx'}, inplace=True)

    vix_df = pd.read_csv(os.path.join(config.DATA_DIR, 'vix_index.csv'))
    vix_df.rename(columns={'Date': 'date'}, inplace=True)
    vix_df['date'] = pd.to_datetime(vix_df['date'])

    ff_df = pd.read_csv(os.path.join(config.DATA_DIR, 'F-F_Research_Data_Factors.csv'), skiprows=3)
    ff_df.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
    ff_df = ff_df[pd.to_numeric(ff_df['date'], errors='coerce').notna()]
    ff_df['date'] = pd.to_datetime(ff_df['date'], format='%Y%m')
    ff_df[['Mkt-RF', 'SMB', 'HML', 'RF']] = ff_df[['Mkt-RF', 'SMB', 'HML', 'RF']].astype(float) / 100
    
    logger.info("원시 데이터 로딩 완료.")
    return daily_df, monthly_df, vix_df, ff_df

def create_feature_dataset(daily_df, monthly_df):
    """
    머신러닝 모델을 위한 통합 피처(feature) 데이터셋 생성.

    `config.py` 설정에 따라 캐싱 및 피처 정상성 검증 기능 지원.

    Args:
        daily_df (pd.DataFrame): 일별 CRSP 데이터.
        monthly_df (pd.DataFrame): 월별 CRSP 데이터.

    Returns:
        pd.DataFrame: 모든 피처와 타겟 변수가 포함된 최종 데이터프레임.
    """
    cache_path = os.path.join(config.CACHE_DIR, f'feature_dataset_stationary_{config.CHECK_STATIONARITY}.feather')
    
    if config.USE_CACHING and os.path.exists(cache_path):
        logger.info(f"캐시된 피처 데이터셋 로드: {cache_path}")
        return pd.read_feather(cache_path)

    logger.info("ML 모델용 피처 데이터셋 신규 생성 시작...")

    # 1. 타겟 변수 생성: 다음 달의 수익률
    monthly_returns = monthly_df.pivot_table(index='date', columns='TICKER', values='retx')
    target_returns = monthly_returns.shift(-1)

    # 2. 월별 피처 계산
    def intra_month_mdd(x):
        "월중 최대 낙폭(MDD) 계산 헬퍼 함수"
        cumulative_returns = (1 + x).cumprod()
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        return drawdown.min() if not drawdown.empty else 0

    features = daily_df.groupby(['TICKER', pd.Grouper(key='date', freq='ME')])['vwretd'].agg(
        realized_vol='std',
        intra_month_mdd=intra_month_mdd
    ).reset_index()
    features['date'] = features['date'] + pd.offsets.MonthEnd(0)
    
    # 3. 거시 경제 데이터 준비
    _, _, vix_df, ff_df = load_raw_data()
    vix_monthly = vix_df.groupby(pd.Grouper(key='date', freq='ME')).agg(
        avg_vix=('^VIX', 'mean'),
        vol_of_vix=('^VIX', 'std')
    ).reset_index()
    vix_monthly['date'] = vix_monthly['date'] + pd.offsets.MonthEnd(0)
    
    ff_df['date'] = ff_df['date'] + pd.offsets.MonthEnd(0)
    macro_df = pd.merge(vix_monthly, ff_df, on='date', how='left')

    # 4. 모든 데이터 병합
    target_returns_long = target_returns.stack().reset_index(name='target_return')
    final_df = pd.merge(features, target_returns_long, on=['date', 'TICKER'], how='left')
    final_df = pd.merge(final_df, macro_df, on='date', how='left')
    final_df = final_df.sort_values(by=['TICKER', 'date']).reset_index(drop=True)

    # 5. 피처 정상성 검증 및 처리
    if config.CHECK_STATIONARITY:
        logger.info("피처 정상성(stationarity) 검증 및 처리 시작...")
        
        features_to_check = [
            'realized_vol', 'intra_month_mdd', 'avg_vix', 'vol_of_vix', 
            'Mkt-RF', 'SMB', 'HML', 'RF'
        ]
        
        for col in features_to_check:
            if col not in final_df.columns:
                continue
            
            p_value = adfuller(final_df[col].dropna())[1]
            
            if p_value > config.STATIONARITY_SIGNIFICANCE_LEVEL:
                logger.info(f"  - '{col}' 피처 비정상성으로 판단, 1차 차분 수행 (p-value: {p_value:.4f})")
                final_df[col] = final_df.groupby('TICKER')[col].diff()
            else:
                logger.info(f"  - '{col}' 피처 정상성 만족 (p-value: {p_value:.4f})")

    # 차분 등으로 발생한 결측치 제거
    final_df = final_df.dropna(subset=['target_return'])

    # 6. 시차 피처 생성 (정상성 처리 이후 수행)
    for lag in [1, 2, 3, 12]:
        final_df[f'realized_vol_lag_{lag}'] = final_df.groupby('TICKER')['realized_vol'].shift(lag)

    # 최종 결측치 제거
    final_df = final_df.dropna().reset_index(drop=True)
    
    # 캐시 저장
    if config.USE_CACHING:
        os.makedirs(config.CACHE_DIR, exist_ok=True)
        final_df.to_feather(cache_path)
        logger.info(f"피처 데이터셋 캐시 저장 완료: {cache_path}")

    logger.info("피처 데이터셋 생성 완료.")
    return final_df

if __name__ == '__main__':
    daily, monthly, _, _ = load_raw_data()
    feature_dataset = create_feature_dataset(daily, monthly)
    logger.info("\n--- 피처 데이터셋 샘플 ---")
    logger.info(feature_dataset.head())
    logger.info("\n--- 피처 데이터셋 정보 ---")
    feature_dataset.info()