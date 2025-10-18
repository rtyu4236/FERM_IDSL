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
    Load and perform initial cleaning of all raw data files.
    """
    logger.info("Starting to load raw data files.")
    
    # Load daily data and unify column names to lowercase
    daily_df = pd.read_csv(os.path.join(config.DATA_DIR, 'crsp_daily_cleaned.csv'), low_memory=False)
    daily_df.columns = [col.lower() for col in daily_df.columns]
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    
    # Load monthly data and unify column names to lowercase
    monthly_df = pd.read_csv(os.path.join(config.DATA_DIR, 'crsp_monthly_cleaned.csv'), low_memory=False)
    monthly_df.columns = [col.lower() for col in monthly_df.columns]
    monthly_df['date'] = pd.to_datetime(monthly_df['date'])
    monthly_df.drop_duplicates(subset=['permno', 'date'], keep='first', inplace=True)
    monthly_df = monthly_df[monthly_df['date'] >= '1990-01-01'].copy()
    monthly_df = monthly_df.dropna(subset=['permno'])
    monthly_df = monthly_df[['date', 'permno', 'ret']].copy()
    monthly_df.rename(columns={'ret': 'total_return'}, inplace=True)
    monthly_df['total_return'] = pd.to_numeric(monthly_df['total_return'], errors='coerce')

    all_permnos = monthly_df['permno'].unique().tolist()
    logger.info(f"Found a total of {len(all_permnos)} unique permnos in the data.")

    vix_df = pd.read_csv(os.path.join(config.DATA_DIR, 'vix_index.csv'))
    vix_df.rename(columns={'Date': 'date'}, inplace=True)
    vix_df['date'] = pd.to_datetime(vix_df['date'])

    ff_df = pd.read_csv(os.path.join(config.DATA_DIR, 'F-F_Research_Data_Factors_monthly.csv'))
    ff_df.rename(columns={'Date': 'date'}, inplace=True)
    ff_df = ff_df[pd.to_numeric(ff_df['date'], errors='coerce').notna()]
    ff_df['date'] = pd.to_datetime(ff_df['date'], format='%Y%m')
    ff_df['date'] = ff_df['date'] + pd.offsets.MonthEnd(0)
    ff_df[['Mkt-RF', 'SMB', 'HML', 'RF']] = ff_df[['Mkt-RF', 'SMB', 'HML', 'RF']].astype(float) / 100
    
    logger.info("Raw data loading complete.")
    return daily_df, monthly_df, vix_df, ff_df, all_permnos

def create_feature_dataset(daily_df, monthly_df, vix_df, ff_df):
    """Create a unified feature dataset for machine learning models."""
    cache_path = os.path.join(config.CACHE_DIR, f'feature_dataset_stationary_{config.CHECK_STATIONARITY}.feather')
    if config.USE_CACHING and os.path.exists(cache_path):
        logger.info(f"Loading cached feature dataset: {cache_path}")
        return pd.read_feather(cache_path)

    logger.info("Starting to create a new feature dataset for ML models.")

    monthly_returns = monthly_df.pivot_table(index='date', columns='permno', values='total_return')
    target_returns = monthly_returns.shift(-1)

    def intra_month_mdd(x):
        cumulative_returns = (1 + x).cumprod()
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        return drawdown.min() if not drawdown.empty else 0

    features = daily_df.groupby(['permno', pd.Grouper(key='date', freq='ME')])['vwretd'].agg(
        realized_vol='std',
        intra_month_mdd=intra_month_mdd
    ).reset_index()
    features['date'] = features['date'] + pd.offsets.MonthEnd(0)
    
    vix_monthly = vix_df.groupby(pd.Grouper(key='date', freq='ME')).agg(
        avg_vix=('^VIX', 'mean'),
        vol_of_vix=('^VIX', 'std')
    ).reset_index()
    vix_monthly['date'] = vix_monthly['date'] + pd.offsets.MonthEnd(0)
    
    macro_df = pd.merge(vix_monthly, ff_df, on='date', how='left')

    target_returns_long = target_returns.stack().reset_index(name='target_return')
    final_df = pd.merge(features, target_returns_long, on=['date', 'permno'], how='left')
    final_df = pd.merge(final_df, macro_df, on='date', how='left')
    final_df = final_df.sort_values(by=['permno', 'date']).reset_index(drop=True)

    logger.info("Starting to generate technical analysis indicators.")
    if 'close' not in daily_df.columns and 'vwretd' in daily_df.columns:
        logger.warning("'close' column not found. Generating a virtual close price based on 'vwretd'.")
        daily_df['close'] = daily_df.groupby('permno')['vwretd'].transform(lambda x: (1 + x).cumprod())
    if 'high' not in daily_df.columns:
        daily_df['high'] = daily_df['close']
    if 'low' not in daily_df.columns:
        daily_df['low'] = daily_df['close']
    if 'open' not in daily_df.columns:
        daily_df['open'] = daily_df.groupby('permno')['close'].shift(1)
    if 'volume' not in daily_df.columns:
        daily_df['volume'] = 0
    daily_df.fillna(method='ffill', inplace=True)

    def apply_ta(df):
        logger.info(f"[DEBUG] apply_ta: Processing permno {df['permno'].iloc[0]} with shape {df.shape}")
        df.ta.atr(append=True)
        df.ta.adx(append=True)
        df.ta.ema(length=20, append=True)
        df.ta.macd(append=True)
        df.ta.sma(length=50, append=True)
        df['HURST'] = calculate_hurst_exponent(df['close'].values)
        df.ta.rsi(append=True)
        return df

    daily_with_ta = daily_df.groupby('permno', group_keys=False).apply(apply_ta)

    monthly_ta = daily_with_ta.groupby('permno').apply(
        lambda df: df.set_index('date').resample('M').last()
    ).drop(columns='permno', errors='ignore').reset_index()
    monthly_ta['date'] = monthly_ta['date'] + pd.offsets.MonthEnd(0)

    ta_feature_columns = ['ATRr_14', 'ADX_14', 'DMP_14', 'DMN_14', 'EMA_20', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'SMA_50', 'HURST', 'RSI_14']

    final_df = pd.merge(final_df, monthly_ta[['permno', 'date'] + ta_feature_columns], on=['permno', 'date'], how='left')

    final_df = final_df.dropna(subset=['target_return'])

    for lag in [1, 2, 3, 12]:
        final_df[f'realized_vol_lag_{lag}'] = final_df.groupby('permno')['realized_vol'].shift(lag)

    final_df = final_df.dropna().reset_index(drop=True)
    
    if config.USE_CACHING:
        os.makedirs(config.CACHE_DIR, exist_ok=True)
        final_df.to_feather(cache_path)
        logger.info(f"Feature dataset cached successfully: {cache_path}")

    logger.info("Feature dataset creation complete.")
    return final_df

def filter_liquid_universe(daily_df, all_permnos, start_year, min_avg_value=1_000_000):
    """Filters the investment universe based on trading value."""
    logger.info("Starting universe pre-filtering based on liquidity...")
    
    if 'prc' not in daily_df.columns or 'vol' not in daily_df.columns:
        logger.warning("'prc' or 'vol' not found in daily_df, skipping liquidity filter.")
        return all_permnos

    daily_df['value'] = daily_df['prc'] * daily_df['vol']
    
    filter_end_date = pd.to_datetime(f"{start_year}-01-01")
    filter_start_date = filter_end_date - pd.DateOffset(months=3)
    
    liquidity_df = daily_df[
        (daily_df['date'] >= filter_start_date) & 
        (daily_df['date'] < filter_end_date)
    ]
    
    avg_daily_value = liquidity_df.groupby('permno')['value'].mean()
    
    liquid_permnos = avg_daily_value[avg_daily_value >= min_avg_value].index.tolist()
    
    logger.info(f"Pre-filtering complete. {len(all_permnos)} permnos -> {len(liquid_permnos)} liquid permnos.")
    
    return liquid_permnos

if __name__ == '__main__':
    daily, monthly, vix, ff, _ = load_raw_data()
    feature_dataset = create_feature_dataset(daily, monthly, vix, ff)
    logger.info("\nFeature dataset sample")
    logger.info(feature_dataset.head())
    logger.info("\nFeature dataset info")
    feature_dataset.info()

def create_daily_feature_dataset_for_tcn(daily_df, vix_df, ff_df):
    """
    Generates a 'daily' time-series feature dataset for TCN-SVR model training, following the paper's methodology.
    """
    cache_path = os.path.join(config.CACHE_DIR, 'daily_feature_dataset_tcn.feather')
    if config.USE_CACHING and os.path.exists(cache_path):
        logger.info(f"Loading cached daily feature dataset: {cache_path}")
        return pd.read_feather(cache_path)

    logger.info("Starting to create a new daily feature dataset for TCN-SVR.")

    df = daily_df.copy()
    if 'close' not in df.columns:
        if 'prc' in df.columns:
            logger.info("'close' column not found, but 'prc' column found. Using 'prc' as 'close'.")
            df['close'] = df['prc']
        elif 'vwretd' in df.columns:
            logger.warning("'close' and 'prc' columns not found. Generating a virtual close price based on 'vwretd'.")
            df['close'] = df.groupby('permno')['vwretd'].transform(lambda x: (1 + x).cumprod())
        else:
            logger.error("Neither 'close', 'prc', nor 'vwretd' found. Cannot generate close price.")
            raise ValueError("Missing required price data for 'close' column.")
    if 'high' not in df.columns:
        df['high'] = df['close']
    if 'low' not in df.columns:
        df['low'] = df['close']
    if 'open' not in df.columns:
        df['open'] = df.groupby('permno')['close'].shift(1)
    if 'volume' not in df.columns:
        df['volume'] = 0
    df.fillna(method='ffill', inplace=True)

    def apply_ta(group):
        group.ta.atr(append=True)
        group.ta.adx(append=True)
        group.ta.ema(length=20, append=True)
        group.ta.macd(append=True)
        group.ta.sma(length=50, append=True)
        group['HURST'] = calculate_hurst_exponent(group['close'].values)
        group.ta.rsi(append=True)
        return group

    logger.info("Generating daily technical analysis indicators...")
    df_with_ta = df.groupby('permno', group_keys=False).apply(apply_ta)
    
    ff_daily = ff_df.set_index('date').resample('D').ffill().reset_index()
    vix_daily = vix_df.copy()
    vix_daily.rename(columns={'^VIX': 'avg_vix'}, inplace=True)
    vix_daily['vol_of_vix'] = vix_daily['avg_vix'].rolling(window=21).std()

    macro_daily_df = pd.merge(vix_daily[['date', 'avg_vix', 'vol_of_vix']], ff_daily, on='date', how='left')
    
    final_df = pd.merge(df_with_ta, macro_daily_df, on='date', how='left')

    final_df['realized_vol'] = final_df.groupby('permno')['vwretd'].transform(lambda x: x.rolling(window=21).std())

    def rolling_max_drawdown(series):
        roll_max = series.rolling(window=21, min_periods=1).max()
        daily_dd = series / roll_max - 1.0
        return daily_dd.rolling(window=21, min_periods=1).min()

    final_df['intra_month_mdd'] = final_df.groupby('permno')['close'].transform(rolling_max_drawdown)
    
    macro_cols = ['avg_vix', 'Mkt-RF', 'SMB', 'HML', 'RF', 'vol_of_vix']
    final_df[macro_cols] = final_df.groupby('permno')[macro_cols].ffill()
    
    final_df[['realized_vol', 'intra_month_mdd']] = final_df.groupby('permno')[['realized_vol', 'intra_month_mdd']].ffill()

    logger.info("Creating target variables for predicting 20 trading days ahead...")
    indicator_features = ['ATRr_14', 'ADX_14', 'EMA_20', 'MACD_12_26_9', 'SMA_50', 'HURST', 'RSI_14']
    
    for col in indicator_features:
        if col in final_df.columns:
            final_df[f'target_{col}'] = final_df.groupby('permno')[col].shift(-20)
            
    future_price = final_df.groupby('permno')['close'].shift(-20)
    final_df['target_return'] = (future_price / final_df['close']) - 1

    final_df = final_df.dropna(subset=['target_return'])
    final_df = final_df.reset_index(drop=True)

    if config.USE_CACHING:
        final_df.to_feather(cache_path)
        logger.info(f"Daily feature dataset cached successfully: {cache_path}")

    logger.info("Daily feature dataset for TCN-SVR created successfully.")
    return final_df
