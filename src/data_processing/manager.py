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
    daily_df = pd.read_csv(os.path.join(config.DATA_DIR, 'crsp_daily_all.csv'), low_memory=False)
    daily_df.columns = [col.lower() for col in daily_df.columns]
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    
    # Load monthly data and unify column names to lowercase
    monthly_df = pd.read_csv(os.path.join(config.DATA_DIR, 'crsp_monthly_all.csv'))
    monthly_df.columns = [col.lower() for col in monthly_df.columns]
    monthly_df['date'] = pd.to_datetime(monthly_df['date'])
    monthly_df = monthly_df[monthly_df['date'] >= '1990-01-01'].copy()
    monthly_df = monthly_df.dropna(subset=['ticker'])
    monthly_df = monthly_df[['date', 'ticker', 'vwretd']].copy()
    monthly_df.rename(columns={'vwretd': 'retx'}, inplace=True)

    all_tickers = monthly_df['ticker'].unique().tolist()
    logger.info(f"Found a total of {len(all_tickers)} unique tickers in the data.")

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
    return daily_df, monthly_df, vix_df, ff_df, all_tickers

def create_feature_dataset(daily_df, monthly_df, vix_df, ff_df):
    """Create a unified feature dataset for machine learning models."""
    cache_path = os.path.join(config.CACHE_DIR, f'feature_dataset_stationary_{config.CHECK_STATIONARITY}.feather')
    if config.USE_CACHING and os.path.exists(cache_path):
        logger.info(f"Loading cached feature dataset: {cache_path}")
        return pd.read_feather(cache_path)

    logger.info("Starting to create a new feature dataset for ML models.")
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

    logger.info("Starting to generate technical analysis indicators.")
    if 'close' not in daily_df.columns and 'vwretd' in daily_df.columns:
        logger.warning("'close' column not found. Generating a virtual close price based on 'vwretd'.")
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

    # if config.CHECK_STATIONARITY:
    #     logger.info("Starting feature stationarity check and processing")
    #     features_to_check = [
    #         'realized_vol', 'intra_month_mdd', 'avg_vix', 'vol_of_vix', 
    #         'Mkt-RF', 'SMB', 'HML', 'RF',
    #         'ATRr_14', 'ADX_14', 'DMP_14', 'DMN_14', 'EMA_20',
    #         'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
    #         'SMA_50', 'HURST', 'RSI_14'
    #     ]
    #     for col in features_to_check:
    #         if col not in final_df.columns or final_df[col].isnull().all():
    #             continue
    #         p_value = adfuller(final_df[col].dropna())[1]
    #         if p_value > config.STATIONARITY_SIGNIFICANCE_LEVEL:
    #             logger.info(f"  - Feature '{col}' is non-stationary, performing 1st differencing (p-value {p_value:.4f})")
    #             final_df[col] = final_df.groupby('ticker')[col].diff()
    #         else:
    #             logger.info(f"  - Feature '{col}' is stationary (p-value {p_value:.4f})")

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
        logger.info(f"Feature dataset cached successfully: {cache_path}")

    logger.info("Feature dataset creation complete.")
    return final_df

def filter_liquid_universe(daily_df, all_tickers, start_year, min_avg_value=1_000_000):
    """Filters the investment universe based on trading value."""
    logger.info("Starting universe pre-filtering based on liquidity...")
    
    # Skip filtering if 'prc' or 'vol' columns are missing.
    if 'prc' not in daily_df.columns or 'vol' not in daily_df.columns:
        logger.warning("'prc' or 'vol' not found in daily_df, skipping liquidity filter.")
        return all_tickers

    # Calculate trading value
    daily_df['value'] = daily_df['prc'] * daily_df['vol']
    
    # Calculate the average daily trading value for the last 3 months from the backtest start date
    filter_end_date = pd.to_datetime(f"{start_year}-01-01")
    filter_start_date = filter_end_date - pd.DateOffset(months=3)
    
    liquidity_df = daily_df[
        (daily_df['date'] >= filter_start_date) & 
        (daily_df['date'] < filter_end_date)
    ]
    
    avg_daily_value = liquidity_df.groupby('ticker')['value'].mean()
    
    # Select only tickers that meet the trading value criteria
    liquid_tickers = avg_daily_value[avg_daily_value >= min_avg_value].index.tolist()
    
    logger.info(f"Pre-filtering complete. {len(all_tickers)} tickers -> {len(liquid_tickers)} liquid tickers.")
    
    return liquid_tickers

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

    # 1. Calculate technical indicators directly on daily data
    # Prepare OHLCV data (similar to existing code)
    df = daily_df.copy()
    if 'close' not in df.columns and 'vwretd' in df.columns:
        logger.warning("'close' column not found. Generating a virtual close price based on 'vwretd'.")
        df['close'] = df.groupby('ticker')['vwretd'].transform(lambda x: (1 + x).cumprod())
    if 'high' not in df.columns:
        df['high'] = df['close']
    if 'low' not in df.columns:
        df['low'] = df['close']
    if 'open' not in df.columns:
        df['open'] = df.groupby('ticker')['close'].shift(1)
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
        # FGI is external data, so separate daily FGI data must be merged at this step.
        return group

    logger.info("Generating daily technical analysis indicators...")
    df_with_ta = df.groupby('ticker', group_keys=False).apply(apply_ta)
    
    # 2. Merge macro-economic data (VIX, F-F) on a daily basis
    # Expand monthly F-F data to daily (Forward Fill)
    ff_daily = ff_df.set_index('date').resample('D').ffill().reset_index()
    vix_daily = vix_df.copy()
    vix_daily.rename(columns={'^VIX': 'avg_vix'}, inplace=True) # VIX is already daily data
    vix_daily['vol_of_vix'] = vix_daily['avg_vix'].rolling(window=21).std()

    macro_daily_df = pd.merge(vix_daily[['date', 'avg_vix', 'vol_of_vix']], ff_daily, on='date', how='left')
    
    final_df = pd.merge(df_with_ta, macro_daily_df, on='date', how='left')

    # Add realized volatility (rolling 21-day std of returns)
    final_df['realized_vol'] = final_df.groupby('ticker')['vwretd'].transform(lambda x: x.rolling(window=21).std())

    # Add intra-month max drawdown (rolling 21-day)
    def rolling_max_drawdown(series):
        roll_max = series.rolling(window=21, min_periods=1).max()
        daily_dd = series / roll_max - 1.0
        return daily_dd.rolling(window=21, min_periods=1).min()

    final_df['intra_month_mdd'] = final_df.groupby('ticker')['close'].transform(rolling_max_drawdown)
    
    # Group by ticker and ffill to fill macro NaNs on weekends/holidays
    macro_cols = ['avg_vix', 'Mkt-RF', 'SMB', 'HML', 'RF', 'vol_of_vix']
    final_df[macro_cols] = final_df.groupby('ticker')[macro_cols].ffill()
    
    # also ffill the new features
    final_df[['realized_vol', 'intra_month_mdd']] = final_df.groupby('ticker')[['realized_vol', 'intra_month_mdd']].ffill()

    # 3. Create Target variables: values after "20 days"
    logger.info("Creating target variables for predicting 20 trading days ahead...")
    indicator_features = ['ATRr_14', 'ADX_14', 'EMA_20', 'MACD_12_26_9', 'SMA_50', 'HURST', 'RSI_14']
    
    # TCN's target: technical indicators after 20 days
    for col in indicator_features:
        if col in final_df.columns:
            final_df[f'target_{col}'] = final_df.groupby('ticker')[col].shift(-20)
            
    # SVR's final target: cumulative return after 20 days
    future_price = final_df.groupby('ticker')['close'].shift(-20)
    final_df['target_return'] = (future_price / final_df['close']) - 1

    # 4. Finalize dataset
    # Create lag features or other additional features (if necessary)
    # ...
    
    # Remove last days' data as they have no target value (NaN)
    final_df = final_df.dropna(subset=['target_return'])
    final_df = final_df.reset_index(drop=True)

    if config.USE_CACHING:
        final_df.to_feather(cache_path)
        logger.info(f"Daily feature dataset cached successfully: {cache_path}")

    logger.info("Daily feature dataset for TCN-SVR created successfully.")
    return final_df