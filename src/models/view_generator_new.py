import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
try:
    import pmdarima as pm
except Exception:
    pm = None
from config import settings as config
from src.utils.logger import logger
import traceback
import torch

# In-process cache for warm-starting TCN per permno across months.
# Structure: { permno: { signature_tuple: state_dict } }

def train_volatility_model(permno, train_df, feature_cols):
    logger.info(f"[train_volatility_model] Function entry. PERMNO: {permno}")
    logger.info(f"[train_volatility_model] Input: train_df shape={train_df.shape}, feature_cols len={len(feature_cols)}")
    y_train = train_df['realized_vol'].values
    X_train = train_df[feature_cols].values

    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    
    logger.info(f"[train_volatility_model] y_train shape={y_train.shape}, X_train shape={X_train.shape}")

    if pm is None:
        raise RuntimeError("pmdarima is not installed; ARIMA-based ML views are disabled.")
    model = pm.auto_arima(y_train, 
                          exogenous=X_train,
                          start_p=0, start_q=0,
                          max_p=3, max_q=3, m=1, d=None,
                          seasonal=False, 
                          stepwise=True, trace=False, 
                          error_action='ignore', suppress_warnings=True)
    logger.info(f"[train_volatility_model] Model trained. Type: {type(model)}")
    logger.info("[train_volatility_model] Function exit.")
    return model

def generate_permno_tensors(permno, full_feature_df, indicator_features, all_features, train_window_rows, tcn_svr_model_params):
    permno_df = full_feature_df[full_feature_df['permno'] == permno].copy()
    
    target_indicator_cols = [f'target_{col}' for col in indicator_features]
    permno_df = permno_df.dropna(subset=all_features + ['target_return'] + target_indicator_cols)

    # If specified, restrict training rows for speed (use most recent rows)
    if train_window_rows is not None and len(permno_df) > train_window_rows:
        permno_df = permno_df.tail(train_window_rows)
    # logger.info(f"[generate_tcn_svr_views] PERMNO {permno}: permno_df shape={permno_df.shape}, columns={permno_df.columns.tolist()}")

    X_data = permno_df[all_features].values
    y_indicators = permno_df[target_indicator_cols].values
    y_returns = permno_df['target_return'].values
    # logger.info(f"[generate_tcn_svr_views] PERMNO {permno}: X_data shape={X_data.shape}, y_indicators shape={y_indicators.shape}, y_returns shape={y_returns.shape}")

    X_seq, y_seq_combined = _create_sequences(
        data=np.hstack([X_data, y_indicators, y_returns.reshape(-1,1)]), 
        lookback_window=tcn_svr_model_params['lookback_window']
    )
    # logger.info(f"[generate_tcn_svr_views] PERMNO {permno}: X_seq shape={X_seq.shape}, y_seq_combined shape={y_seq_combined.shape}")
    
    X_train_seq = X_seq[:, :, :-len(indicator_features)-1]
    y_train_indicators_seq = y_seq_combined[:, -len(indicator_features)-1:-1]
    y_train_returns_seq = y_seq_combined[:, -1]
    # logger.info(f"[generate_tcn_svr_views] PERMNO {permno}: X_train_seq shape={X_train_seq.shape}, y_train_indicators_seq shape={y_train_indicators_seq.shape}, y_train_returns_seq shape={y_train_returns_seq.shape}")
    X_train_tensor = torch.from_numpy(X_train_seq).float()
    y_train_indicators_tensor = torch.from_numpy(y_train_indicators_seq).float()
    
    X_test_seq = np.array([X_data[-tcn_svr_model_params['lookback_window']:]])
    X_test_tensor = torch.from_numpy(X_test_seq).float()
    return X_train_tensor, y_train_indicators_tensor, y_train_returns_seq, X_test_tensor

def _create_sequences(data, lookback_window):
    # logger.info("[_create_sequences] Function entry.")
    # logger.info(f"[_create_sequences] Input: data shape={data.shape}, lookback_window={lookback_window}")
    xs, ys = [], []
    for i in range(len(data) - lookback_window):
        x = data[i:(i + lookback_window)]
        y = data[i + lookback_window]
        xs.append(x)
        ys.append(y)
    # logger.info(f"[_create_sequences] Generated {len(xs)} sequences. Example x shape: {xs[0].shape if xs else 'N/A'}, Example y shape: {ys[0].shape if ys else 'N/A'}")
    # logger.info("[_create_sequences] Function exit.")
    return np.array(xs), np.array(ys)

def generate_tcn_svr_views(analysis_date, permnos, full_feature_df, model_params, model):
    logger.info("[generate_tcn_svr_views] Function entry.")
    logger.info(f"[generate_tcn_svr_views] Input: analysis_date={analysis_date}, permnos len={len(permnos)}, full_feature_df shape={full_feature_df.shape}, model_params={model_params}")
    
    num_assets = len(permnos)
    predicted_returns = np.zeros(num_assets)
    model_errors = np.zeros(num_assets) # Placeholder for future implementation
    
    indicator_features = ['ATRr_14', 'ADX_14', 'EMA_20', 'MACD_12_26_9', 'SMA_50', 'HURST', 'RSI_14']
    other_features = ['realized_vol', 'intra_month_mdd', 'avg_vix', 'vol_of_vix', 'Mkt-RF', 'SMB', 'HML', 'RF']
    use_lag = model_params.get('use_lag_features', True)
    max_lag = model_params.get('max_lag_features')
    lag_features_all = [col for col in full_feature_df.columns if '_lag_' in col]
    if not use_lag:
        lag_features = []
    else:
        lag_features = lag_features_all
        if isinstance(max_lag, int) and max_lag > 0 and len(lag_features) > max_lag:
            # Keep the last N lag features (assumes later-added columns are more recent)
            lag_features = lag_features[-max_lag:]
    all_features = indicator_features + other_features + lag_features
    logger.info(f"[generate_tcn_svr_views] Defined features: indicator_features len={len(indicator_features)}, other_features len={len(other_features)}, lag_features len={len(lag_features)}, all_features len={len(all_features)}")

    train_window_rows = model_params.get('train_window_rows')
    
    all_X_train_tensors_list = []
    all_y_indicators_tensors_list = []
    all_y_returns_seq_list = []
    all_X_test_tensors_list = []
    
    logger.info(f"[generate_tcn_svr_views] Gathering data for TCN_SVR training...")
    
    for i, permno in enumerate(permnos):
        # logger.info(f"[generate_tcn_svr_views] Processing permno: {permno}")
        X_train_tensor, y_indicators_tensor, y_returns_seq, X_test_tensor = generate_permno_tensors(
            i, permno, full_feature_df, indicator_features, all_features, train_window_rows, predicted_returns, model_params
        )
        
        all_X_train_tensors_list.append(X_train_tensor)
        all_y_indicators_tensors_list.append(y_indicators_tensor)
        all_y_returns_seq_list.append(y_returns_seq)
        all_X_test_tensors_list.append(X_test_tensor)
    
    all_X_train_tensors = torch.from_numpy(np.concatenate(all_X_train_tensors_list, axis=0))
    all_y_indicator_tensors = torch.from_numpy(np.concatenate(all_y_indicators_tensors_list, axis=0))
    all_y_returns_seq = np.hstack(all_y_returns_seq_list)
    logger.info(f"[generate_tcn_svr_views] Combined training data shapes: all_X_train_tensors shape={all_X_train_tensors.shape}, all_y_indicator_tensors shape={all_y_indicator_tensors.shape}, all_y_returns_seq shape={all_y_returns_seq.shape}")
    
    logger.info("Shuffling training data...")
    perm = torch.randperm(all_X_train_tensors.size(0))
    all_X_train_tensors = all_X_train_tensors[perm]
    all_y_indicator_tensors = all_y_indicator_tensors[perm]
    all_y_returns_seq = all_y_returns_seq[perm.numpy()]

    logger.info(f"[generate_tcn_svr_views] Preparing to fit TCN_SVR_Model with input_size={len(all_features)}, output_size={len(indicator_features)}")
    
    model.fit(all_X_train_tensors, all_y_indicator_tensors, all_y_returns_seq,
            epochs=model_params.get('epochs', 50),
            patience=model_params.get('early_stopping_patience', 10),
            min_delta=model_params.get('early_stopping_min_delta', 0.0001))
    
    return model