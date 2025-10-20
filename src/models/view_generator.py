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
from src.models.tcn_svr import TCN_SVR_Model

# In-process cache for warm-starting TCN per permno across months.
# Structure: { permno: { signature_tuple: state_dict } }
_TCN_STATE_CACHE = {}

def clear_tcn_cache():
    """Clear the warm start cache. Useful when architecture changes or to reset state."""
    global _TCN_STATE_CACHE
    _TCN_STATE_CACHE.clear()
    logger.info("[clear_tcn_cache] Warm start cache cleared.")

def _tcn_signature(input_size, output_size, num_channels, kernel_size, lookback_window):
    """Return a tuple that uniquely describes the TCN architecture for warm start."""
    return (
        int(input_size),
        int(output_size),
        tuple(int(c) for c in (num_channels or [])),
        int(kernel_size),
        int(lookback_window),
    )

def _filter_compatible_state(model, cached_state):
    """Return a filtered state_dict containing only keys with matching shapes."""
    try:
        model_state = model.net.state_dict()
    except Exception:
        return None
    compatible = {}
    for k, v in cached_state.items():
        if k in model_state and hasattr(v, 'shape') and getattr(model_state[k], 'shape', None) == v.shape:
            compatible[k] = v
    return compatible if compatible else None

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

def generate_ml_views(analysis_date, permnos, full_feature_df, Sigma, tau, benchmark_permno, view_outperformance):
    logger.info("[generate_ml_views] Function entry.")
    logger.info(f"[generate_ml_views] Input: analysis_date={analysis_date}, permnos len={len(permnos)}, full_feature_df shape={full_feature_df.shape}, Sigma shape={Sigma.shape}, tau={tau}, benchmark_permno={benchmark_permno}, view_outperformance={view_outperformance}")
    
    train_df = full_feature_df[full_feature_df['date'] < analysis_date]
    features_for_prediction = full_feature_df[full_feature_df['date'] == analysis_date]
    logger.info(f"[generate_ml_views] train_df shape: {train_df.shape}, features_for_prediction shape: {features_for_prediction.shape}")

    if features_for_prediction.empty:
        logger.warning("No features for prediction date, cannot generate views.")
        logger.info("[generate_ml_views] Function exit (no features for prediction).")
        return np.array([]), np.array([]), np.array([])

    predictions = {}
    residual_variances = {}
    
    for permno in permnos:
        logger.info(f"[generate_ml_views] Processing permno: {permno}")
        permno_train_df = train_df[train_df['permno'] == permno].set_index('date')
        permno_features = features_for_prediction[features_for_prediction['permno'] == permno]
        logger.info(f"[generate_ml_views] PERMNO {permno}: permno_train_df shape={permno_train_df.shape}, permno_features shape={permno_features.shape}")
        
        MIN_TRAIN_SAMPLES = 15
        if len(permno_train_df) < MIN_TRAIN_SAMPLES:
            logger.warning(f"Skipping {permno} due to insufficient training data: {len(permno_train_df)} samples < {MIN_TRAIN_SAMPLES}")
            continue

        if permno_train_df.empty or permno_features.empty:
            logger.warning(f"[generate_ml_views] PERMNO {permno}: Empty train_df or features_for_prediction, skipping.")
            continue

        try:
            original_features = [
                'intra_month_mdd', 'avg_vix', 'vol_of_vix', 
                'Mkt-RF', 'SMB', 'HML', 'RF'
            ]
            lag_features = [col for col in permno_train_df.columns if '_lag_' in col]
            arima_features = original_features + lag_features
            arima_features = [f for f in arima_features if f in permno_train_df.columns]
            logger.info(f"[generate_ml_views] PERMNO {permno}: ARIMAX features len={len(arima_features)}: {arima_features}")

            model = train_volatility_model(permno, permno_train_df, arima_features)
            
            X_pred_df = permno_features[arima_features]
            logger.info(f"[generate_ml_views] PERMNO {permno}: X_pred_df shape={X_pred_df.shape}")

            X_pred = np.array(X_pred_df)
            if X_pred.ndim == 1:
                X_pred = X_pred.reshape(-1, 1)
            logger.info(f"[generate_ml_views] PERMNO {permno}: X_pred shape={X_pred.shape}")

            prediction_result = model.predict(n_periods=1, X=X_pred)
            if isinstance(prediction_result, pd.Series):
                pred_vol = prediction_result.iloc[0]
            else:
                pred_vol = prediction_result[0]
            logger.info(f"[generate_ml_views] PERMNO {permno}: Predicted volatility={pred_vol:.4f}")
            
            predictions[permno] = pred_vol
            
            residual_variances[permno] = np.var(model.resid())
        except Exception as e:
            logger.error(f"[generate_ml_views] PERMNO {permno}: ERROR processing permno: {e}")
            logger.error(f"[generate_ml_views] PERMNO {permno}: Traceback: {traceback.format_exc()}")
            continue

    if benchmark_permno not in predictions:
        logger.warning(f"No prediction for benchmark permno {benchmark_permno}, cannot generate views.")
        logger.info("[generate_ml_views] Function exit (no benchmark prediction).")
        return np.array([]), np.array([]), np.array([])

    sorted_predictions = sorted(predictions.items(), key=lambda item: item[1])
    non_benchmark_predictions = [p for p in sorted_predictions if p[0] != benchmark_permno]
    logger.info(f"[generate_ml_views] Sorted predictions len={len(sorted_predictions)}, non_benchmark_predictions len={len(non_benchmark_predictions)}")

    if not non_benchmark_predictions:
        logger.info("No predictions for non-benchmark assets, cannot generate views.")
        logger.info("[generate_ml_views] Function exit (no non-benchmark predictions).")
        return np.array([]), np.array([]), np.array([])

    best_performer_permno, best_performer_vol = non_benchmark_predictions[0]
    worst_performer_permno, worst_performer_vol = non_benchmark_predictions[-1]
    benchmark_vol = predictions[benchmark_permno]
    logger.info(f"[generate_ml_views] Best performer: {best_performer_permno} ({best_performer_vol:.4f}), Worst performer: {worst_performer_permno} ({worst_performer_vol:.4f}), Benchmark vol: {benchmark_vol:.4f}")

    view_definitions = []
    if best_performer_vol < benchmark_vol:
        view_definitions.append({
            'winner': best_performer_permno,
            'loser': benchmark_permno,
            'confidence': view_outperformance
        })

    if worst_performer_vol > benchmark_vol:
        view_definitions.append({
            'winner': benchmark_permno,
            'loser': worst_performer_permno,
            'confidence': view_outperformance
        })
    logger.info(f"[generate_ml_views] View definitions len={len(view_definitions)}")

    if not view_definitions:
        logger.info("No confident views were generated.")
        logger.info("[generate_ml_views] Function exit (no view definitions).")
        return np.array([]), np.array([]), np.array([])

    num_views = len(view_definitions)
    num_assets = len(permnos)
    
    P = np.zeros((num_views, num_assets))
    Q = np.zeros((num_views, 1))
    logger.info(f"[generate_ml_views] Initial P shape={P.shape}, Q shape={Q.shape}")
    
    temp_P = P.copy()
    for i, view in enumerate(view_definitions):
        winner_idx = permnos.index(view['winner'])
        loser_idx = permnos.index(view['loser'])
        temp_P[i, winner_idx] = 1
        temp_P[i, loser_idx] = -1
    omega_diag_vector = np.diag(temp_P @ (tau * Sigma) @ temp_P.T).copy()
    logger.info(f"[generate_ml_views] omega_diag_vector shape={omega_diag_vector.shape}")

    for i, view in enumerate(view_definitions):
        winner_idx = permnos.index(view['winner'])
        loser_idx = permnos.index(view['loser'])
        P[i, winner_idx] = 1
        P[i, loser_idx] = -1
        Q[i] = view['confidence']
        logger.info(f"View {i}: Expect {view['winner']} to outperform {view['loser']}, confidence={view['confidence']}")

        if config.USE_DYNAMIC_OMEGA:
            winner_resid_var = residual_variances.get(view['winner'], 0)
            loser_resid_var = residual_variances.get(view['loser'], 0)
            avg_resid_var = (winner_resid_var + loser_resid_var) / 2
            omega_diag_vector[i] += avg_resid_var
            logger.info(f"Applying dynamic omega, adding {avg_resid_var:.6f} to base uncertainty. New omega_diag_vector[{i}]={omega_diag_vector[i]:.6f}")

    Omega = np.diag(omega_diag_vector)

    logger.info(f"[generate_ml_views] Final P shape={P.shape}, Q shape={Q.shape}, Omega shape={Omega.shape}")
    logger.info("[generate_ml_views] Function exit.")
    return P, Q, Omega

def _create_sequences(data, lookback_window):
    logger.info("[_create_sequences] Function entry.")
    logger.info(f"[_create_sequences] Input: data shape={data.shape}, lookback_window={lookback_window}")
    xs, ys = [], []
    for i in range(len(data) - lookback_window):
        x = data[i:(i + lookback_window)]
        y = data[i + lookback_window]
        xs.append(x)
        ys.append(y)
    logger.info(f"[_create_sequences] Generated {len(xs)} sequences. Example x shape: {xs[0].shape if xs else 'N/A'}, Example y shape: {ys[0].shape if ys else 'N/A'}")
    logger.info("[_create_sequences] Function exit.")
    return np.array(xs), np.array(ys)

def generate_tcn_svr_views(analysis_date, permnos, full_feature_df, model_params):
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

    for i, permno in enumerate(permnos):
        logger.info(f"[generate_tcn_svr_views] Processing permno: {permno}")
        permno_df = full_feature_df[full_feature_df['permno'] == permno].copy()
        permno_df = permno_df.dropna(subset=all_features + ['target_return'])
        # If specified, restrict training rows for speed (use most recent rows)
        if train_window_rows is not None and len(permno_df) > train_window_rows:
            permno_df = permno_df.tail(train_window_rows)
        logger.info(f"[generate_tcn_svr_views] PERMNO {permno}: permno_df shape={permno_df.shape}, columns={permno_df.columns.tolist()}")
        
        if len(permno_df) < model_params['lookback_window'] + 1:
            logger.warning(f"Skipping view generation for {permno} due to insufficient data. (Data count: {len(permno_df)}, lookback: {model_params['lookback_window']})")
            predicted_returns[i] = 0
            continue

        X_data = permno_df[all_features].values
        y_indicators = permno_df[indicator_features].values
        y_returns = permno_df['target_return'].values
        logger.info(f"[generate_tcn_svr_views] PERMNO {permno}: X_data shape={X_data.shape}, y_indicators shape={y_indicators.shape}, y_returns shape={y_returns.shape}")

        X_seq, y_seq_combined = _create_sequences(np.hstack([X_data, y_indicators, y_returns.reshape(-1,1)]), model_params['lookback_window'])
        logger.info(f"[generate_tcn_svr_views] PERMNO {permno}: X_seq shape={X_seq.shape}, y_seq_combined shape={y_seq_combined.shape}")
        
        X_train_seq = X_seq[:, :, :-len(indicator_features)-1]
        y_train_indicators_seq = y_seq_combined[:, -len(indicator_features)-1:-1]
        y_train_returns_seq = y_seq_combined[:, -1]
        logger.info(f"[generate_tcn_svr_views] PERMNO {permno}: X_train_seq shape={X_train_seq.shape}, y_train_indicators_seq shape={y_train_indicators_seq.shape}, y_train_returns_seq shape={y_train_returns_seq.shape}")

        X_test_seq = np.array([X_data[-model_params['lookback_window']:]])
        logger.info(f"[generate_tcn_svr_views] PERMNO {permno}: X_test_seq shape={X_test_seq.shape}")

        X_train_tensor = torch.from_numpy(X_train_seq).float()
        y_train_indicators_tensor = torch.from_numpy(y_train_indicators_seq).float()
        X_test_tensor = torch.from_numpy(X_test_seq).float()
        logger.info(f"[generate_tcn_svr_views] PERMNO {permno}: X_train_tensor shape={X_train_tensor.shape}, y_train_indicators_tensor shape={y_train_indicators_tensor.shape}, X_test_tensor shape={X_test_tensor.shape}")

        model = TCN_SVR_Model(
            input_size=len(all_features),
            output_size=len(indicator_features),
            num_channels=model_params['num_channels'],
            kernel_size=model_params['kernel_size'],
            dropout=model_params['dropout'],
            lookback_window=model_params['lookback_window'],
            svr_C=model_params.get('svr_C', 1.0),
            svr_gamma=model_params.get('svr_gamma', 'scale'),
            lr=model_params.get('lr', 0.001) # Pass the tuned learning rate
        )
        # Warm start: load previous TCN weights for this permno if architecture matches; else skip.
        if model_params.get('warm_start', True):
            sig = _tcn_signature(
                input_size=len(all_features),
                output_size=len(indicator_features),
                num_channels=model_params['num_channels'],
                kernel_size=model_params['kernel_size'],
                lookback_window=model_params['lookback_window'],
            )
            cached_for_permno = _TCN_STATE_CACHE.get(permno)
            loaded = False
            if isinstance(cached_for_permno, dict):
                # New-style cache: keyed by signature
                cached_state = cached_for_permno.get(sig)
                if cached_state is not None:
                    compat = _filter_compatible_state(model, cached_state)
                    if compat:
                        try:
                            model.net.load_state_dict(compat, strict=False)
                            loaded = True
                            logger.info(f"[generate_tcn_svr_views] Warm-started TCN for {permno} with matching signature.")
                        except Exception as e:
                            logger.warning(f"[generate_tcn_svr_views] Warm-start load failed for {permno} despite signature match: {e}")
            else:
                # Backward-compat: older cache stored raw state_dict directly.
                cached_state = cached_for_permno
                if cached_state is not None:
                    compat = _filter_compatible_state(model, cached_state)
                    if compat:
                        try:
                            model.net.load_state_dict(compat, strict=False)
                            loaded = True
                            logger.info(f"[generate_tcn_svr_views] Warm-started TCN for {permno} (compat keys only).")
                        except Exception as e:
                            logger.warning(f"[generate_tcn_svr_views] Warm-start (compat) failed for {permno}: {e}")
            if not loaded:
                logger.info(f"[generate_tcn_svr_views] Skipping warm-start for {permno} (no compatible cached weights for current architecture).")
        logger.info(f"[generate_tcn_svr_views] PERMNO {permno}: TCN_SVR_Model initialized. Input size={len(all_features)}, Output size={len(indicator_features)}")
        model.fit(X_train_tensor, y_train_indicators_tensor, y_train_returns_seq,
                  epochs=model_params.get('epochs', 50),
                  patience=model_params.get('early_stopping_patience', 10),
                  min_delta=model_params.get('early_stopping_min_delta', 0.0001))
        logger.info(f"[generate_tcn_svr_views] PERMNO {permno}: TCN_SVR_Model fitted.")
        # Save state dict to cache for next month warm start (keyed by architecture signature)
        if model_params.get('warm_start', True):
            try:
                sig = _tcn_signature(
                    input_size=len(all_features),
                    output_size=len(indicator_features),
                    num_channels=model_params['num_channels'],
                    kernel_size=model_params['kernel_size'],
                    lookback_window=model_params['lookback_window'],
                )
                cache_bucket = _TCN_STATE_CACHE.setdefault(permno, {})
                if isinstance(cache_bucket, dict):
                    cache_bucket[sig] = model.net.state_dict()
                else:
                    # If stale format exists, replace with new dict format
                    _TCN_STATE_CACHE[permno] = {sig: model.net.state_dict()}
            except Exception:
                pass
        
        prediction = model.predict(X_test_tensor)
        predicted_returns[i] = prediction[0]
        logger.info(f"Predicted return for {permno}: {prediction[0]:.4f}")

    P = np.identity(num_assets)
    Q = predicted_returns.reshape(-1, 1)

    base_uncertainty = model_params.get('base_uncertainty', 0.1)
    omega_diag = np.full(num_assets, base_uncertainty) + model_errors
    Omega = np.diag(omega_diag)

    logger.info(f"[generate_tcn_svr_views] Final P shape={P.shape}, Q shape={Q.shape}, Omega shape={Omega.shape}")
    logger.info("Absolute view generation based on TCN-SVR complete.")
    logger.info("[generate_tcn_svr_views] Function exit.")
    return P, Q, Omega