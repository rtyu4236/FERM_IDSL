import os
import json
from copy import deepcopy

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
CACHE_DIR = os.path.join(ROOT_DIR, 'cache')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Main Settings ---
START_YEAR = 2010
END_YEAR = 2024

# ETF Ranking Feature Settings
USE_ETF_RANKING = True
TOP_N_ETFS = 100  # Reduced for faster processing

# --- Data File Paths ---
try:
    with open(os.path.join(DATA_DIR, 'etf_costs.json'), 'r') as f:
        ETF_COSTS = json.load(f)
except FileNotFoundError:
    print(f"Warning: etf_costs.json not found in {DATA_DIR}. Using empty costs.")
    ETF_COSTS = {}

# GPU Settings
GPU_SETTINGS = {
    'use_gpu': True,  # GPU 사용 여부
    'gpu_id': 0,      # 사용할 GPU 번호 (0, 1, 2, ...)
    'force_cpu': False  # True로 설정하면 GPU가 있어도 CPU 사용
}

# Model Parameter Settings
MODEL_PARAMS = {
    # Common Parameters
    'max_weight': 0.7,
    'market_proxy_permno': 84398,  # SPY

    # ARIMAX Model Parameters
    'use_arima': False,
    'arima_params': {
        'lookback_months': 48,
        'tau': 1.0,
        'view_outperformance': 0.02 / 12,
    },

    # ETF Ranking Parameters
    'ranking_lookback_years': 5,  # Only use last N years for ranking features (None for all)
    'ranking_cache': True,        # Enable disk cache for monthly ranking results

    # TCN-SVR Model Parameters
    'use_tcn_svr': True,
    'tcn_svr_params': {
        'tau': 1.0,
                'lookback_window': 120,
        'lookback_window_step': 6,
        'num_channels_step': 8,
        # Lighter default model for faster monthly training
    # Model strength controls (adjust for heavier training)
    'num_channels': [64, 512],
    'kernel_size': 3,
    'dropout': 0.3,
    'base_uncertainty': 0.05,
    'epochs': 1500,
    'lr': 0.0001,
    'early_stopping_patience': 80,
        'early_stopping_min_delta': 0.00001,
        # Limit how much history to use for training per permno in daily rows (None = all)
        'train_window_rows': 720,
    # Optimization toggles
    'warm_start': False,
    'use_lag_features': True,
    # Keep at most this many lag features if set (e.g., 64). None uses all available.
    'max_lag_features': 64,
        # Hyperparameter tuning controls
        'tune_trials_per_month': 5,
        # Tune every K months (reduce monthly overhead). 1 = every month
        'tune_every_k_months': 1,
        # Keep modest parallelism to avoid oversubscription on shared machines
        # Use 1 for GPU, or 2~4 for CPU-only environments
        'optuna_n_jobs': 1,
        # Optuna search space parameters
        'lookback_window_min': 120,
        'num_channels_min': 128,
        'num_channels_max': 1024,
        'dropout_min': 0.1,
        'dropout_max': 0.5,
        'dropout_step': 0.1,
        'svr_C_min': 1.0,
        'svr_C_max': 100.0,
        'svr_gamma_min': 0.01,
        'svr_gamma_max': 1.0,
        'loss_function': 'huber',
        'huber_delta': 1.0,
        'huber_delta_min': 0.1,
        'huber_delta_max': 1.5
    }}

# Data Caching Settings
USE_CACHING = True

# Feature Stationarity Check Settings
CHECK_STATIONARITY = True
STATIONARITY_SIGNIFICANCE_LEVEL = 0.05

# Dynamic Omega (Ω) Calculation Settings
USE_DYNAMIC_OMEGA = True

# If True, constrains the portfolio's monthly turnover to be below MAX_TURNOVER
USE_TURNOVER_CONSTRAINT = True
MAX_TURNOVER = 0.20  # 20%로 더 엄격하게 제한


# Benchmark list settings for comparative analysis
# None represents a 1/N portfolio (equal weight on all assets) benchmark
BENCHMARK_PERMNOS = [84398, 88320, None]  # SPY, QQQ

def get_model_params(use_tuned_params=True):
    """
    Returns the model parameters, optionally updated with tuned hyperparameters.

    Args:
        use_tuned_params (bool): If True, tries to load and apply tuned params
                                 from 'best_tcn_svr_params.json'.

    Returns:
        dict: A dictionary of model parameters.
    """
    model_params = deepcopy(MODEL_PARAMS)

    if use_tuned_params and model_params.get('use_tcn_svr', False):
        best_params_path = os.path.join(OUTPUT_DIR, 'best_tcn_svr_params.json')
        try:
            with open(best_params_path, 'r') as f:
                best_params = json.load(f)
            
            # Update the nested tcn_svr_params dictionary
            model_params['tcn_svr_params'].update(best_params)
        except FileNotFoundError:
            # This is not an error, just means we use defaults.
            pass
        except Exception:
            # Could be a JSON decoding error, etc. Fall back to defaults.
            pass
            
    return model_params
