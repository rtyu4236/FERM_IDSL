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
START_YEAR = 2023
END_YEAR = 2023

# ETF Ranking Feature Settings
USE_ETF_RANKING = True
TOP_N_ETFS = 50

# --- Data File Paths ---
try:
    with open(os.path.join(DATA_DIR, 'etf_costs.json'), 'r') as f:
        ETF_COSTS = json.load(f)
except FileNotFoundError:
    print(f"Warning: etf_costs.json not found in {DATA_DIR}. Using empty costs.")
    ETF_COSTS = {}

# Model Parameter Settings
MODEL_PARAMS = {
    # Common Parameters
    'max_weight': 0.7,
    'market_proxy_permno': 84398,  # SPY

    # ARIMAX Model Parameters
    'use_arima': False,
    'arima_params': {
        'lookback_months': 48,
        'tau': 2.0,
        'view_outperformance': 0.02 / 12,
    },

    # TCN-SVR Model Parameters
    'use_tcn_svr': True,
    'tcn_svr_params': {
        'tau': 0.025,
        'lookback_window': 48,
        'num_channels': [32, 32],
        'kernel_size': 3,
        'dropout': 0.2,
        'base_uncertainty': 0.05,
        'epochs': 2,  # Default epochs
        'early_stopping_patience': 2,
        'early_stopping_min_delta': 0.0001
    }
}

# Data Caching Settings
USE_CACHING = False

# Feature Stationarity Check Settings
CHECK_STATIONARITY = True
STATIONARITY_SIGNIFICANCE_LEVEL = 0.05

# Dynamic Omega (Ω) Calculation Settings
USE_DYNAMIC_OMEGA = True

# If True, constrains the portfolio's monthly turnover to be below MAX_TURNOVER
USE_TURNOVER_CONSTRAINT = False
MAX_TURNOVER = 0.40

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
