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

START_YEAR = 2024
END_YEAR = 2024

USE_ETF_RANKING = True
TOP_N_ETFS = 100

try:
    with open(os.path.join(DATA_DIR, 'etf_costs.json'), 'r') as f:
        ETF_COSTS = json.load(f)
except FileNotFoundError:
    print(f"Warning: etf_costs.json not found in {DATA_DIR}. Using empty costs.")
    ETF_COSTS = {}

GPU_SETTINGS = {
    'use_gpu': True,
    'gpu_id': 0,
    'force_cpu': False
}

MODEL_PARAMS = {
    'max_weight': 0.4,
    'market_proxy_permno': 84398,
    
    'use_arima': False,
    'arima_params': {
        'lookback_months': 48,
        'tau': 1.0,
        'view_outperformance': 0.02 / 12,
    },
    
    'ranking_lookback_years': 5,
    'ranking_cache': True,
    
    'use_tcn_svr': True,
    'tcn_svr_params': {
        'tau': 1.0,
        'lookback_window': 120,
        'lookback_window_step': 6,
        'num_channels_step': 8,
        'num_channels': [64, 512],
        'kernel_size': 3,
        'dropout': 0.3,
        'base_uncertainty': 0.05,
        'epochs': 1500,
        'lr': 0.0001,
        'early_stopping_patience': 80,
        'early_stopping_min_delta': 0.00001,
        'train_window_rows': 720,
        'warm_start': False,
        'use_lag_features': True,
        'max_lag_features': 64,
        'tune_trials_per_month': 1,
        'tune_every_k_months': 1,
        'optuna_n_jobs': 1,
        'lookback_window_min': 120,
        'num_channels_min': 128,
        'num_channels_max': 768,
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
    }
}

USE_CACHING = True
CHECK_STATIONARITY = True
STATIONARITY_SIGNIFICANCE_LEVEL = 0.05
USE_DYNAMIC_OMEGA = True
USE_TURNOVER_CONSTRAINT = True
MAX_TURNOVER = 0.20

BENCHMARK_PERMNOS = [84398, 88320, None]

def get_model_params(use_tuned_params=True):
    model_params = deepcopy(MODEL_PARAMS)
    
    if use_tuned_params and model_params.get('use_tcn_svr', False):
        best_params_path = os.path.join(OUTPUT_DIR, 'best_tcn_svr_params.json')
        try:
            with open(best_params_path, 'r') as f:
                best_params = json.load(f)
            model_params['tcn_svr_params'].update(best_params)
        except FileNotFoundError:
            pass
        except Exception:
            pass
    
    return model_params
