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

# --- 주요 설정 ---
START_YEAR = 2005
END_YEAR = 2023

# ETF 랭킹 기능 사용 여부 설정
USE_ETF_RANKING = True
TOP_N_ETFS = 50

# --- 데이터 파일 경로 ---
try:
    with open(os.path.join(DATA_DIR, 'etf_costs.json'), 'r') as f:
        ETF_COSTS = json.load(f)
except FileNotFoundError:
    print(f"Warning: etf_costs.json not found in {DATA_DIR}. Using empty costs.")
    ETF_COSTS = {}

# 모델 파라미터 설정
MODEL_PARAMS = {
    # 공통 파라미터
    'max_weight': 0.7,
    'market_proxy_ticker': 'SPY',

    # ARIMAX 모델 파라미터
    'use_arima': False,
    'arima_params': {
        'lookback_months': 48,
        'tau': 2.0,
        'view_outperformance': 0.02 / 12,
    },

    # TCN-SVR 모델 파라미터
    'use_tcn_svr': True,
    'tcn_svr_params': {
        'tau': 0.025,
        'lookback_window': 48,
        'num_channels': [32, 32],
        'kernel_size': 3,
        'dropout': 0.2,
        'base_uncertainty': 0.05,
        'epochs': 50, # Default epochs
        'early_stopping_patience': 10,
        'early_stopping_min_delta': 0.0001
    }
}

# 데이터 캐싱 설정 
USE_CACHING = False

# 피처 정상성 검증 설정 
CHECK_STATIONARITY = True
STATIONARITY_SIGNIFICANCE_LEVEL = 0.05

# 동적 오메가(Ω) 계산 설정
USE_DYNAMIC_OMEGA = True

# 자산 그룹 및 제약 조건 설정
ASSET_GROUPS = {
    'US_EQUITY': ['SPY', 'QQQ', 'QQQQ', 'IWM', 'VTI', 'MDY', 'DIA', 'RSP', 'VTV', 'VUG', 'MTUM', 'QUAL', 'USMV', 'SCHD'],
    'US_SECTOR': ['XLK', 'XLV', 'XLE', 'XLF', 'XLI', 'XLP'],
    'INTL_EQUITY': ['VEA', 'VWO', 'EWJ', 'VGK'],
    'FIXED_INCOME': ['AGG', 'TLT', 'TIP', 'HYG'],
    'COMMODITIES': ['GLD', 'SLV'],
    'REAL_ESTATE': ['VNQ']
}

GROUP_CONSTRAINTS = {
    # 그룹명: {'min': 최소비중, 'max': 최대비중}
    'US_EQUITY': {'min': 0.10, 'max': 0.70},
    'US_SECTOR': {'min': 0.00, 'max': 0.50},
    'INTL_EQUITY': {'min': 0.00, 'max': 0.40},
    'FIXED_INCOME': {'min': 0.10, 'max': 0.60},
    'COMMODITIES': {'min': 0.00, 'max': 0.4},
    'REAL_ESTATE': {'min': 0.00, 'max': 0.4},
}

# True로 설정하면 포트폴리오의 월간 회전율을 MAX_TURNOVER 이하로 제한
USE_TURNOVER_CONSTRAINT = False
MAX_TURNOVER = 0.40

# 비교 분석을 위한 벤치마크 리스트 설정
# None은 1/N 포트폴리오(모든 자산에 동일 비중) 벤치마크를 의미
BENCHMARK_TICKERS = ['SPY', 'QQQ', None]

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