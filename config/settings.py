import os
import json

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
CACHE_DIR = os.path.join(ROOT_DIR, 'cache')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

START_YEAR = 2004
END_YEAR = 2024

# ETF 랭킹 기능 사용 여부 설정
USE_ETF_RANKING = True
TOP_N_ETFS = 50

# 각 ETF의 연간 운용 보수(expense_ratio)와 평균 매수-매도 스프레드(trading_cost_spread)를 정의
ETF_COSTS = json.load(open(os.path.join(DATA_DIR, 'etf_costs.json')))
# ETF_COSTS = {
#     # US Equity (Broad)
#     'SPY': {'expense_ratio': 0.0009, 'trading_cost_spread': 0.0001},
#     'QQQ': {'expense_ratio': 0.0020, 'trading_cost_spread': 0.0001},
#     'QQQQ': {'expense_ratio': 0.0020, 'trading_cost_spread': 0.0001}, # QQQ의 과거 티커
#     'IWM': {'expense_ratio': 0.0019, 'trading_cost_spread': 0.0001},
#     'VTI': {'expense_ratio': 0.0003, 'trading_cost_spread': 0.0001},
#     'MDY': {'expense_ratio': 0.0023, 'trading_cost_spread': 0.0002},
#     'DIA': {'expense_ratio': 0.0016, 'trading_cost_spread': 0.0002},
#     'RSP': {'expense_ratio': 0.0020, 'trading_cost_spread': 0.0001},
#     # US Equity (Factor)
#     'VTV': {'expense_ratio': 0.0004, 'trading_cost_spread': 0.0001},
#     'VUG': {'expense_ratio': 0.0004, 'trading_cost_spread': 0.0002},
#     'MTUM': {'expense_ratio': 0.0015, 'trading_cost_spread': 0.0002},
#     'QUAL': {'expense_ratio': 0.0015, 'trading_cost_spread': 0.0001},
#     'USMV': {'expense_ratio': 0.0015, 'trading_cost_spread': 0.0001},
#     'SCHD': {'expense_ratio': 0.0006, 'trading_cost_spread': 0.0004},
#     # US Equity (Sector)
#     'XLK': {'expense_ratio': 0.0010, 'trading_cost_spread': 0.0001}, # 2023-12부터 0.0008
#     'XLV': {'expense_ratio': 0.0010, 'trading_cost_spread': 0.0001}, # 2023-12부터 0.0008
#     'XLE': {'expense_ratio': 0.0010, 'trading_cost_spread': 0.0001}, # Assumed
#     'XLF': {'expense_ratio': 0.0010, 'trading_cost_spread': 0.0001}, # Assumed
#     'XLI': {'expense_ratio': 0.0010, 'trading_cost_spread': 0.0001}, # Assumed
#     'XLP': {'expense_ratio': 0.0010, 'trading_cost_spread': 0.0001}, # Assumed
#     # International Equity
#     'VEA': {'expense_ratio': 0.0005, 'trading_cost_spread': 0.0002},
#     'VWO': {'expense_ratio': 0.0008, 'trading_cost_spread': 0.0002},
#     'EWJ': {'expense_ratio': 0.0050, 'trading_cost_spread': 0.0001},
#     'VGK': {'expense_ratio': 0.0008, 'trading_cost_spread': 0.0001},
#     # Fixed Income
#     'AGG': {'expense_ratio': 0.0003, 'trading_cost_spread': 0.0001},
#     'TLT': {'expense_ratio': 0.0015, 'trading_cost_spread': 0.0001},
#     'TIP': {'expense_ratio': 0.0018, 'trading_cost_spread': 0.0004},
#     'HYG': {'expense_ratio': 0.0049, 'trading_cost_spread': 0.0001},
#     # Commodities
#     'GLD': {'expense_ratio': 0.0040, 'trading_cost_spread': 0.0001},
#     'SLV': {'expense_ratio': 0.0050, 'trading_cost_spread': 0.0002}, # Assumed
#     # Real Estate
#     'VNQ': {'expense_ratio': 0.0012, 'trading_cost_spread': 0.0001},
# }

# 모델 파라미터 설정
MODEL_PARAMS = {
    'lookback_months': 12,
    'tau': 2.0,
    'max_weight': 0.7, # 개별 자산 최대 비중
    'market_proxy_ticker': 'SPY',
    'view_outperformance': 0.02 / 12
}

# 데이터 캐싱 설정 
USE_CACHING = True

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
