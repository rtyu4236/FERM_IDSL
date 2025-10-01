import backtester
import warnings
import config
import logger_setup 
warnings.simplefilter(action='ignore', category=FutureWarning)

if __name__ == '__main__':
    """
    메인 진입점(Entry Point).

    - 전체 백테스팅 프로세스를 시작하고 조율하는 최상위 컨트롤러.
    - `config.py`의 모든 설정을 불러와 백테스터에 전달.
    """
    # `config.py`에 정의된 파라미터를 `run_backtest` 함수에 전달하여 백테스팅 실행
    backtester.run_backtest(
        start_year=config.START_YEAR, 
        end_year=config.END_YEAR,
        etf_costs=config.ETF_COSTS,
        model_params=config.MODEL_PARAMS,
        benchmark_tickers=config.BENCHMARK_TICKERS
    )
    