import os
import pandas as pd
import quantstats as qs
from typing import Tuple


def generate_performance_reports(
    cumulative_results_df: pd.DataFrame,
    output_dir: str,
    risk_free_rate: float = 0.02
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    누적 수익률 데이터로부터 2가지 핵심 성과 보고서 테이블을 생성하고 저장합니다.

    Args:
        cumulative_results_df (pd.DataFrame):
            - Index: datetime
            - Columns: 전략 및 벤치마크 이름 (예: ['BL_ML_Strategy', 'SPY', 'QQQ', '1/N Portfolio'])
            - Values: 누적 수익률 (예: 1.0, 1.01, 1.03...)
        output_dir (str): 결과 CSV 파일을 저장할 디렉토리 경로.
        risk_free_rate (float): 샤프 지수 계산에 사용할 무위험 수익률(연 단위).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            1. 주요 성과 지표 요약 테이블 (performance_summary_df)
            2. 연도별 수익률 테이블 (yearly_returns_df)
    """

    os.makedirs(output_dir, exist_ok=True)
    # Part 1) 월간 수익률로 변환
    monthly_returns_df = cumulative_results_df.pct_change().dropna(how='all')

    # 성과 지표 계산
    metrics_dict = {}
    for col in monthly_returns_df.columns:
        series = monthly_returns_df[col].dropna()
        if series.empty:
            # 비어있으면 NaN으로 채움
            metrics_dict[col] = {
                'Annualized Return': float('nan'),
                'Annualized Volatility': float('nan'),
                'Sharpe Ratio': float('nan'),
                'Max Drawdown': float('nan'),
                'Calmar Ratio': float('nan'),
            }
            continue

        try:
            ann_ret = qs.stats.cagr(series)
        except Exception:
            ann_ret = float('nan')
        try:
            ann_vol = qs.stats.annualized_volatility(series)
        except Exception:
            ann_vol = float('nan')
        try:
            sharpe = qs.stats.sharpe(series, rf=risk_free_rate)
        except Exception:
            sharpe = float('nan')
        try:
            mdd = qs.stats.max_drawdown(series)
        except Exception:
            mdd = float('nan')
        try:
            calmar = qs.stats.calmar(series)
        except Exception:
            calmar = float('nan')

        metrics_dict[col] = {
            'Annualized Return': ann_ret,
            'Annualized Volatility': ann_vol,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': mdd,
            'Calmar Ratio': calmar,
        }

    performance_summary_numeric = pd.DataFrame.from_dict(metrics_dict, orient='index')
    performance_summary_numeric = performance_summary_numeric[
        ['Annualized Return', 'Annualized Volatility', 'Sharpe Ratio', 'Max Drawdown', 'Calmar Ratio']
    ]

    # 서식 지정: 퍼센트 및 소수점 자리수
    def pct_fmt(x):
        try:
            return f"{x * 100:.2f}%" if pd.notnull(x) else ""
        except Exception:
            return ""

    def ratio_fmt(x):
        try:
            return f"{x:.2f}" if pd.notnull(x) else ""
        except Exception:
            return ""

    performance_summary_df = performance_summary_numeric.copy()
    for col in ['Annualized Return', 'Annualized Volatility', 'Max Drawdown']:
        performance_summary_df[col] = performance_summary_df[col].apply(pct_fmt)
    for col in ['Sharpe Ratio', 'Calmar Ratio']:
        performance_summary_df[col] = performance_summary_df[col].apply(ratio_fmt)

    # CSV 저장
    perf_path = os.path.join(output_dir, 'performance_summary.csv')
    performance_summary_df.to_csv(perf_path)

    # Part 2) 연도별 수익률 계산 및 서식
    yearly_returns = monthly_returns_df.resample('Y').apply(lambda x: (1 + x).prod() - 1)
    # 연도만 인덱스로 표시
    yearly_returns.index = yearly_returns.index.year
    yearly_returns.index.name = 'Year'

    yearly_returns_df = yearly_returns.applymap(lambda v: f"{v * 100:.2f}%" if pd.notnull(v) else "")

    yearly_path = os.path.join(output_dir, 'yearly_returns.csv')
    yearly_returns_df.to_csv(yearly_path)

    return performance_summary_df, yearly_returns_df
