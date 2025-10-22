import os
import pandas as pd
import quantstats as qs
from typing import Tuple
import numpy as np


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
    
    # 데이터 검증
    if cumulative_results_df.empty:
        raise ValueError("cumulative_results_df가 비어있습니다.")
    
    if not isinstance(cumulative_results_df.index, pd.DatetimeIndex):
        raise ValueError("cumulative_results_df의 인덱스는 DatetimeIndex여야 합니다.")
    
    # 실제 데이터 기간 계산
    data_start_date = cumulative_results_df.index.min()
    data_end_date = cumulative_results_df.index.max()
    total_months = len(cumulative_results_df)
    
    print(f"데이터 기간: {data_start_date.strftime('%Y-%m-%d')} ~ {data_end_date.strftime('%Y-%m-%d')}")
    print(f"총 데이터 포인트: {total_months}개")
    print(f"컬럼: {list(cumulative_results_df.columns)}")
    
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
            # 실제 데이터 기간을 기반으로 연간 수익률 계산
            if len(series) > 0 and not series.isna().all():
                # 월간 수익률을 누적 수익률로 변환
                cumulative_return = (1 + series).prod()
                # 실제 데이터 기간을 기반으로 연간화
                actual_months = len(series)
                if actual_months > 0 and cumulative_return > 0:
                    ann_ret = (cumulative_return ** (12 / actual_months)) - 1
                else:
                    ann_ret = float('nan')
            else:
                ann_ret = float('nan')
        except (ValueError, ZeroDivisionError, OverflowError):
            ann_ret = float('nan')
        try:
            # 월간 변동성을 연간 변동성으로 변환
            if len(series) > 1 and not series.isna().all():
                monthly_vol = series.std()
                if pd.notna(monthly_vol) and monthly_vol > 0:
                    ann_vol = monthly_vol * (12 ** 0.5)  # sqrt(12)로 연간화
                else:
                    ann_vol = float('nan')
            else:
                ann_vol = float('nan')
        except (ValueError, ZeroDivisionError):
            ann_vol = float('nan')
            
        try:
            # Sharpe Ratio = (연간 수익률 - 무위험 수익률) / 연간 변동성
            if (pd.notna(ann_ret) and pd.notna(ann_vol) and 
                ann_vol != 0 and ann_vol > 0):
                sharpe = (ann_ret - risk_free_rate) / ann_vol
            else:
                sharpe = float('nan')
        except (ValueError, ZeroDivisionError):
            sharpe = float('nan')
        try:
            # 최대 낙폭 계산
            if len(series) > 0 and not series.isna().all():
                mdd = qs.stats.max_drawdown(series)
                if pd.isna(mdd):
                    mdd = float('nan')
            else:
                mdd = float('nan')
        except (ValueError, ZeroDivisionError):
            mdd = float('nan')
            
        try:
            # Calmar Ratio = 연간 수익률 / 최대 낙폭
            if (pd.notna(ann_ret) and pd.notna(mdd) and 
                mdd != 0 and abs(mdd) > 0):
                calmar = ann_ret / abs(mdd)  # mdd는 음수이므로 절댓값 사용
            else:
                calmar = float('nan')
        except (ValueError, ZeroDivisionError):
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
    # 실제 데이터의 연도별 분포를 고려하여 연간 수익률 계산
    yearly_returns_dict = {}
    
    for year in range(data_start_date.year, data_end_date.year + 1):
        year_data = monthly_returns_df[monthly_returns_df.index.year == year]
        if not year_data.empty:
            # 해당 연도의 월간 수익률을 복리 계산하여 연간 수익률 계산
            year_returns = {}
            for col in year_data.columns:
                col_data = year_data[col].dropna()
                if len(col_data) > 0:
                    try:
                        # 복리 계산으로 연간 수익률 계산
                        cumulative_return = (1 + col_data).prod()
                        year_returns[col] = cumulative_return - 1
                    except (ValueError, ZeroDivisionError):
                        year_returns[col] = np.nan
                else:
                    year_returns[col] = np.nan
            yearly_returns_dict[year] = year_returns
    
    yearly_returns = pd.DataFrame.from_dict(yearly_returns_dict, orient='index')
    yearly_returns.index.name = 'Year'

    yearly_returns_df = yearly_returns.applymap(lambda v: f"{v * 100:.2f}%" if pd.notnull(v) else "")

    yearly_path = os.path.join(output_dir, 'yearly_returns.csv')
    yearly_returns_df.to_csv(yearly_path)

    return performance_summary_df, yearly_returns_df
