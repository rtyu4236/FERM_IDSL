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
    os.makedirs(output_dir, exist_ok=True)
    
    # 데이터 검증
    if cumulative_results_df.empty:
        raise ValueError("cumulative_results_df가 비어있음.")
    if not isinstance(cumulative_results_df.index, pd.DatetimeIndex):
        raise ValueError("cumulative_results_df의 인덱스는 DatetimeIndex여야 함")
    
    # 데이터 기간간
    data_start_date = cumulative_results_df.index.min()
    data_end_date = cumulative_results_df.index.max()
    total_months = len(cumulative_results_df)
    print(f"데이터 기간: {data_start_date.strftime('%Y-%m-%d')} ~ {data_end_date.strftime('%Y-%m-%d')}")
    print(f"총 데이터 포인트: {total_months}개")
    print(f"컬럼: {list(cumulative_results_df.columns)}")
    
    monthly_returns_df = cumulative_results_df.pct_change().dropna(how='all')

    # 성과 지표 계산
    metrics_dict = {}
    for col in monthly_returns_df.columns:
        series = monthly_returns_df[col].dropna()
        if series.empty:
            metrics_dict[col] = {
                'Annualized Return': float('nan'),
                'Annualized Volatility': float('nan'),
                'Sharpe Ratio': float('nan'),
                'Max Drawdown': float('nan'),
                'Calmar Ratio': float('nan'),
            }
            continue

        try:
            # 연간 수익률 계산
            if len(series) > 0 and not series.isna().all():
                cumulative_return = (1 + series).prod()
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
            # 연간 변동성 계산
            if len(series) > 1 and not series.isna().all():
                monthly_vol = series.std()
                if pd.notna(monthly_vol) and monthly_vol > 0:
                    ann_vol = monthly_vol * (12 ** 0.5)
                else:
                    ann_vol = float('nan')
            else:
                ann_vol = float('nan')
        except (ValueError, ZeroDivisionError):
            ann_vol = float('nan')
            
        try:
            # sharpe ratio 계산
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
            # Calmar Ratio 계산
            if (pd.notna(ann_ret) and pd.notna(mdd) and 
                mdd != 0 and abs(mdd) > 0):
                calmar = ann_ret / abs(mdd)
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

    # 퍼센트 및 소수점 자리수 서식 지정 
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

    #---------------------------------------------------------------------------------
    yearly_returns_dict = {}
    
    for year in range(data_start_date.year, data_end_date.year + 1):
        year_data = monthly_returns_df[monthly_returns_df.index.year == year]
        if not year_data.empty:
            year_returns = {}
            for col in year_data.columns:
                col_data = year_data[col].dropna()
                if len(col_data) > 0:
                    try:
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
