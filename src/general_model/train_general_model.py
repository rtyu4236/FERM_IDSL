import pandas as pd
import numpy as np
import os
import json
import traceback
from src.data_processing import manager as data_manager
from src.models import view_generator_new as ml_view_generator
from src.models.black_litterman import get_current_universe
import quantstats as qs
from config import settings as config
from src.utils.logger import logger
from src.tuning import tcn_svr_tuner
from src.models.etf_quant_ranker import ETFQuantRanker, create_etf_universe_from_daily
from src.models.tcn_svr import TCN_SVR_Model
from src.models.view_generator_new import generate_permno_tensors
import torch

def get_backtest_dates(all_dates_in_df, start_year, end_year):
    backtest_dates_in_range = all_dates_in_df[
        (all_dates_in_df.year >= start_year) & (all_dates_in_df.year <= end_year)
    ]
    calendar_month_ends = backtest_dates_in_range.to_period('M').unique().to_timestamp(how='end')
    backtest_dates = []
    all_dates_set = set(all_dates_in_df)
    
    for date in calendar_month_ends:
        current_date = date.normalize()
        # 월말이 데이터에 없을 경우, 가장 가까운 과거의 날짜를 찾음
        loop_count = 0
        while current_date not in all_dates_set:
            current_date -= pd.DateOffset(days=1)
            loop_count += 1
            if loop_count > 365:  # 무한 루프 방지
                logger.warning(f"Could not find a valid date for month end {date.strftime('%Y-%m')}. Skipping this month.")
                current_date = None
                break
        if current_date:
            backtest_dates.append(current_date)
    # Ensure chronological order of backtest dates
    backtest_dates = pd.DatetimeIndex(backtest_dates).unique().sort_values()
    return backtest_dates

def train_general_tcn_svr(daily_df, monthly_df, vix_df, ff_df, all_permnos, train_start_year, train_end_year, model_params, use_etf_ranking, run_rolling_tune, liquid_universe_dict=None):
    logger.info("[train_general_model] Function entry.")
    qs.extend_pandas()

    ml_features_df = data_manager.create_daily_feature_dataset_for_tcn(daily_df, vix_df, ff_df)
    logger.info(f"[run_backtest] ML features created: ml_features_df shape={ml_features_df.shape}")

    ranker = None
    if use_etf_ranking:
        logger.info("Initializing ETFQuantRanker...")
        ranker = ETFQuantRanker()
        logger.info("ETFQuantRanker initialized.")

    all_dates_in_df = pd.to_datetime(ml_features_df['date'].unique()).normalize()
    backtest_dates = get_backtest_dates(all_dates_in_df, train_start_year, train_end_year)
    
    logger.info(f"[run_backtest] Backtest dates range: {backtest_dates.min()} to {backtest_dates.max()}, total {len(backtest_dates)} dates.")
    
    bl_returns = []

    # For detailed exports
    liquidity_log_records = []  # [{'as_of': date, 'before_count': int, 'after_count': int}]
    ranking_selected_per_month = {} if use_etf_ranking else None

    # 모델 훈련 용 텐서 담을 리스트 준비
    all_X_train_tensors_list = []
    all_y_indicators_tensors_list = []
    all_y_returns_seq_list = []

    for idx, analysis_date in enumerate(backtest_dates[:-1]):
        logger.info(f"\n--- Processing {analysis_date.strftime('%Y-%m')} ---")

        # 1) 유동성 필터 전/후 개수 로깅 및 후보군 생성
        pre_liq_count = len(all_permnos)
        if liquid_universe_dict is not None:
            # Align analysis_date to its month-end to match dict keys and handle edge cases safely
            month_end_key = (pd.Timestamp(analysis_date) + pd.offsets.MonthEnd(0)).normalize()
            available_keys = sorted(liquid_universe_dict.keys())
            if month_end_key in liquid_universe_dict:
                key = month_end_key
            else:
                # Find the latest available key not after month_end_key; if none, fall back to earliest key
                prior_keys = [d for d in available_keys if d <= month_end_key]
                if prior_keys:
                    key = prior_keys[-1]
                else:
                    # analysis_date precedes the first available liquidity snapshot; use the earliest
                    key = available_keys[0]
            candidate_permnos = liquid_universe_dict.get(key, all_permnos)
        else:
            candidate_permnos = all_permnos
        after_liq_count = len(candidate_permnos)
        logger.info(f"[Rebalance {analysis_date.strftime('%Y-%m')}] Liquidity filter: before={pre_liq_count}, after={after_liq_count}")
        liquidity_log_records.append({
            'as_of': analysis_date.normalize(),
            'before_count': int(pre_liq_count),
            'after_count': int(after_liq_count)
        })
        
        # 2) 유동성 필터를 통과한 전체 ETF 사용
        universe_for_month = candidate_permnos
        logger.info(f"[Rebalance {analysis_date.strftime('%Y-%m')}] Ranking disabled. Using candidate universe ({len(universe_for_month)}).")


        if not universe_for_month:
            logger.warning(f"Universe for {analysis_date.strftime('%Y-%m')} is empty. Skipping month.")
            bl_returns.append(pd.Series([0.0], index=[analysis_date + pd.offsets.MonthEnd(1)]))
            continue

        # Check tuning cadence (every K months)
        if run_rolling_tune and model_params.get('use_tcn_svr', False):
            logger.info(f"Running hyperparameter tuning for {analysis_date.strftime('%Y-%m')}...")
            tcn_svr_tuner.run_tuning(ml_features_df, n_trials=config.MODEL_PARAMS['tcn_svr_params']['tune_trials_per_month'], end_date=analysis_date)
            current_model_params = config.get_model_params()
            active_model_params = current_model_params['tcn_svr_params']
            logger.info(f"Tuned parameters for {analysis_date.strftime('%Y-%m')}:\n{json.dumps(current_model_params, indent=2)}")
        else:
            current_model_params = model_params
            active_model_params = current_model_params.get('tcn_svr_params')

        monthly_df_filtered = monthly_df[monthly_df['permno'].isin(universe_for_month)]


        current_permnos, _ = get_current_universe(all_returns_df=monthly_df_filtered, analysis_date=analysis_date, lookback_months=active_model_params.get('lookback_window', 24))
        
        full_feature_df = ml_features_df[ml_features_df['permno'].isin(current_permnos)]

        
        logger.info(f"[generate_tcn_svr_views] Gathering data for TCN_SVR training...")
        
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
        train_window_rows = model_params.get('train_window_rows')
        num_assets = len(current_permnos)

        for i, permno in enumerate(current_permnos):
            # logger.info(f"[generate_tcn_svr_views] Processing permno: {permno}")
            
            X_train_tensor, y_indicators_tensor, y_returns_seq, _ = generate_permno_tensors(
                permno, full_feature_df, indicator_features, all_features, train_window_rows, model_params
            )
            
            all_X_train_tensors_list.append(X_train_tensor)
            all_y_indicators_tensors_list.append(y_indicators_tensor)
            all_y_returns_seq_list.append(y_returns_seq)
        
    all_X_train_tensors = torch.from_numpy(np.concatenate(all_X_train_tensors_list, axis=0))
    all_y_indicator_tensors = torch.from_numpy(np.concatenate(all_y_indicators_tensors_list, axis=0))
    all_y_returns_seq = np.hstack(all_y_returns_seq_list)
    logger.info(f"[generate_tcn_svr_views] Combined training data shapes: all_X_train_tensors shape={all_X_train_tensors.shape}, all_y_indicator_tensors shape={all_y_indicator_tensors.shape}, all_y_returns_seq shape={all_y_returns_seq.shape}")
    
    logger.info("Shuffling training data...")
    perm = torch.randperm(all_X_train_tensors.size(0))
    all_X_train_tensors = all_X_train_tensors[perm]
    all_y_indicator_tensors = all_y_indicator_tensors[perm]
    all_y_returns_seq = all_y_returns_seq[perm.numpy()]
    
    model = TCN_SVR_Model(
        input_size=15, # len(all_features),
        output_size=7,# len(indicator_features),
        num_channels=model_params['num_channels'],
        kernel_size=model_params['kernel_size'],
        dropout=model_params['dropout'],
        lookback_window=model_params['lookback_window'],
        svr_C=model_params.get('svr_C', 1.0),
        svr_gamma=model_params.get('svr_gamma', 'scale'),
        lr=model_params.get('lr', 0.001) # Pass the tuned learning rate
    )

    model.fit(all_X_train_tensors, all_y_indicator_tensors, all_y_returns_seq,
        epochs=model_params.get('epochs', 50),
        patience=model_params.get('early_stopping_patience', 10),
        min_delta=model_params.get('early_stopping_min_delta', 0.0001))