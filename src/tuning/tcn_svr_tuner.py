import optuna
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import json
import os

from src.models.tcn_svr import TCN_SVR_Model
from src.data_processing import manager as data_manager
from config import settings as config
from src.utils.logger import logger

class TCN_SVR_Objective:
    def __init__(self, full_feature_df, config_model_params, end_date=None):
        self.full_feature_df = full_feature_df
        self.config_model_params = config_model_params
        self.end_date = end_date
        self.indicator_features = ['ATRr_14', 'ADX_14', 'EMA_20', 'MACD_12_26_9', 'SMA_50', 'HURST', 'RSI_14']
        self.other_features = ['realized_vol', 'intra_month_mdd', 'avg_vix', 'vol_of_vix', 'Mkt-RF', 'SMB', 'HML', 'RF']
        self.lag_features = [col for col in full_feature_df.columns if '_lag_' in col]
        self.all_features = self.indicator_features + self.other_features + self.lag_features

    def _create_sequences(self, data, lookback_window):
        xs, ys = [], []
        for i in range(len(data) - lookback_window):
            x = data[i:(i + lookback_window)]
            y = data[i + lookback_window]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    def __call__(self, trial):
        tcn_lookback_window = trial.suggest_int('lookback_window', 12, 48, step=6)
        tcn_num_channels_layer1 = trial.suggest_int('num_channels_layer1', 8, 64, step=16)
        tcn_num_channels_layer2 = trial.suggest_int('num_channels_layer2', 8, 64, step=16)
        tcn_kernel_size = trial.suggest_int('kernel_size', 1, 5)
        tcn_dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
        tcn_epochs = trial.suggest_int('epochs', 30, 100, step=10)
        tcn_lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
        svr_C = trial.suggest_float('svr_C', 1.0, 100.0, log=True)
        svr_gamma = trial.suggest_float('svr_gamma', 0.01, 1.0, log=True)

        # Use data up to the specified end_date for tuning
        if self.end_date:
            tuning_df = self.full_feature_df[self.full_feature_df['date'] < self.end_date].copy()
        else:
            tuning_df = self.full_feature_df.copy()

        if tuning_df.empty:
            logger.warning("Tuning data is empty.")
            return float('inf')

        target_ticker = 'SPY'
        if target_ticker not in tuning_df['ticker'].unique():
            target_ticker = tuning_df['ticker'].unique()[0]
        
        ticker_df = tuning_df[tuning_df['ticker'] == target_ticker].copy()
        ticker_df = ticker_df.dropna(subset=self.all_features + ['target_return'])

        if len(ticker_df) < tcn_lookback_window + 1:
            logger.warning(f"Not enough data for {target_ticker} to tune with lookback {tcn_lookback_window}.")
            return float('inf')

        X_data = ticker_df[self.all_features].values
        y_indicators = ticker_df[self.indicator_features].values
        y_returns = ticker_df['target_return'].values

        X_seq, y_seq_combined = self._create_sequences(np.hstack([X_data, y_indicators, y_returns.reshape(-1,1)]), tcn_lookback_window)
        
        X_train_seq = X_seq[:, :, :-len(self.indicator_features)-1]
        y_train_indicators_seq = y_seq_combined[:, -len(self.indicator_features)-1:-1]
        y_train_returns_seq = y_seq_combined[:, -1]

        X_train_tensor = torch.from_numpy(X_train_seq).float()
        y_train_indicators_tensor = torch.from_numpy(y_train_indicators_seq).float()

        model = TCN_SVR_Model(
            input_size=len(self.all_features),
            output_size=len(self.indicator_features),
            num_channels=[tcn_num_channels_layer1, tcn_num_channels_layer2],
            kernel_size=tcn_kernel_size,
            dropout=tcn_dropout,
            lookback_window=tcn_lookback_window,
            svr_C=svr_C,
            svr_gamma=svr_gamma
        )
        model.optimizer = torch.optim.Adam(model.net.parameters(), lr=tcn_lr)

        model.fit(X_train_tensor, y_train_indicators_tensor, y_train_returns_seq,
                  epochs=tcn_epochs,
                  patience=self.config_model_params['tcn_svr_params']['early_stopping_patience'],
                  min_delta=self.config_model_params['tcn_svr_params']['early_stopping_min_delta'])
        
        final_val_loss = model.best_loss
        return final_val_loss

def run_tuning(full_feature_df, n_trials=50, end_date=None):
    logger.info(f"TCN-SVR Hyperparameter Tuning started. Using data up to {end_date if end_date else 'the end'}.")
    
    objective = TCN_SVR_Objective(full_feature_df, config.MODEL_PARAMS, end_date=end_date)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)

    logger.info("Tuning finished.")
    logger.info(f"Best trial for period ending {end_date}:")
    logger.info(f"  Value: {study.best_value}")
    logger.info(f"  Params: {study.best_params}")

    best_params = study.best_params
    tcn_params = {
        'lookback_window': best_params.get('lookback_window'),
        'num_channels': [best_params.get('num_channels_layer1'), best_params.get('num_channels_layer2')],
        'kernel_size': best_params.get('kernel_size'),
        'dropout': best_params.get('dropout'),
        'epochs': best_params.get('epochs'),
        'lr': best_params.get('lr'),
        'svr_C': best_params.get('svr_C'),
        'svr_gamma': best_params.get('svr_gamma'),
    }

    output_path = os.path.join(config.OUTPUT_DIR, 'best_tcn_svr_params.json')
    try:
        with open(output_path, 'w') as f:
            json.dump(tcn_params, f, indent=4)
        logger.info(f"Best parameters saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save best parameters: {e}")

    logger.info("TCN-SVR Hyperparameter Tuning completed.")
    return tcn_params

if __name__ == '__main__':
    logger.info("Loading data for tuning...")
    daily_df, monthly_df, vix_df, ff_df, all_tickers = data_manager.load_raw_data()
    full_feature_df = data_manager.create_daily_feature_dataset_for_tcn(daily_df, vix_df, ff_df)
    logger.info("Data for tuning loaded.")
    run_tuning(full_feature_df, n_trials=50)
