import torch
import torch.nn as nn
from pytorch_tcn import TCN
from sklearn.svm import SVR
from src.utils.logger import logger


class LastTimeStep(nn.Module):
    """A custom PyTorch module to extract the last time step from a sequence output."""
    def forward(self, x):
        logger.debug(f"[LastTimeStep.forward] Function entry. Input x shape={x.shape}, dtype={x.dtype}")
        output = x[:, :, -1]
        logger.debug(f"[LastTimeStep.forward] Output shape={output.shape}, dtype={output.dtype}")
        return output

class TCN_SVR_Model:
    """
    A hybrid model combining a Temporal Convolutional Network (TCN) for feature extraction
    from time-series data and a Support Vector Regression (SVR) for final prediction.

    The TCN part is a PyTorch model that learns to predict future technical indicators.
    The SVR part is a scikit-learn model that takes the TCN\'s predicted indicators as input
    to predict the final return.
    """
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout, lookback_window, svr_C=1.0, svr_gamma='scale', lr=0.001):
        """
        Initializes the TCN and SVR models.

        Args:
            input_size (int): The number of input features for the TCN.
            output_size (int): The number of output indicators for the TCN to predict.
            num_channels (list): A list of integers defining the number of channels in each TCN layer.
            kernel_size (int): The size of the convolutional kernel.
            dropout (float): The dropout rate for regularization.
            lookback_window (int): The number of past time steps to use for prediction.
            svr_C (float): The C parameter for the SVR model.
            svr_gamma (str or float): The gamma parameter for the SVR model.
            lr (float): The learning rate for the TCN optimizer.
        """
        logger.info("[TCN_SVR_Model.__init__] Function entry.")
        logger.info(f"[TCN_SVR_Model.__init__] Input: input_size={input_size}, output_size={output_size}, num_channels={num_channels}, kernel_size={kernel_size}, dropout={dropout}, lookback_window={lookback_window}, svr_C={svr_C}, svr_gamma={svr_gamma}, lr={lr}")
        # Select device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"[TCN_SVR_Model.__init__] Using device: {self.device}")
        self.tcn_model = TCN(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        logger.info(f"[TCN_SVR_Model.__init__] TCN model initialized. Type: {type(self.tcn_model)}")
        self.net = nn.Sequential(
            self.tcn_model,
            LastTimeStep(), # Use the custom module
            nn.Linear(num_channels[-1], output_size)
        )
        logger.info(f"[TCN_SVR_Model.__init__] Sequential network initialized. Type: {type(self.net)}")
        # Move network to device
        self.net.to(self.device)
        self.use_amp = True # Enable mixed precision if supported
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.lookback_window = lookback_window

        self.svr_model = SVR(kernel='rbf', C=svr_C, gamma=svr_gamma)
        logger.info(f"[TCN_SVR_Model.__init__] SVR model initialized with C={svr_C} and gamma={svr_gamma}. Type: {type(self.svr_model)}")
        logger.info("[TCN_SVR_Model.__init__] Function exit.")

    def fit(self, X_train_tensor, y_train_indicators_tensor, y_train_returns_numpy, epochs=50, patience=10, min_delta=0.0001):
        logger.info("[TCN_SVR_Model.fit] Function entry.")
        logger.info(f"[TCN_SVR_Model.fit] Input: X_train_tensor shape={X_train_tensor.shape}, dtype={X_train_tensor.dtype}")
        logger.info(f"[TCN_SVR_Model.fit] Input: y_train_indicators_tensor shape={y_train_indicators_tensor.shape}, dtype={y_train_indicators_tensor.dtype}")
        logger.info(f"[TCN_SVR_Model.fit] Input: y_train_returns_numpy shape={y_train_returns_numpy.shape}, dtype={y_train_returns_numpy.dtype}")
        logger.info(f"[TCN_SVR_Model.fit] Training TCN for {epochs} epochs with patience={patience}, min_delta={min_delta}")

        from sklearn.model_selection import train_test_split
        MIN_SAMPLES_FOR_SPLIT = 5
        can_validate = X_train_tensor.shape[0] >= MIN_SAMPLES_FOR_SPLIT

        # Move training tensors to device and cast to half precision if AMP is enabled
        if self.device.type == 'cuda' and self.use_amp:
            X_train_tensor = X_train_tensor.to(self.device).half()
            y_train_indicators_tensor = y_train_indicators_tensor.to(self.device).half()
        else:
            X_train_tensor = X_train_tensor.to(self.device)
            y_train_indicators_tensor = y_train_indicators_tensor.to(self.device)

        if can_validate:
            X_train_tcn, X_val_tcn, y_train_tcn, y_val_tcn = train_test_split(
                X_train_tensor, y_train_indicators_tensor, test_size=0.2, shuffle=False
            )
            logger.info(f"[TCN_SVR_Model.fit] TCN train split: X_train_tcn shape={X_train_tcn.shape}, y_train_tcn shape={y_train_tcn.shape}")
            logger.info(f"[TCN_SVR_Model.fit] TCN val split: X_val_tcn shape={X_val_tcn.shape}, y_val_tcn shape={y_val_tcn.shape}")
        else:
            logger.warning(f"Not enough samples ({X_train_tensor.shape[0]}) to create a validation set. Skipping early stopping.")
            X_train_tcn = X_train_tensor
            y_train_tcn = y_train_indicators_tensor

        best_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        # 1. Train the TCN model
        scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == 'cuda' and self.use_amp))

        for epoch in range(epochs):
            logger.debug(f"[TCN_SVR_Model.fit] Epoch {epoch+1}/{epochs} - Starting TCN training step.")
            self.net.train()
            self.optimizer.zero_grad()
            logger.debug(f"[TCN_SVR_Model.fit] Epoch {epoch+1} - Before TCN forward pass. X_train_tcn shape={X_train_tcn.shape}")
            with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda' and self.use_amp)):
                output = self.net(X_train_tcn.permute(0, 2, 1))
            logger.debug(f"[TCN_SVR_Model.fit] Epoch {epoch+1} - After TCN forward pass. Output shape={output.shape}")
            loss = self.criterion(output, y_train_tcn)
            logger.debug(f"[TCN_SVR_Model.fit] Epoch {epoch+1} - Loss calculated: {loss.item():.4f}")
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            logger.debug(f"[TCN_SVR_Model.fit] Epoch {epoch+1} - Optimizer step completed.")

            if can_validate:
                self.net.eval()
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda' and self.use_amp)):
                        val_output = self.net(X_val_tcn.permute(0, 2, 1))
                    val_loss = self.criterion(val_output, y_val_tcn)
                
                if (epoch + 1) % 10 == 0:
                    logger.info(f"[TCN_SVR_Model.fit] Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

                if val_loss.item() < best_loss - min_delta:
                    best_loss = val_loss.item()
                    patience_counter = 0
                    best_model_state = self.net.state_dict()
                    logger.info(f"[TCN_SVR_Model.fit] New best validation loss: {best_loss:.4f}")
                else:
                    patience_counter += 1
                    logger.info(f"[TCN_SVR_Model.fit] Validation loss not improved. Patience counter: {patience_counter}")

                if patience_counter >= patience:
                    logger.info(f"[TCN_SVR_Model.fit] Early stopping triggered after {epoch+1} epochs.")
                    break
            elif (epoch + 1) % 10 == 0:
                logger.info(f"[TCN_SVR_Model.fit] Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}")
        
        if can_validate and best_model_state:
            self.net.load_state_dict(best_model_state)
            logger.info("[TCN_SVR_Model.fit] Restored best model weights.")
        self.best_loss = best_loss if can_validate else loss.item()

        logger.info("[TCN_SVR_Model.fit] TCN training complete. Predicting indicators on training data.")
        self.net.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda' and self.use_amp)):
                predicted_indicators_train_tensor = self.net(X_train_tensor.permute(0, 2, 1))
        
        predicted_indicators_train_numpy = predicted_indicators_train_tensor.detach().cpu().numpy()
        logger.info(f"[TCN_SVR_Model.fit] Predicted indicators (train) shape={predicted_indicators_train_numpy.shape}, dtype={predicted_indicators_train_numpy.dtype}")

        logger.info("[TCN_SVR_Model.fit] Training SVR model.")
        self.svr_model.fit(predicted_indicators_train_numpy, y_train_returns_numpy)
        logger.info("[TCN_SVR_Model.fit] SVR training complete. Function exit.")

    def predict(self, X_test_tensor):
        logger.info("[TCN_SVR_Model.predict] Function entry.")
        logger.info(f"[TCN_SVR_Model.predict] Input: X_test_tensor shape={X_test_tensor.shape}, dtype={X_test_tensor.dtype}")
        
        self.net.eval()
        with torch.no_grad():
            if self.device.type == 'cuda' and self.use_amp:
                X_test_tensor = X_test_tensor.to(self.device).half()
            else:
                X_test_tensor = X_test_tensor.to(self.device)
            with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda' and self.use_amp)):
                predicted_indicators_test_tensor = self.net(X_test_tensor.permute(0, 2, 1))
            predicted_indicators_test_numpy = predicted_indicators_test_tensor.detach().cpu().numpy()
        logger.info(f"[TCN_SVR_Model.predict] Predicted indicators (test) shape={predicted_indicators_test_numpy.shape}, dtype={predicted_indicators_test_numpy.dtype}")

        predicted_return = self.svr_model.predict(predicted_indicators_test_numpy)
        logger.info(f"[TCN_SVR_Model.predict] Predicted return shape={predicted_return.shape}, dtype={predicted_return.dtype}")
        logger.info("[TCN_SVR_Model.predict] Function exit.")
        return predicted_return
