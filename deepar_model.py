"""
DeepAR model implemented in PyTorch.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import copy

class DeepAR(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, x, hidden=None):
        # x shape: (batch, seq_len, input_size)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)  # (batch, seq_len, output_size)
        return out, hidden

class DeepARTrainer:
    def __init__(self, context_len=60, pred_len=22, hidden_size=32, num_layers=2,
                 epochs=50, batch_size=64, lr=0.001, patience=5, seed=42):
        self.context_len = context_len
        self.pred_len = pred_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.seed = seed

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, series: np.ndarray):
        """Train DeepAR on a univariate time series."""
        # Scale data
        scaled = self.scaler.fit_transform(series.reshape(-1, 1)).flatten()

        # Create sequences
        X, y = [], []
        for i in range(len(scaled) - self.context_len - self.pred_len + 1):
            X.append(scaled[i:i+self.context_len])
            y.append(scaled[i+self.context_len:i+self.context_len+self.pred_len])
        if len(X) == 0:
            return False

        X = np.array(X).reshape(-1, self.context_len, 1)
        y = np.array(y).reshape(-1, self.pred_len, 1)

        # Train/validation split
        split = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                      torch.tensor(y_train, dtype=torch.float32))
        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                    torch.tensor(y_val, dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        self.model = DeepAR(input_size=1, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, output_size=1).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        best_loss = float('inf')
        best_state = None
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                output, _ = self.model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch_X.size(0)
            train_loss /= len(train_loader.dataset)

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    output, _ = self.model(batch_X)
                    loss = criterion(output, batch_y)
                    val_loss += loss.item() * batch_X.size(0)
            val_loss /= len(val_loader.dataset)

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"    Early stopping at epoch {epoch+1}")
                    break

        if best_state:
            self.model.load_state_dict(best_state)
        return True

    def forecast(self, recent_series: np.ndarray, num_samples=100) -> dict:
        """Generate probabilistic forecasts for multiple horizons."""
        self.model.eval()
        scaled = self.scaler.transform(recent_series.reshape(-1, 1)).flatten()
        input_seq = torch.tensor(scaled[-self.context_len:], dtype=torch.float32).view(1, -1, 1).to(self.device)

        # Monte Carlo dropout or just single deterministic forecast
        with torch.no_grad():
            output, _ = self.model(input_seq)
            pred = output[0, :self.pred_len, 0].cpu().numpy()
        pred = self.scaler.inverse_transform(pred.reshape(-1, 1)).flatten()

        forecasts = {}
        for h in [1, 5, 22]:
            if h <= len(pred):
                forecasts[h] = float(pred[h-1])
        return forecasts
