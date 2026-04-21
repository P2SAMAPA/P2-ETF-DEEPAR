"""
N-BEATS model implemented in PyTorch.
Interpretable architecture with trend and seasonality stacks.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import copy

class NBEATSBlock(nn.Module):
    def __init__(self, input_size, hidden_size, thetas_dim, backcast_length, forecast_length):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.thetas_dim = thetas_dim
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length

        self.fc1 = nn.Linear(backcast_length, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.theta_backcast = nn.Linear(hidden_size, thetas_dim)
        self.theta_forecast = nn.Linear(hidden_size, thetas_dim)

    def forward(self, x):
        # x shape: (batch, backcast_length)
        out = torch.relu(self.fc1(x))
        out = torch.relu(self.fc2(out))
        out = torch.relu(self.fc3(out))
        out = torch.relu(self.fc4(out))
        theta_b = self.theta_backcast(out)
        theta_f = self.theta_forecast(out)
        return theta_b, theta_f

class TrendStack(nn.Module):
    def __init__(self, n_blocks, hidden_size, backcast_length, forecast_length):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(NBEATSBlock(
                input_size=backcast_length,
                hidden_size=hidden_size,
                thetas_dim=4,  # 2 for polynomial (trend) + 2 for forecast
                backcast_length=backcast_length,
                forecast_length=forecast_length
            ))

    def forward(self, x):
        backcast = x
        forecast = torch.zeros(x.size(0), forecast_length).to(x.device)
        for block in self.blocks:
            theta_b, theta_f = block(backcast)
            # Trend basis: polynomial
            t = torch.linspace(0, 1, backcast_length).to(x.device)
            backcast_basis = torch.stack([t**i for i in range(4)], dim=1)  # (backcast_length, 4)
            backcast_block = theta_b @ backcast_basis.T  # (batch, backcast_length)

            t_f = torch.linspace(0, 1, forecast_length).to(x.device)
            forecast_basis = torch.stack([t_f**i for i in range(4)], dim=1)
            forecast_block = theta_f @ forecast_basis.T

            backcast = backcast - backcast_block
            forecast = forecast + forecast_block
        return backcast, forecast

class SeasonalityStack(nn.Module):
    def __init__(self, n_blocks, hidden_size, backcast_length, forecast_length):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(NBEATSBlock(
                input_size=backcast_length,
                hidden_size=hidden_size,
                thetas_dim=8,  # Fourier basis (4 harmonics * 2)
                backcast_length=backcast_length,
                forecast_length=forecast_length
            ))

    def forward(self, x):
        backcast = x
        forecast = torch.zeros(x.size(0), forecast_length).to(x.device)
        for block in self.blocks:
            theta_b, theta_f = block(backcast)
            # Fourier basis
            t = torch.linspace(0, 1, backcast_length).to(x.device)
            basis = []
            for k in range(1, 5):
                basis.append(torch.sin(2 * np.pi * k * t))
                basis.append(torch.cos(2 * np.pi * k * t))
            backcast_basis = torch.stack(basis, dim=1)
            backcast_block = theta_b @ backcast_basis.T

            t_f = torch.linspace(0, 1, forecast_length).to(x.device)
            basis_f = []
            for k in range(1, 5):
                basis_f.append(torch.sin(2 * np.pi * k * t_f))
                basis_f.append(torch.cos(2 * np.pi * k * t_f))
            forecast_basis = torch.stack(basis_f, dim=1)
            forecast_block = theta_f @ forecast_basis.T

            backcast = backcast - backcast_block
            forecast = forecast + forecast_block
        return backcast, forecast

class NBEATS(nn.Module):
    def __init__(self, backcast_length, forecast_length, stack_types, n_blocks_per_stack,
                 hidden_size, thetas_dim):
        super().__init__()
        self.stacks = nn.ModuleList()
        for stack_type in stack_types:
            if stack_type == "trend":
                self.stacks.append(TrendStack(n_blocks_per_stack, hidden_size,
                                              backcast_length, forecast_length))
            elif stack_type == "seasonality":
                self.stacks.append(SeasonalityStack(n_blocks_per_stack, hidden_size,
                                                    backcast_length, forecast_length))
            else:
                raise ValueError(f"Unknown stack type: {stack_type}")

    def forward(self, x):
        # x: (batch, backcast_length)
        forecast = torch.zeros(x.size(0), forecast_length).to(x.device)
        for stack in self.stacks:
            _, stack_forecast = stack(x)
            forecast = forecast + stack_forecast
        return forecast

class NBEATSTrainer:
    def __init__(self, backcast_length=126, forecast_length=22,
                 stack_types=["trend", "seasonality"], n_blocks_per_stack=3,
                 hidden_size=128, epochs=100, batch_size=32, lr=0.0005,
                 patience=10, seed=42):
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.stack_types = stack_types
        self.n_blocks_per_stack = n_blocks_per_stack
        self.hidden_size = hidden_size
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
        scaled = self.scaler.fit_transform(series.reshape(-1, 1)).flatten()

        X, y = [], []
        for i in range(len(scaled) - self.backcast_length - self.forecast_length + 1):
            X.append(scaled[i:i+self.backcast_length])
            y.append(scaled[i+self.backcast_length:i+self.backcast_length+self.forecast_length])
        if len(X) == 0:
            return False

        X = np.array(X)
        y = np.array(y)

        split = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                      torch.tensor(y_train, dtype=torch.float32))
        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                    torch.tensor(y_val, dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        self.model = NBEATS(
            backcast_length=self.backcast_length,
            forecast_length=self.forecast_length,
            stack_types=self.stack_types,
            n_blocks_per_stack=self.n_blocks_per_stack,
            hidden_size=self.hidden_size,
            thetas_dim=4  # Not directly used in stacks; each stack defines its own
        ).to(self.device)

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
                output = self.model(batch_X)
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
                    output = self.model(batch_X)
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

    def forecast(self, recent_series: np.ndarray) -> dict:
        self.model.eval()
        scaled = self.scaler.transform(recent_series.reshape(-1, 1)).flatten()
        input_seq = torch.tensor(scaled[-self.backcast_length:], dtype=torch.float32).view(1, -1).to(self.device)

        with torch.no_grad():
            output = self.model(input_seq)
            pred = output[0].cpu().numpy()
        pred = self.scaler.inverse_transform(pred.reshape(-1, 1)).flatten()

        forecasts = {}
        for h in [1, 5, 22]:
            if h <= len(pred):
                forecasts[h] = float(pred[h-1])
        return forecasts
