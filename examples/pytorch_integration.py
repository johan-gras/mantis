#!/usr/bin/env python3
"""
PyTorch Integration Example for Mantis Backtest Engine

This example demonstrates how to:
1. Export features from the Rust engine for ML training
2. Train a simple LSTM model in PyTorch
3. Generate predictions and feed them back to the Rust engine for backtesting

Requirements:
    pip install torch numpy pandas

Workflow:
    1. Use `mantis features --csv` to export features
    2. Train your model in Python
    3. Export predictions to CSV
    4. Use `mantis backtest --signals predictions.csv` to run the backtest
"""

import subprocess
import json
import sys
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install torch numpy pandas")
    sys.exit(1)


# ============================================================================
# Step 1: Generate Features using Mantis CLI
# ============================================================================

def export_features(data_path: str, output_path: str = "features.csv"):
    """Export features from Mantis backtest engine."""
    cmd = [
        "cargo", "run", "--release", "--",
        "features",
        "--data", data_path,
        "--output", output_path,
        "--target-horizon", "5",  # 5-day forward returns as target
        "--config", "comprehensive",
    ]

    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        return None

    return output_path


# ============================================================================
# Step 2: PyTorch LSTM Model for Time Series Prediction
# ============================================================================

class LSTMPredictor(nn.Module):
    """LSTM model for predicting future returns."""

    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Take the last time step's output
        last_out = lstm_out[:, -1, :]
        return self.fc(last_out)


class TransformerPredictor(nn.Module):
    """Transformer model for time series prediction."""

    def __init__(self, input_size: int, d_model: int = 64,
                 nhead: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()

        self.input_projection = nn.Linear(input_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x = self.input_projection(x)
        x = self.transformer(x)
        # Take the last time step
        return self.fc(x[:, -1, :])


# ============================================================================
# Step 3: Data Preparation
# ============================================================================

def prepare_sequences(features: np.ndarray, targets: np.ndarray,
                      seq_len: int = 20, train_ratio: float = 0.7):
    """
    Prepare sequences for LSTM training.

    Args:
        features: Feature matrix (n_samples, n_features)
        targets: Target values (n_samples,)
        seq_len: Sequence length for LSTM
        train_ratio: Ratio of data for training

    Returns:
        Train and validation datasets
    """
    n_samples = len(features) - seq_len

    X = np.zeros((n_samples, seq_len, features.shape[1]))
    y = np.zeros(n_samples)

    for i in range(n_samples):
        X[i] = features[i:i+seq_len]
        y[i] = targets[i+seq_len]

    # Remove samples with NaN targets
    valid_mask = ~np.isnan(y)
    X = X[valid_mask]
    y = y[valid_mask]

    # Normalize features
    mean = X.mean(axis=(0, 1), keepdims=True)
    std = X.std(axis=(0, 1), keepdims=True) + 1e-8
    X = (X - mean) / std

    # Train/val split (time series aware - no shuffling)
    split_idx = int(len(X) * train_ratio)

    X_train = torch.FloatTensor(X[:split_idx])
    y_train = torch.FloatTensor(y[:split_idx])
    X_val = torch.FloatTensor(X[split_idx:])
    y_val = torch.FloatTensor(y[split_idx:])

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    return train_dataset, val_dataset, (mean, std)


# ============================================================================
# Step 4: Training Loop
# ============================================================================

def train_model(model: nn.Module, train_dataset: TensorDataset,
                val_dataset: TensorDataset, epochs: int = 50,
                batch_size: int = 32, lr: float = 0.001):
    """Train the model."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    model = model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(X_batch).squeeze()
            loss = criterion(predictions, y_batch)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                predictions = model(X_batch).squeeze()
                val_loss += criterion(predictions, y_batch).item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    # Restore best model
    model.load_state_dict(best_model_state)
    return model


# ============================================================================
# Step 5: Generate Predictions for Backtesting
# ============================================================================

def generate_predictions(model: nn.Module, features: np.ndarray,
                         timestamps: np.ndarray, seq_len: int,
                         normalization: tuple) -> pd.DataFrame:
    """Generate predictions for all data points."""

    device = next(model.parameters()).device
    mean, std = normalization

    model.eval()
    predictions = []

    with torch.no_grad():
        for i in range(seq_len, len(features)):
            seq = features[i-seq_len:i]
            seq_norm = (seq - mean[0, 0, :]) / std[0, 0, :]
            seq_tensor = torch.FloatTensor(seq_norm).unsqueeze(0).to(device)
            pred = model(seq_tensor).item()
            predictions.append({
                'timestamp': timestamps[i],
                'signal': pred,
            })

    return pd.DataFrame(predictions)


def export_predictions(predictions: pd.DataFrame, output_path: str = "predictions.csv"):
    """Export predictions for Mantis backtest engine."""
    predictions.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")
    return output_path


# ============================================================================
# Step 6: Run Backtest with Predictions
# ============================================================================

def run_backtest(data_path: str, signals_path: str,
                 threshold: float = 0.5, output: str = "results.json"):
    """Run backtest using the Mantis engine with ML predictions."""

    cmd = [
        "cargo", "run", "--release", "--",
        "backtest",
        "--data", data_path,
        "--strategy", "external-signal",
        "--signals", signals_path,
        "--threshold", str(threshold),
        "--output", output,
    ]

    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running backtest: {e.stderr}")
        return None

    # Load and return results
    with open(output) as f:
        return json.load(f)


# ============================================================================
# Example Usage - Simulated Data
# ============================================================================

def create_synthetic_data(n_samples: int = 1000):
    """Create synthetic data for demonstration."""

    np.random.seed(42)

    # Generate timestamps
    timestamps = pd.date_range('2020-01-01', periods=n_samples, freq='D')

    # Generate price with trend and mean reversion
    price = 100.0
    prices = [price]
    for i in range(1, n_samples):
        # Mean-reverting with trend
        trend = 0.0001
        mean_rev = 0.01 * (100 - prices[-1]) / 100
        noise = np.random.randn() * 0.02
        ret = trend + mean_rev + noise
        price = prices[-1] * (1 + ret)
        prices.append(price)

    prices = np.array(prices)

    # Generate features
    features = {}

    # Returns at different horizons
    for lag in [1, 5, 10, 20]:
        features[f'return_{lag}'] = np.concatenate([
            np.zeros(lag),
            (prices[lag:] - prices[:-lag]) / prices[:-lag]
        ])

    # Moving averages
    for period in [5, 10, 20, 50]:
        ma = pd.Series(prices).rolling(period).mean().values
        features[f'sma_{period}'] = (prices - ma) / ma

    # Volatility
    returns = np.diff(np.log(prices))
    returns = np.concatenate([[0], returns])
    for period in [5, 10, 20]:
        vol = pd.Series(returns).rolling(period).std().values
        features[f'volatility_{period}'] = vol

    # RSI approximation
    delta = np.diff(prices)
    delta = np.concatenate([[0], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(14).mean().values
    avg_loss = pd.Series(loss).rolling(14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    features['rsi'] = 100 - 100 / (1 + rs)

    # Create DataFrame
    df = pd.DataFrame(features)
    df['timestamp'] = timestamps
    df['close'] = prices

    # Target: 5-day forward return
    df['target'] = df['return_1'].shift(-5)

    return df


def main():
    """Main example workflow."""

    print("=" * 60)
    print("PyTorch Integration Example for Mantis Backtest")
    print("=" * 60)

    # Create synthetic data for demonstration
    print("\n1. Creating synthetic data...")
    df = create_synthetic_data(1000)

    # Prepare feature columns (exclude timestamp, close, target)
    feature_cols = [c for c in df.columns if c not in ['timestamp', 'close', 'target']]

    features = df[feature_cols].values
    targets = df['target'].values
    timestamps = df['timestamp'].values

    # Remove NaN rows
    valid_mask = ~np.isnan(features).any(axis=1)
    features = features[valid_mask]
    targets = targets[valid_mask]
    timestamps = timestamps[valid_mask]

    print(f"   Features shape: {features.shape}")
    print(f"   Feature columns: {feature_cols[:5]}...")

    # Prepare sequences
    print("\n2. Preparing sequences for LSTM...")
    seq_len = 20
    train_dataset, val_dataset, normalization = prepare_sequences(
        features, targets, seq_len=seq_len, train_ratio=0.7
    )
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")

    # Create and train model
    print("\n3. Training LSTM model...")
    input_size = features.shape[1]
    model = LSTMPredictor(input_size=input_size, hidden_size=64, num_layers=2)

    model = train_model(
        model,
        train_dataset,
        val_dataset,
        epochs=30,
        batch_size=32,
        lr=0.001
    )

    # Generate predictions
    print("\n4. Generating predictions...")
    predictions_df = generate_predictions(
        model, features, timestamps, seq_len, normalization
    )
    print(f"   Generated {len(predictions_df)} predictions")
    print(f"   Signal range: [{predictions_df['signal'].min():.4f}, {predictions_df['signal'].max():.4f}]")

    # Export predictions
    print("\n5. Exporting predictions...")
    export_predictions(predictions_df, "predictions.csv")

    # Calculate simple metrics
    print("\n6. Evaluating signal quality...")
    actual_returns = targets[seq_len:][:len(predictions_df)]
    predicted_signals = predictions_df['signal'].values

    # Correlation
    valid_idx = ~np.isnan(actual_returns)
    correlation = np.corrcoef(actual_returns[valid_idx], predicted_signals[valid_idx])[0, 1]
    print(f"   Signal-Return Correlation: {correlation:.4f}")

    # Directional accuracy
    actual_direction = (actual_returns[valid_idx] > 0).astype(int)
    predicted_direction = (predicted_signals[valid_idx] > 0).astype(int)
    directional_accuracy = (actual_direction == predicted_direction).mean()
    print(f"   Directional Accuracy: {directional_accuracy:.2%}")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)
    print("\nTo run a full backtest with the Mantis engine:")
    print("  cargo run --release -- backtest --data your_data.csv \\")
    print("      --strategy external-signal --signals predictions.csv \\")
    print("      --threshold 0.001")


if __name__ == "__main__":
    main()
