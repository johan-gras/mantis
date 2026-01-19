# Multi-Symbol Backtesting

Backtest across multiple assets with automatic portfolio math.

## Load Multiple Symbols

```python
import mantis as mt
import numpy as np

# From paths
data = mt.load_multi({
    "AAPL": "data/samples/AAPL.csv",
    "SPY": "data/samples/SPY.csv",
    "BTC": "data/samples/BTC.csv"
})

# From directory
data = mt.load_dir("data/samples/*.csv")
```

## Multi-Symbol Backtests

Run a separate backtest per symbol:

```python
signals = {
    "AAPL": np.ones(len(data["AAPL"]["close"])),
    "SPY": np.ones(len(data["SPY"]["close"])),
    "BTC": np.ones(len(data["BTC"]["close"]))
}

results = {
    symbol: mt.backtest(symbol_data, signals[symbol])
    for symbol, symbol_data in data.items()
}
```

## Portfolio Strategies (CLI)

Use the CLI for portfolio-level strategies:

```bash
# Equal weight across all symbols
mantis portfolio -d ./data/stocks/ -p "*.csv" --strategy equal-weight

# Risk parity (equal risk contribution)
mantis portfolio -d ./data/stocks/ -p "*.csv" --strategy risk-parity

# Momentum (overweight recent winners)
mantis portfolio -d ./data/stocks/ -p "*.csv" --strategy momentum

# Minimum variance
mantis portfolio -d ./data/stocks/ -p "*.csv" --strategy min-variance

# Maximum Sharpe
mantis portfolio -d ./data/stocks/ -p "*.csv" --strategy max-sharpe
```

## Available Portfolio Strategies

| Strategy | Description |
|----------|-------------|
| `equal-weight` | Equal allocation across all assets |
| `momentum` | Overweight recent winners |
| `inverse-vol` | Inverse volatility weighting |
| `risk-parity` | Equal risk contribution |
| `min-variance` | Minimum variance portfolio |
| `max-sharpe` | Maximum Sharpe ratio portfolio |
| `hrp` | Hierarchical Risk Parity |
| `drift-equal` | Equal weight with drift rebalancing |
| `drift-momentum` | Momentum with drift rebalancing |

## Portfolio Constraints

```bash
mantis portfolio -d ./data/ -p "*.csv" --strategy risk-parity \
    --max-position 0.20 \      # Max 20% per asset
    --max-leverage 1.0 \       # No leverage
    --max-turnover 0.50 \      # Max 50% turnover per rebalance
    --min-holdings 5 \         # Hold at least 5 assets
    --max-holdings 15          # Hold at most 15 assets
```

## Rebalancing

```bash
# Rebalance every 21 trading days
mantis portfolio -d ./data/ -p "*.csv" --strategy equal-weight \
    --rebalance-freq 21

# Monthly rebalancing with lookback
mantis portfolio -d ./data/ -p "*.csv" --strategy momentum \
    --rebalance-freq 21 \
    --lookback 63  # 3-month momentum
```

## Output Formats

```bash
# Text (default)
mantis portfolio -d ./data/ -p "*.csv" --strategy risk-parity

# JSON
mantis portfolio -d ./data/ -p "*.csv" --strategy risk-parity -o json

# CSV
mantis portfolio -d ./data/ -p "*.csv" --strategy risk-parity -o csv
```

## Per-Symbol Metrics

Inspect metrics per symbol:

```python
for symbol, result in results.items():
    print(f"{symbol}: Sharpe {result.sharpe:.2f}, Return {result.total_return:.1%}")
```

## Correlation and Diversification

Portfolio strategies automatically handle:

- Correlation matrix estimation
- Diversification benefits
- Rebalancing costs
- Position constraints

## Example: Momentum Portfolio

```bash
mantis portfolio -d ./data/stocks/ -p "*.csv" \
    --strategy momentum \
    --rebalance-freq 21 \        # Monthly rebalance
    --lookback 252 \             # 1-year momentum
    --max-position 0.10 \        # Max 10% per stock
    --max-holdings 10            # Top 10 momentum stocks
```

## Example: Risk Parity

```bash
mantis portfolio -d ./data/etfs/ -p "*.csv" \
    --strategy risk-parity \
    --rebalance-freq 63 \        # Quarterly rebalance
    --max-leverage 1.5           # Allow 50% leverage
```
