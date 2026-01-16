# Quick Start

Get from install to validated backtest in 5 minutes.

## Install

```bash
pip install mantis-bt
```

No Rust toolchain needed. Pre-built wheels for Linux, macOS, and Windows.

## Your First Backtest

```python
import mantis as mt
import numpy as np

# 1. Load data
data = mt.load_sample("AAPL")  # 10 years of daily OHLCV, bundled

# 2. Create a signal
#    1 = long, -1 = short, 0 = flat
np.random.seed(42)
signal = np.random.choice([-1, 0, 1], size=len(data))

# 3. Run backtest
results = mt.backtest(data, signal)

# 4. See results
print(results)
```

**Output:**
```
┌────────────────────────────────────────────────────┐
│ Total Return: 12.4%    Sharpe: 0.42                │
│ Max Drawdown: -18.3%   Win Rate: 48.2%             │
│ Trades: 847            Profit Factor: 1.12        │
│ ▁▂▃▄▅▆▅▇█▇▆▅▄▃▂▁▂▃▄▅▆▇█▇▅▄▃▂▁                     │
└────────────────────────────────────────────────────┘
```

## Validate Your Results

Random signals shouldn't produce consistent profits. Let's check:

```python
# Walk-forward validation (12 folds by default)
validation = results.validate()

print(f"In-sample Sharpe:  {validation.is_sharpe:.2f}")
print(f"Out-of-sample:     {validation.oos_sharpe:.2f}")
print(f"OOS/IS ratio:      {validation.efficiency:.0%}")
print(f"Verdict:           {validation.verdict}")
```

**Output:**
```
In-sample Sharpe:  0.45
Out-of-sample:     0.38
OOS/IS ratio:      84%
Verdict:           borderline
```

!!! info "What the verdict means"
    - **robust**: OOS/IS > 80%, consistent across folds
    - **borderline**: OOS/IS 60-80%, some degradation
    - **likely_overfit**: OOS/IS < 60%, significant degradation

## Visualize

```python
# In Jupyter: interactive Plotly chart
# In terminal: ASCII sparkline
results.plot()

# Export to file
results.report("my_backtest.html")
```

## Use Your Own Data

```python
# CSV file
data = mt.load("my_data.csv")

# Parquet file
data = mt.load("my_data.parquet")

# pandas DataFrame
import pandas as pd
df = pd.read_csv("my_data.csv")
results = mt.backtest(df, signal)

# polars DataFrame
import polars as pl
df = pl.read_csv("my_data.csv")
results = mt.backtest(df, signal)
```

Mantis auto-detects OHLCV columns:

- **Open**: `open`, `Open`, `OPEN`, `o`
- **High**: `high`, `High`, `HIGH`, `h`
- **Low**: `low`, `Low`, `LOW`, `l`
- **Close**: `close`, `Close`, `CLOSE`, `c`
- **Volume**: `volume`, `Volume`, `VOLUME`, `vol`, `v`

## Use Your Model's Predictions

```python
# Your model outputs predictions
predictions = model.predict(features)  # e.g., [0.9, 0.3, 0.7, 0.1]

# Convert to signals
# Option 1: Binary threshold
signal = np.where(predictions > 0.5, 1, -1)

# Option 2: Three-state with dead zone (recommended)
signal = np.where(predictions > 0.6, 1,
         np.where(predictions < 0.4, -1, 0))

# Option 3: Continuous for position sizing
signal = (predictions - 0.5) * 2  # Maps [0,1] to [-1,1]

# Run backtest
results = mt.backtest(data, signal)
```

## Customize Costs

```python
results = mt.backtest(
    data,
    signal,
    commission=0.001,   # 0.1% per trade (default)
    slippage=0.001,     # 0.1% per trade (default)
    size=0.10,          # 10% of equity per trade (default)
    cash=100_000,       # Starting capital
    allow_short=True,   # Allow short positions
)
```

## Fluent API

For more complex configurations:

```python
results = (
    mt.Backtest(data, signal)
    .commission(0.0005)      # 0.05%
    .slippage(0.0005)        # 0.05%
    .size(0.15)              # 15% per trade
    .cash(50_000)            # Starting capital
    .stop_loss(0.02)         # 2% stop loss
    .take_profit(0.05)       # 5% take profit
    .run()
)
```

## Important Disclaimer

!!! warning "Risk Disclosure"
    **Past performance does not guarantee future results.** Backtest results are hypothetical and do not represent actual trading. Real trading involves risk of loss. All performance metrics shown are based on historical simulations with assumptions that may not reflect actual market conditions.

## What's Next?

- **[Cookbook](cookbook/index.md)** - Common tasks with copy-paste code
- **[Concepts](concepts/how-it-works.md)** - Understand the execution model
- **[API Reference](api/index.md)** - Full function documentation
- **[Playground](playground.md)** - Try in Google Colab
