# Position Sizing

Control how much capital to allocate per trade.

## Sizing Methods

### Percent of Equity (Default)

Position size as percentage of current equity:

```python
results = mt.backtest(data, signal, size=0.10)  # 10% of equity
```

- Position grows as equity grows
- Default: 10%

### Fixed Dollar Amount

Always trade a fixed dollar amount:

```bash
# CLI
mantis run -d data.csv --sizing-method fixed --fixed-dollar 10000
```

- Position stays constant regardless of equity
- Good for consistent exposure

### Volatility-Targeted

Target a specific annualized volatility:

```bash
# CLI
mantis run -d data.csv --sizing-method volatility \
    --target-vol 0.15 \     # Target 15% annual volatility
    --vol-lookback 20       # 20-day lookback
```

- Low-vol assets get larger positions
- High-vol assets get smaller positions
- Maintains consistent portfolio risk

### Signal-Scaled

Scale position by signal strength:

```python
# Signal magnitude affects size
signal = model_confidence * np.sign(predictions)  # e.g., 0.8 * 1 = 0.8

results = mt.backtest(data, signal, size="signal", base_size=0.10)
# Signal of 0.8 → 8% position
# Signal of 0.5 → 5% position
```

### Risk-Based (ATR)

Risk a fixed percentage with ATR-based stop:

```bash
# CLI
mantis run -d data.csv --sizing-method risk \
    --risk-per-trade 0.01 \   # Risk 1% per trade
    --stop-atr 2.0 \          # 2 ATR stop loss
    --atr-period 14           # 14-day ATR
```

Formula: `position = (equity × risk%) / (ATR × multiplier)`

- Ensures consistent risk per trade
- Position size adapts to volatility

## Fluent API

```python
results = (
    mt.Backtest(data, signal)
    .size(0.15)                    # 15% of equity
    .run()
)
```

## Position Constraints

### Maximum Position Size

```python
# Never exceed 25% of equity
results = mt.backtest(
    data, signal,
    size=0.20,
    max_position=0.25
)
```

### Maximum Leverage

```python
# Allow up to 2x leverage
results = mt.backtest(
    data, signal,
    max_leverage=2.0
)
```

## CLI Options

```bash
mantis run -d data.csv \
    --sizing-method percent \     # percent|fixed|volatility|signal|risk
    --position-size 0.10 \        # Base size for percent method
    --fixed-dollar 10000 \        # For fixed method
    --target-vol 0.15 \           # For volatility method
    --vol-lookback 20 \           # Volatility lookback
    --risk-per-trade 0.01 \       # For risk method
    --stop-atr 2.0 \              # ATR stop multiplier
    --atr-period 14               # ATR period
```

## Comparison Table

| Method | When to Use |
|--------|-------------|
| **Percent of Equity** | Default, works for most cases |
| **Fixed Dollar** | Consistent exposure, starting out |
| **Volatility-Targeted** | Multi-asset, want consistent vol |
| **Signal-Scaled** | Model outputs confidence scores |
| **Risk-Based** | Professional risk management |

## Fractional Shares

By default, positions round down to whole shares:

```python
# Allow fractional shares
results = mt.backtest(
    data, signal,
    fractional=True
)
```

## Edge Cases

| Scenario | Behavior |
|----------|----------|
| Equity = 0 | Error |
| Position > equity | Apply max_position, warn |
| Volatility = 0 | Error |
| ATR = 0 | Use minimum (1 share), warn |
| Insufficient cash | Skip trade with warning |

## Example: Risk Parity Sizing

```python
import numpy as np

# Calculate volatility for each bar
volatility = data["close"].rolling(20).std() / data["close"]

# Target 15% portfolio volatility
target_vol = 0.15

# Inverse volatility sizing
signal_sized = signal * (target_vol / (volatility * np.sqrt(252)))

results = mt.backtest(data, signal_sized, size="signal")
```
