# How Mantis Works

Understanding the architecture and design decisions behind Mantis.

## Architecture

```
┌─────────────────────────────────────────────┐
│           Python API (mantis)               │
│    load() · backtest() · validate()         │
└─────────────────┬───────────────────────────┘
                  │ PyO3 bindings
┌─────────────────▼───────────────────────────┐
│           Rust Core (_mantis)               │
│   Data loading · Engine · Analytics         │
└─────────────────────────────────────────────┘
```

**Why Rust?**

- **Speed**: 10-100x faster than pure Python
- **Memory**: Efficient data handling for large datasets
- **Correctness**: Type safety prevents many bugs

**Why PyO3?**

- Seamless Python integration
- Zero-copy numpy array interop
- Feels like native Python

## The Backtest Loop

When you call `mt.backtest(data, signal)`:

```python
# 1. Signal at bar[i] close
signal[i] = 1  # Your model says "go long"

# 2. Order submitted at bar[i] close
order = Order(direction="long", signal=signal[i])

# 3. Order fills at bar[i+1] open
fill_price = data["open"][i+1] * (1 + slippage)

# 4. Position held until signal changes
# ...

# 5. Exit signal at bar[j] close
signal[j] = 0  # Your model says "go flat"

# 6. Exit fills at bar[j+1] open
exit_price = data["open"][j+1] * (1 - slippage)
```

This is the **realistic** execution model that prevents lookahead bias.

## Why "Next Bar Open"?

Consider a naive backtest:

```python
# WRONG: Lookahead bias
if close[i] > threshold:
    buy_at_price = close[i]  # Impossible in real trading!
```

In reality:

1. You see the close price
2. Market is closed
3. Next day, you submit order
4. Order fills at next open

Mantis enforces this automatically.

## Cost Model

Every trade incurs costs:

```python
# Buy order
buy_price = open[i+1]                    # Next bar open
slippage_cost = buy_price * 0.001        # 0.1% slippage
commission_cost = buy_price * 0.001      # 0.1% commission
actual_buy = buy_price + slippage_cost + commission_cost

# Sell order
sell_price = open[j+1]
slippage_cost = sell_price * 0.001
commission_cost = sell_price * 0.001
actual_sell = sell_price - slippage_cost - commission_cost
```

These costs compound:

- 1 trade per day × 252 days × 0.2% = 50% annual cost!
- This is why high-frequency strategies often fail in practice

## Position Sizing

Default: 10% of equity per trade.

```python
equity = 100_000
size = 0.10
position_value = equity * size  # $10,000

price = 150.00
shares = position_value / price  # 66 shares
```

As equity grows/shrinks, position sizes adjust proportionally.

## Short Selling

When signal is negative:

```python
signal[i] = -1  # Short signal

# Borrow shares and sell
sell_price = open[i+1] * (1 - slippage)

# While short, pay borrow costs
daily_cost = position_value * (borrow_rate / 252)

# Cover when signal changes
buy_price = open[j+1] * (1 + slippage)
```

Profit = sell_price - buy_price - costs

## Metric Calculation

### Sharpe Ratio

```python
# Example equity curve
equity = np.array([100_000, 101_500, 99_750, 102_300, 101_900])

# Daily returns
returns = np.diff(equity) / equity[:-1]

# Annualized Sharpe
sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
```

**Frequency-aware annualization:**

| Frequency | Factor |
|-----------|--------|
| Daily (stocks) | √252 |
| Daily (crypto) | √365 |
| Hourly (stocks) | √1638 |
| 5-min (stocks) | √19656 |

### Maximum Drawdown

```python
# Example equity curve
equity = np.array([100_000, 101_500, 99_750, 102_300, 101_900])

# Running maximum
peak = np.maximum.accumulate(equity)

# Drawdown at each point
drawdown = (equity - peak) / peak

# Worst drawdown
max_drawdown = drawdown.min()  # Negative value
```

## Walk-Forward Validation

Tests if strategy works on unseen data:

```
Full Data: [=====================================]

Fold 1:   [=====Train=====][OOS]
Fold 2:   [======Train======][OOS]
Fold 3:   [=======Train=======][OOS]
...
Fold 12:  [==================Train==================][OOS]

IS Sharpe:  Average Sharpe on training periods
OOS Sharpe: Average Sharpe on test periods
Efficiency: OOS / IS ratio
```

If OOS << IS, strategy is likely overfit.

## Data Flow

```
CSV/Parquet
    │
    ▼
┌─────────────────┐
│   Auto-detect   │ Columns, dates, frequency
│   Validate      │ No future dates, sane prices
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  numpy arrays   │ timestamp, open, high, low, close, volume
│  Bar objects    │ For detailed access
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Engine       │ Signal → Orders → Fills → Positions
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Analytics     │ Sharpe, drawdown, trades, warnings
└────────┬────────┘
         │
         ▼
  BacktestResult
```

## Thread Safety

Mantis uses Rust's thread-safe primitives for parallel operations:

- Parameter sweeps use all CPU cores
- Walk-forward folds can run in parallel
- No GIL contention

## Memory Efficiency

- Data loaded once, shared across operations
- Zero-copy numpy interop via Arrow
- Streaming computation where possible
- 10GB Parquet loads in < 5 seconds
