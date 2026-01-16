# Cost Model

Understanding trading costs and their impact on strategy performance.

## Why Costs Matter

A strategy with 20% annual return sounds great. But after costs:

```
Gross return:        +20.0%
Commission (0.1%):   -12.6%  (1 trade/week × 52 × 0.2%)
Slippage (0.1%):     -12.6%
Net return:          -5.2%
```

Many backtests look profitable until realistic costs are applied.

## Cost Components

### Commission

Fee paid to broker per trade.

```python
commission = 0.001  # 0.1% = $1 per $1000 traded

# Applied to trade value
trade_value = shares * price
commission_cost = trade_value * commission
```

**Common commission structures:**

| Type | Example | Mantis Setting |
|------|---------|---------------|
| Percentage | 0.1% | `commission=0.001` |
| Fixed | $1/trade | `commission_type="fixed"` |
| Per share | $0.005/share | `commission_type="per_share"` |

### Slippage

Price movement between decision and execution.

```python
slippage = 0.001  # 0.1%

# Buy order
actual_buy = price * (1 + slippage)  # Pay more

# Sell order
actual_sell = price * (1 - slippage)  # Receive less
```

**Why slippage occurs:**

1. **Bid-ask spread**: Buy at ask, sell at bid
2. **Market impact**: Your order moves the market
3. **Delay**: Price moves while order executes

### Borrow Cost (Shorts)

Interest paid to borrow shares for shorting.

```python
borrow_rate = 0.03  # 3% annual

# Daily cost accrual
daily_cost = position_value * (borrow_rate / 252)
```

**Why it matters for shorts:**

- Long positions: Hold indefinitely, no carrying cost
- Short positions: Pay borrow fee every day
- Some stocks (hard-to-borrow): 10-50%+ annual rate

## Default Values

```python
mt.backtest(
    data, signal,
    commission=0.001,    # 0.1%
    slippage=0.001,      # 0.1%
    borrow_cost=0.03,    # 3% annual
)
```

These are conservative but realistic for retail trading.

## Cost Impact Analysis

### Trade Frequency Impact

```
Trades/year | Round-trip cost (0.2%) | Annual drag
-----------+------------------------+------------
12         | 0.2% × 12 = 2.4%      | -2.4%
52         | 0.2% × 52 = 10.4%     | -10.4%
252        | 0.2% × 252 = 50.4%    | -50.4%
```

Higher frequency = higher costs = harder to be profitable.

### Position Size Impact

Smaller positions have higher relative costs:

```
Position   | Trade value | Commission | % of profit needed
$1,000     | $1,000      | $1         | 0.2%
$10,000    | $10,000     | $10        | 0.2%
```

Percentage-based costs scale proportionally.

### Holding Period Impact

```
Holding period | Trades/year | Cost drag
--------------+-------------+-----------
1 day          | 252         | 50%+
1 week         | 52          | ~10%
1 month        | 12          | ~2%
1 year         | 1           | 0.2%
```

Longer holding = fewer trades = lower costs.

## Cost Sensitivity Testing

Test how robust your strategy is to higher costs:

```bash
# CLI
mantis cost-sensitivity -d data.csv --strategy sma-crossover

# Tests at 1x, 2x, 5x, 10x baseline costs
```

**Interpreting results:**

| Sharpe at 2x costs | Interpretation |
|--------------------|----------------|
| > 80% of baseline | Robust to costs |
| 50-80% | Moderate sensitivity |
| < 50% | Fragile, costs critical |

## Reducing Costs

### Trade less frequently

```python
# Instead of daily signals
signal = daily_model_signal

# Use weekly signals
signal = daily_model_signal[::5]  # Every 5 days
```

### Larger position sizes

Fixed costs amortize better over larger positions.

### Limit orders

```bash
mantis run -d data.csv --execution-price limit
```

Can reduce slippage but risk non-fills.

### Lower-cost assets

- ETFs: Lower spread than individual stocks
- Futures: Often lower percentage costs
- Forex: Tight spreads on majors

## Realistic Cost Ranges

| Asset Class | Commission | Slippage | Borrow |
|-------------|-----------|----------|--------|
| US Stocks (retail) | 0.05-0.1% | 0.1-0.2% | 2-10% |
| US Stocks (institution) | 0.01% | 0.05% | 0.5-3% |
| Crypto | 0.1-0.5% | 0.2-1% | N/A |
| Forex | 0.01% | 0.02% | N/A |
| Futures | $2-5/contract | 1 tick | N/A |

## Zero-Cost Analysis

Sometimes useful to understand theoretical performance:

```bash
mantis cost-sensitivity -d data.csv --include-zero-cost
```

But never trust zero-cost results for real trading.

## Common Mistakes

### 1. Ignoring costs entirely

```python
# WRONG
results = mt.backtest(data, signal, commission=0, slippage=0)
```

### 2. Underestimating slippage

Market impact is often larger than expected, especially for:
- Small-cap stocks
- Large positions
- Fast-moving markets

### 3. Forgetting borrow costs

A short strategy that looks great long-only often fails when borrow costs are included.

### 4. Not testing cost sensitivity

If Sharpe drops 80% at 2x costs, your strategy edge is fragile.
