# Execution Model

How Mantis simulates realistic trade execution.

## The Fundamental Rule

> **Signal at bar[i] close → Fill at bar[i+1] open**

This is the most important concept in Mantis. It prevents lookahead bias.

## Why This Matters

Consider this scenario:

```
Bar 100: Close = $150.00
Bar 101: Open = $148.00, Close = $152.00
```

**Naive (Wrong) Approach:**
```python
if close[100] > threshold:
    buy_at = close[100]  # $150.00
```

But in reality:
- You see $150.00 close
- Market closes
- Next day opens at $148.00
- Your order fills at $148.00 (or worse with slippage)

**Mantis Approach:**
```python
signal[100] = 1  # Signal generated at close
fill_price = open[101] * (1 + slippage)  # $148.15
```

## Order Types

### Market Orders (Default)

```python
# Buy signal
buy_price = next_bar.open * (1 + slippage)

# Sell signal
sell_price = next_bar.open * (1 - slippage)
```

Market orders always fill at next bar open with slippage.

### Limit Orders (CLI)

```bash
mantis run -d data.csv --execution-price limit
```

Limit orders only fill if price touches the limit:

```python
# Buy limit at $148.00
if bar.low <= 148.00:
    fill_price = min(limit_price, bar.open)
else:
    # Order carries to next bar or expires
```

## Slippage Models

### Fixed Percentage (Default)

```python
slippage = 0.001  # 0.1%

buy_price = open * (1 + slippage)   # Pay more
sell_price = open * (1 - slippage)  # Receive less
```

### Square-Root Market Impact

For large orders:

```python
slippage = k * sqrt(trade_size / avg_volume)
```

Larger orders have more market impact.

## Volume Participation

Limit trades to a percentage of bar volume:

```bash
mantis run -d data.csv --fill-probability 0.10  # Max 10% of volume
```

If order size exceeds limit:
- Partial fill this bar
- Remainder carries to next bar

## Fill Probability

Model random non-fills:

```bash
mantis run -d data.csv --fill-probability 0.95  # 95% fill rate
```

## Order Lifecycle

```
Signal Generated (bar[i] close)
    │
    ▼
Order Buffered
    │
    ▼
Next Bar Opens (bar[i+1])
    │
    ▼
Fill Check:
├── Market order → Fill at open + slippage
├── Limit order → Fill if price touched
└── Zero volume → Carry to next bar
    │
    ▼
Position Updated
    │
    ▼
Costs Applied:
├── Commission
├── Slippage
└── Borrow cost (if short)
```

## Stop-Loss and Take-Profit

Evaluated at bar close, fill at next bar open:

```python
# Position entered at $100
# Stop-loss at 2%

if current_price <= 98.00:  # Evaluated at close
    exit_signal = True
    # Exit order fills at next_bar.open
```

**Why not intrabar stops?**

With daily data, we don't know intrabar price path:

```
Open: $100, High: $102, Low: $95, Close: $99

Did price go:
- 100 → 102 → 95 → 99 (stop triggered at 95)
- 100 → 95 → 102 → 99 (stop triggered at 95)
- 100 → 99 → 95 → 102 (stop triggered at 95)
```

We can't know. So Mantis evaluates at close only.

## Gap Handling

When price gaps through limits:

```python
# Limit buy at $148.00
# Bar opens at $145.00 (gap down)

fill_price = open  # $145.00, better than limit
```

```python
# Stop-loss at $98.00
# Bar opens at $95.00 (gap down)

fill_price = open  # $95.00, worse than stop
```

Gaps are filled at open price (realistic).

## Zero Volume Bars

When a bar has zero volume:

```python
if bar.volume == 0:
    # Order carries to next bar with warning
    carry_order()
    log.warning("Zero volume, order carried")
```

## Short Selling

```python
signal = -1  # Short signal

# Borrow shares
shares_borrowed = position_size / price

# Sell borrowed shares
sell_price = next_bar.open * (1 - slippage)

# Daily borrow cost accrual
while position_open:
    daily_cost = position_value * (borrow_rate / 252)
    equity -= daily_cost

# Cover (buy back)
buy_price = exit_bar.open * (1 + slippage)
```

## Realistic Defaults

```python
commission = 0.001   # 0.1% per trade
slippage = 0.001     # 0.1% per trade
borrow_cost = 0.03   # 3% annual for shorts
```

These are conservative but realistic for retail trading.

## Edge Cases

| Scenario | Behavior |
|----------|----------|
| Signal on last bar | No trade (no next bar to fill) |
| Gap through stop | Fill at open (worse price) |
| Zero volume | Order carries with warning |
| Slippage > 10% | Cap at 10%, warn |
| Negative price | Error on data load |
