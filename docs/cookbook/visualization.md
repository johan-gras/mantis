# Visualization

Create charts and reports from your backtest results.

## Basic Plot

```python
import mantis as mt

data = mt.load_sample("AAPL")
results = mt.backtest(data, signal)

# Plot equity curve
results.plot()
```

**Environment detection:**

- **Jupyter**: Interactive Plotly chart
- **Terminal**: ASCII sparkline

## Plot Options

```python
# Show drawdown
results.plot(drawdown=True)

# Customize width (terminal)
results.plot(width=60)

# Hide drawdown in Plotly
results.plot(show_drawdown=False)
```

## Save to File

```python
# Interactive HTML
results.plot(save="results.html")

# Self-contained HTML report
results.report("experiment_001.html")
```

## HTML Report

Generate a comprehensive report:

```python
results.report("backtest_report.html")
```

Report includes:

- Summary metrics (color-coded)
- Performance grid
- Trade statistics
- Equity curve chart (SVG)
- Drawdown chart
- Trade list table
- Dark/light theme support

## Validation Plot

```python
validation = results.validate()

# In Jupyter: Plotly bar chart (IS vs OOS per fold)
# In terminal: ASCII chart
validation.plot()

# Generate HTML report
validation.report("validation_report.html")
```

## Compare Strategies

```python
results_a = mt.backtest(data, signal_a)
results_b = mt.backtest(data, signal_b)
results_c = mt.backtest(data, signal_c)

comparison = mt.compare(
    [results_a, results_b, results_c],
    names=["LSTM", "Transformer", "Baseline"]
)
print(comparison)
```

Output:
```
┌──────────────┬────────┬──────────┬───────────┐
│ Strategy     │ Sharpe │ Return   │ MaxDD     │
├──────────────┼────────┼──────────┼───────────┤
│ LSTM         │ 1.24   │ 18.5%    │ -12.3%    │
│ Transformer  │ 0.98   │ 14.2%    │ -15.7%    │
│ Baseline     │ 0.45   │ 8.1%     │ -22.4%    │
└──────────────┴────────┴──────────┴───────────┘
```

## ASCII Sparklines

Terminal-friendly visualization:

```python
print(results)
# Shows summary with sparkline:
# ▁▂▃▄▅▆▅▇█▇▆▅▄▃▂▁▂▃▄▅▆▇█▇▅▄▃▂▁
```

## Jupyter Rich Display

In Jupyter notebooks, results auto-display:

```python
results  # Just evaluate, no print() needed
```

Shows formatted metrics table with equity sparkline.

## CLI Visualization

### Sensitivity Heatmap

```bash
# Generate SVG heatmap
mantis sensitivity -d data.csv --strategy sma-crossover --heatmap sensitivity.svg
```

### Walk-Forward Chart

```bash
# View walk-forward results
mantis walk-forward -d data.csv --folds 12 --strategy sma-crossover
# Shows ASCII chart of fold performance
```

## Export Equity Curve

```python
# Get equity as numpy array
equity = results.equity_curve

# Plot with matplotlib
import matplotlib.pyplot as plt
plt.plot(equity)
plt.title("Equity Curve")
plt.xlabel("Bar")
plt.ylabel("Equity")
plt.savefig("equity.png")
```

## Export Metrics

```python
# Save to JSON
results.save("results.json")

# Get as dictionary
metrics = results.metrics()
print(metrics["sharpe"])
print(metrics["max_drawdown"])
```

## Plotly Customization

When in Jupyter with Plotly installed:

```python
# Get the Plotly figure object
fig = results.plot()

# Customize further
fig.update_layout(
    title="My Strategy",
    template="plotly_dark"
)
fig.show()
```

## Dependencies

For interactive charts in Jupyter:

```bash
pip install mantis-bt[jupyter]  # Includes plotly
```

ASCII visualization works without any extra dependencies.
