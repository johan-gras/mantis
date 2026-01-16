# Playground

Try Mantis without installing anything.

## Google Colab

Run Mantis directly in your browser with Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/johan-gras/mantis/blob/main/notebooks/quickstart.ipynb)

**What's in the notebook:**

1. Install Mantis (< 30 seconds)
2. Load sample data
3. Run your first backtest
4. Validate results
5. Visualize equity curve

## Try It Now

Copy and paste this into a new Colab notebook:

```python
# Install Mantis
!pip install mantis-bt -q

import mantis as mt
import numpy as np

# Load sample data (bundled, works offline)
data = mt.load_sample("AAPL")
print(f"Loaded {len(data['bars'])} bars of AAPL data")

# Create a simple signal
np.random.seed(42)
signal = np.random.choice([-1, 0, 1], size=len(data))

# Run backtest
results = mt.backtest(data, signal)
print(results)

# Validate
validation = results.validate()
print(f"\nVerdict: {validation.verdict}")
print(f"OOS/IS: {validation.efficiency_ratio:.0%}")
```

## Binder

For a full Python environment:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/johan-gras/mantis/main?labpath=notebooks%2Fquickstart.ipynb)

**Note:** Binder takes longer to start but provides a complete Jupyter environment.

## Local Quick Start

Prefer to run locally? Install in < 30 seconds:

```bash
# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install Mantis
pip install mantis-bt

# Start Python
python
```

Then:

```python
import mantis as mt
import numpy as np

data = mt.load_sample("AAPL")
signal = np.random.choice([-1, 0, 1], size=len(data))
results = mt.backtest(data, signal)
print(results)
```

## Sample Notebooks

| Notebook | Description |
|----------|-------------|
| [Quick Start](https://colab.research.google.com/github/johan-gras/mantis/blob/main/notebooks/quickstart.ipynb) | Your first backtest |
| [Validation](https://colab.research.google.com/github/johan-gras/mantis/blob/main/notebooks/validation.ipynb) | Walk-forward validation |
| [Multi-Symbol](https://colab.research.google.com/github/johan-gras/mantis/blob/main/notebooks/multi_symbol.ipynb) | Portfolio backtesting |

## What You Can Do

### Load and explore data

```python
data = mt.load_sample("SPY")
print(f"Bars: {len(data['bars'])}")
print(f"Date range: {data['timestamp'][0]} to {data['timestamp'][-1]}")
```

### Run backtests

```python
# With a signal array
results = mt.backtest(data, signal)

# With a built-in strategy
results = mt.backtest(data, strategy="sma-crossover")
```

### Validate strategies

```python
validation = results.validate()
print(validation.verdict)
```

### Visualize results

```python
# Interactive Plotly chart in Colab
results.plot()
```

### Compare strategies

```python
results_a = mt.backtest(data, signal_a)
results_b = mt.backtest(data, signal_b)
print(mt.compare([results_a, results_b], names=["A", "B"]))
```

## Tips for Colab

1. **GPU not needed** - Mantis is CPU-bound and very fast
2. **Free tier is fine** - Backtests complete in seconds
3. **Mount Drive** - To use your own data:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   data = mt.load('/content/drive/MyDrive/data.csv')
   ```

## Need Help?

- [Quick Start Guide](quickstart.md)
- [Cookbook](cookbook/index.md)
- [API Reference](api/index.md)
- [GitHub Issues](https://github.com/johan-gras/mantis/issues)
