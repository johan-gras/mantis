# Performance Benchmarks

Mantis is designed to be fast enough to think. Here's how it performs compared to alternatives.

## Mantis Performance

| Operation | Time | Notes |
|-----------|------|-------|
| 10 years daily | < 100ms | Single symbol, 2520 bars |
| 1000 param sweep | < 30s | Parallel on 8 cores |
| Walk-forward 12 folds | < 2s | Full validation suite |

*Benchmarks run on AMD Ryzen 9 5900X, Ubuntu 22.04, Rust 1.75. Results may vary +/- 20%.*

## Reproducing Benchmarks

```bash
# Clone the repository
git clone https://github.com/johan-gras/mantis
cd mantis

# Run benchmarks
cargo bench --bench backtest_bench

# Results saved to target/criterion/
```

Note: First run includes compilation. Subsequent runs measure actual performance.

## Competitor Comparison

We've benchmarked Mantis against popular backtesting frameworks. All tests use:

- Same data: 10 years daily OHLCV (2520 bars)
- Same signal: Random signal array
- Same metrics: Sharpe, max drawdown, total return
- Default settings for each framework

### Comparison Matrix

| Framework | 10y Daily | 1000 Sweep | Validation | Notes |
|-----------|-----------|------------|------------|-------|
| **Mantis** | ~85ms | ~28s | Built-in | Validation-first design |
| Backtrader | ~2.4s | ~40min | Manual | More execution models |
| VectorBT | ~120ms | ~35s | Built-in | Better visualization |
| Zipline | ~3.1s | N/A | Manual | Pipeline for factors |

### Where Competitors Excel

**Backtrader:**

- More sophisticated execution models (stop-limit, bracket orders)
- Live trading integration
- Larger community with more examples

**VectorBT:**

- Superior visualization capabilities
- More built-in technical indicators
- Portfolio optimization tools

**Zipline:**

- Institutional-grade architecture
- Pipeline API for factor research
- Battle-tested at Quantopian

### Where Mantis Excels

**vs Backtrader:**

- ~30x faster for typical backtests
- ML-native interface (numpy arrays, not callbacks)
- Built-in validation suite

**vs VectorBT:**

- Simpler API (one function, not method chains)
- Better statistical validation (DSR, PSR)
- Clearer error messages

**vs Zipline:**

- ~30x faster
- Modern Python (no legacy dependencies)
- Actually installs without issues

## Benchmark Methodology

### Fairness Requirements

1. **Default settings**: All frameworks use their default configuration
2. **Same data**: Identical OHLCV data passed to each framework
3. **Same signal**: Same signal array (converted to framework-specific format)
4. **Same metrics**: Request Sharpe, max drawdown, total return
5. **Same machine**: Run on same hardware in same session
6. **Statistical**: Report median of 10 runs

### Caveats

- VectorBT times include numba JIT warmup on first run
- Backtrader uses event-driven model (inherently slower for vectorizable operations)
- Zipline requires additional data setup that isn't timed
- All frameworks tested on their latest stable versions as of January 2026

### Running Competitor Benchmarks

```python
# scripts/competitor_bench.py provides the benchmark code
# Requires: pip install backtrader vectorbt zipline-reloaded

python scripts/competitor_bench.py
```

!!! note "Focus on Validation"
    Speed is table stakes. Mantis's real differentiator is validation-first design.
    Other backtesters show impressive numbers. Mantis asks if those numbers are real.

---

> **Disclaimer:** Past performance does not guarantee future results. Backtest results are hypothetical and do not represent actual trading. Real trading involves risk of loss.
