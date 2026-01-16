"""
Mantis - A high-performance backtesting engine for quantitative trading.

Mantis is a backtester for deep learning quants who value their time.
It's designed to be correct by default, fast enough to think, and ML-native.

Quick Start:
    >>> import mantis as mt
    >>> data = mt.load("AAPL.csv")
    >>> signal = model.predict(features)  # Your model's output
    >>> results = mt.backtest(data, signal)
    >>> print(results)

Key Features:
- Correct by default: Market orders fill at next bar's open
- Conservative costs: 0.1% commission and slippage by default
- Signal-based: Works with numpy arrays from your ML models
- Fast: Written in Rust, 10 years of data backtested in <100ms
- Honest: Validation tools to detect overfitting
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np

# Import the Rust extension module
from mantis._mantis import (
    __version__,
    load,
    load_dir,
    load_multi,
    load_sample,
    list_samples,
    backtest,
    signal_check,
    BacktestResult,
    ValidationResult,
    Bar,
    BacktestConfig,
)

# Re-export all public API
__all__ = [
    "__version__",
    "load",
    "load_dir",
    "load_multi",
    "load_sample",
    "list_samples",
    "backtest",
    "signal_check",
    "BacktestResult",
    "ValidationResult",
    "Bar",
    "BacktestConfig",
    "compare",
    "sweep",
]


def compare(
    results: List[BacktestResult],
    names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compare multiple backtest results side by side.

    Args:
        results: List of BacktestResult objects to compare
        names: Optional list of names for each result

    Returns:
        Dictionary with comparison metrics

    Example:
        >>> lstm_results = mt.backtest(data, lstm_signal)
        >>> baseline = mt.backtest(data, baseline_signal)
        >>> comparison = mt.compare([lstm_results, baseline], names=["LSTM", "Baseline"])
    """
    if names is None:
        names = [f"Strategy {i+1}" for i in range(len(results))]

    comparison = {}
    for name, result in zip(names, results):
        comparison[name] = {
            "total_return": result.total_return,
            "sharpe": result.sharpe,
            "max_drawdown": result.max_drawdown,
            "win_rate": result.win_rate,
            "total_trades": result.total_trades,
            "profit_factor": result.profit_factor,
        }

    return comparison


def sweep(
    data: Any,
    signal_fn: callable,
    params: Dict[str, List[Any]],
    n_jobs: int = -1,
    **backtest_kwargs,
) -> Dict[str, Any]:
    """
    Run a parameter sweep, testing multiple parameter combinations.

    Args:
        data: Data dictionary from load() or file path
        signal_fn: Function that takes params and returns a signal array
        params: Dictionary of parameter names to lists of values to test
        n_jobs: Number of parallel jobs (-1 for all cores)
        **backtest_kwargs: Additional arguments passed to backtest()

    Returns:
        Dictionary with results for each parameter combination

    Example:
        >>> def generate_signal(threshold):
        ...     return (predictions > threshold).astype(int)
        >>>
        >>> sweep_results = mt.sweep(
        ...     data,
        ...     generate_signal,
        ...     params={"threshold": [0.1, 0.2, 0.3, 0.4, 0.5]}
        ... )
        >>> best = max(sweep_results.items(), key=lambda x: x[1].sharpe)
    """
    from itertools import product

    # Generate all parameter combinations
    param_names = list(params.keys())
    param_values = list(params.values())
    combinations = list(product(*param_values))

    results = {}
    for combo in combinations:
        param_dict = dict(zip(param_names, combo))
        param_key = str(param_dict)

        # Generate signal with these parameters
        if len(param_names) == 1:
            signal = signal_fn(combo[0])
        else:
            signal = signal_fn(**param_dict)

        # Run backtest
        result = backtest(data, signal, **backtest_kwargs)
        results[param_key] = result

    return results


def _format_comparison_table(comparison: Dict[str, Dict[str, Any]]) -> str:
    """Format comparison as a table string."""
    header = f"{'Strategy':<15} {'Return':>10} {'Sharpe':>8} {'Max DD':>10} {'Win Rate':>10}"
    lines = [header, "-" * len(header)]

    for name, metrics in comparison.items():
        line = (
            f"{name:<15} "
            f"{metrics['total_return']*100:>+9.1f}% "
            f"{metrics['sharpe']:>8.2f} "
            f"{metrics['max_drawdown']*100:>9.1f}% "
            f"{metrics['win_rate']*100:>9.1f}%"
        )
        lines.append(line)

    return "\n".join(lines)
