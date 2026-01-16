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
    load_results,
    backtest,
    signal_check,
    validate,
    BacktestResult,
    ValidationResult,
    FoldDetail,
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
    "load_results",
    "backtest",
    "signal_check",
    "validate",
    "BacktestResult",
    "ValidationResult",
    "FoldDetail",
    "Bar",
    "BacktestConfig",
    "Backtest",
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


class Backtest:
    """
    Fluent API for configuring and running backtests.

    Provides a chainable interface for setting backtest parameters
    before running. All parameters have sensible defaults.

    Example:
        >>> results = (
        ...     mt.Backtest(data, signal)
        ...     .commission(0.001)
        ...     .slippage(0.0005)
        ...     .size(0.15)
        ...     .run()
        ... )
        >>> print(results.sharpe)
        1.24

    See Also:
        backtest(): Functional API that accepts all parameters directly
    """

    def __init__(
        self,
        data: Any,
        signal: Optional[np.ndarray] = None,
        strategy: Optional[str] = None,
        strategy_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a backtest configuration.

        Args:
            data: Data dictionary from load() or path to CSV/Parquet file
            signal: numpy array of signals (1=long, -1=short, 0=flat)
            strategy: Name of built-in strategy if signal is None
            strategy_params: Dictionary of strategy parameters
        """
        self._data = data
        self._signal = signal
        self._strategy = strategy
        self._strategy_params = strategy_params

        # Default configuration values
        self._config = {
            "commission": 0.001,
            "slippage": 0.001,
            "size": 0.10,
            "cash": 100_000.0,
            "stop_loss": None,
            "take_profit": None,
            "allow_short": True,
            "borrow_cost": 0.03,
        }

    def commission(self, rate: float) -> "Backtest":
        """
        Set the commission rate.

        Args:
            rate: Commission as a decimal (e.g., 0.001 = 0.1%)

        Returns:
            Self for method chaining
        """
        self._config["commission"] = rate
        return self

    def slippage(self, rate: float) -> "Backtest":
        """
        Set the slippage rate.

        Args:
            rate: Slippage as a decimal (e.g., 0.001 = 0.1%)

        Returns:
            Self for method chaining
        """
        self._config["slippage"] = rate
        return self

    def size(self, fraction: float) -> "Backtest":
        """
        Set the position size as a fraction of equity.

        Args:
            fraction: Position size (e.g., 0.10 = 10% of equity)

        Returns:
            Self for method chaining
        """
        self._config["size"] = fraction
        return self

    def cash(self, amount: float) -> "Backtest":
        """
        Set the initial capital.

        Args:
            amount: Starting capital in dollars

        Returns:
            Self for method chaining
        """
        self._config["cash"] = amount
        return self

    def stop_loss(self, pct: float) -> "Backtest":
        """
        Set a stop loss percentage.

        Args:
            pct: Stop loss as a decimal (e.g., 0.05 = 5%)

        Returns:
            Self for method chaining
        """
        self._config["stop_loss"] = pct
        return self

    def take_profit(self, pct: float) -> "Backtest":
        """
        Set a take profit percentage.

        Args:
            pct: Take profit as a decimal (e.g., 0.10 = 10%)

        Returns:
            Self for method chaining
        """
        self._config["take_profit"] = pct
        return self

    def allow_short(self, enabled: bool = True) -> "Backtest":
        """
        Enable or disable short selling.

        Args:
            enabled: Whether to allow short positions

        Returns:
            Self for method chaining
        """
        self._config["allow_short"] = enabled
        return self

    def borrow_cost(self, rate: float) -> "Backtest":
        """
        Set the annual borrow cost for short positions.

        Args:
            rate: Annual borrow rate (e.g., 0.03 = 3%)

        Returns:
            Self for method chaining
        """
        self._config["borrow_cost"] = rate
        return self

    def run(self) -> BacktestResult:
        """
        Execute the backtest with the configured parameters.

        Returns:
            BacktestResult with metrics, equity curve, and trades
        """
        return backtest(
            self._data,
            self._signal,
            strategy=self._strategy,
            strategy_params=self._strategy_params,
            **self._config,
        )
