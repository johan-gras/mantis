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
    backtest as _backtest_raw,
    signal_check,
    validate as _validate_raw,
    BacktestResult as _BacktestResult,
    ValidationResult as _ValidationResult,
    FoldDetail,
    Bar,
    BacktestConfig,
)


# =============================================================================
# Jupyter/Plotly detection utilities
# =============================================================================

def _is_jupyter() -> bool:
    """Detect if we're running in a Jupyter notebook environment."""
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is None:
            return False
        # Check for known Jupyter shell classes
        shell_name = shell.__class__.__name__
        return shell_name in ('ZMQInteractiveShell', 'TerminalInteractiveShell')
    except (ImportError, NameError):
        return False


def _has_plotly() -> bool:
    """Check if plotly is installed."""
    try:
        import plotly
        return True
    except ImportError:
        return False


# =============================================================================
# Result wrapper classes with Plotly support
# =============================================================================

class BacktestResult:
    """
    Wrapper around the Rust BacktestResult that adds Plotly visualization support.

    In Jupyter notebooks with plotly installed, plot() returns an interactive
    Plotly figure. In other environments, it falls back to ASCII sparklines.
    """

    def __init__(self, rust_result: _BacktestResult):
        self._rust = rust_result

    # Forward all attributes to the underlying Rust result
    @property
    def strategy_name(self) -> str:
        return self._rust.strategy_name

    @property
    def symbols(self) -> List[str]:
        return self._rust.symbols

    @property
    def initial_capital(self) -> float:
        return self._rust.initial_capital

    @property
    def final_equity(self) -> float:
        return self._rust.final_equity

    @property
    def total_return(self) -> float:
        return self._rust.total_return

    @property
    def cagr(self) -> float:
        return self._rust.cagr

    @property
    def sharpe(self) -> float:
        return self._rust.sharpe

    @property
    def sortino(self) -> float:
        return self._rust.sortino

    @property
    def calmar(self) -> float:
        return self._rust.calmar

    @property
    def max_drawdown(self) -> float:
        return self._rust.max_drawdown

    @property
    def win_rate(self) -> float:
        return self._rust.win_rate

    @property
    def profit_factor(self) -> float:
        return self._rust.profit_factor

    @property
    def total_trades(self) -> int:
        return self._rust.total_trades

    @property
    def winning_trades(self) -> int:
        return self._rust.winning_trades

    @property
    def losing_trades(self) -> int:
        return self._rust.losing_trades

    @property
    def avg_win(self) -> float:
        return self._rust.avg_win

    @property
    def avg_loss(self) -> float:
        return self._rust.avg_loss

    @property
    def trading_days(self) -> int:
        return self._rust.trading_days

    @property
    def equity_curve(self) -> np.ndarray:
        return self._rust.equity_curve

    @property
    def equity_timestamps(self) -> np.ndarray:
        return self._rust.equity_timestamps

    @property
    def trades(self) -> List[Any]:
        return self._rust.trades

    @property
    def deflated_sharpe(self) -> float:
        return self._rust.deflated_sharpe

    @property
    def psr(self) -> float:
        return self._rust.psr

    def metrics(self) -> Dict[str, Any]:
        return self._rust.metrics()

    def summary(self) -> str:
        return self._rust.summary()

    def warnings(self) -> List[str]:
        return self._rust.warnings()

    def save(self, path: str) -> None:
        return self._rust.save(path)

    def report(self, path: str) -> None:
        return self._rust.report(path)

    def validate(
        self,
        folds: int = 12,
        train_ratio: float = 0.75,
        anchored: bool = True,
    ) -> "ValidationResult":
        rust_validation = self._rust.validate(folds, train_ratio, anchored)
        return ValidationResult(rust_validation)

    def plot(self, width: int = 40, show_drawdown: bool = True) -> Any:
        """
        Display a visualization of the equity curve.

        In Jupyter notebooks with plotly installed, returns an interactive
        Plotly figure with equity curve and drawdown subplots.
        In terminal or without plotly, returns an ASCII sparkline string.

        Args:
            width: Width of the visualization (characters for ASCII, ignored for Plotly)
            show_drawdown: Whether to show drawdown subplot (Plotly only)

        Returns:
            Plotly Figure object in Jupyter with plotly, ASCII string otherwise.

        Example:
            >>> results = mt.backtest(data, signal)
            >>> results.plot()  # Shows interactive chart in Jupyter
        """
        if _is_jupyter() and _has_plotly():
            return self._plot_plotly(show_drawdown)
        else:
            return self._rust.plot(width)

    def _plot_plotly(self, show_drawdown: bool = True) -> Any:
        """Create an interactive Plotly figure."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import datetime

        # Get data
        equity = self.equity_curve
        timestamps = self.equity_timestamps

        # Convert timestamps to datetime
        dates = [datetime.datetime.fromtimestamp(ts) for ts in timestamps]

        # Calculate drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak * 100  # as percentage

        if show_drawdown:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.08,
                row_heights=[0.7, 0.3],
                subplot_titles=('Equity Curve', 'Drawdown')
            )

            # Equity curve
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=equity,
                    mode='lines',
                    name='Equity',
                    line=dict(color='#2E86AB', width=2),
                    hovertemplate='%{x}<br>Equity: $%{y:,.0f}<extra></extra>'
                ),
                row=1, col=1
            )

            # Drawdown
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=drawdown,
                    mode='lines',
                    name='Drawdown',
                    line=dict(color='#E94F37', width=1.5),
                    fill='tozeroy',
                    fillcolor='rgba(233, 79, 55, 0.3)',
                    hovertemplate='%{x}<br>Drawdown: %{y:.1f}%<extra></extra>'
                ),
                row=2, col=1
            )

            # Update layout
            fig.update_layout(
                title=dict(
                    text=f"Backtest Results: {', '.join(self.symbols)}",
                    x=0.5,
                    xanchor='center'
                ),
                showlegend=False,
                height=500,
                margin=dict(l=60, r=40, t=60, b=40),
                hovermode='x unified'
            )

            # Add annotation with key metrics
            metrics_text = (
                f"Return: {self.total_return*100:+.1f}% | "
                f"Sharpe: {self.sharpe:.2f} | "
                f"Max DD: {self.max_drawdown*100:.1f}%"
            )
            fig.add_annotation(
                text=metrics_text,
                xref="paper", yref="paper",
                x=0.5, y=1.08,
                showarrow=False,
                font=dict(size=11, color='gray')
            )

            fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
            fig.update_yaxes(title_text="DD (%)", row=2, col=1)

        else:
            # Single plot without drawdown
            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=equity,
                    mode='lines',
                    name='Equity',
                    line=dict(color='#2E86AB', width=2),
                    hovertemplate='%{x}<br>Equity: $%{y:,.0f}<extra></extra>'
                )
            )

            fig.update_layout(
                title=dict(
                    text=f"Backtest Results: {', '.join(self.symbols)}",
                    x=0.5,
                    xanchor='center'
                ),
                showlegend=False,
                height=400,
                margin=dict(l=60, r=40, t=60, b=40),
                hovermode='x unified',
                yaxis_title="Equity ($)"
            )

        return fig

    def _repr_html_(self) -> str:
        """Rich HTML display for Jupyter notebooks."""
        if _is_jupyter() and _has_plotly():
            # Return the Plotly figure's HTML
            fig = self._plot_plotly()
            return fig.to_html(include_plotlyjs='cdn', full_html=False)
        else:
            # Return a simple HTML summary
            return f"""
            <div style="font-family: monospace; padding: 10px;">
                <h3>Backtest Results: {', '.join(self.symbols)}</h3>
                <table>
                    <tr><td>Total Return:</td><td>{self.total_return*100:+.1f}%</td></tr>
                    <tr><td>Sharpe Ratio:</td><td>{self.sharpe:.2f}</td></tr>
                    <tr><td>Max Drawdown:</td><td>{self.max_drawdown*100:.1f}%</td></tr>
                    <tr><td>Win Rate:</td><td>{self.win_rate*100:.1f}%</td></tr>
                    <tr><td>Total Trades:</td><td>{self.total_trades}</td></tr>
                </table>
            </div>
            """

    def __repr__(self) -> str:
        return repr(self._rust)

    def __str__(self) -> str:
        return str(self._rust)


class ValidationResult:
    """
    Wrapper around the Rust ValidationResult that adds Plotly visualization support.

    In Jupyter notebooks with plotly installed, plot() returns an interactive
    Plotly figure. In other environments, it falls back to ASCII visualization.
    """

    def __init__(self, rust_result: _ValidationResult):
        self._rust = rust_result

    # Forward all attributes
    @property
    def folds(self) -> int:
        return self._rust.folds

    @property
    def is_sharpe(self) -> float:
        return self._rust.is_sharpe

    @property
    def oos_sharpe(self) -> float:
        return self._rust.oos_sharpe

    @property
    def oos_degradation(self) -> float:
        return self._rust.oos_degradation

    @property
    def verdict(self) -> str:
        return self._rust.verdict

    @property
    def avg_is_return(self) -> float:
        return self._rust.avg_is_return

    @property
    def avg_oos_return(self) -> float:
        return self._rust.avg_oos_return

    @property
    def efficiency_ratio(self) -> float:
        return self._rust.efficiency_ratio

    @property
    def parameter_stability(self) -> float:
        return self._rust.parameter_stability

    def fold_details(self) -> List[FoldDetail]:
        return self._rust.fold_details()

    def is_robust(self) -> bool:
        return self._rust.is_robust()

    def summary(self) -> str:
        return self._rust.summary()

    def report(self, path: str) -> None:
        return self._rust.report(path)

    def plot(self, width: int = 20) -> Any:
        """
        Display a visualization of fold-by-fold performance.

        In Jupyter notebooks with plotly installed, returns an interactive
        Plotly figure with in-sample vs out-of-sample comparison.
        In terminal or without plotly, returns an ASCII bar chart string.

        Args:
            width: Width of ASCII bars (ignored for Plotly)

        Returns:
            Plotly Figure object in Jupyter with plotly, ASCII string otherwise.

        Example:
            >>> validation = mt.validate(data, signal)
            >>> validation.plot()  # Shows interactive chart in Jupyter
        """
        if _is_jupyter() and _has_plotly():
            return self._plot_plotly()
        else:
            return self._rust.plot(width)

    def _plot_plotly(self) -> Any:
        """Create an interactive Plotly figure for validation results."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        folds = self.fold_details()
        fold_nums = [f"Fold {f.fold}" for f in folds]
        is_sharpes = [f.is_sharpe for f in folds]
        oos_sharpes = [f.oos_sharpe for f in folds]
        efficiencies = [f.efficiency * 100 for f in folds]

        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.12,
            row_heights=[0.6, 0.4],
            subplot_titles=('Sharpe Ratio by Fold', 'Efficiency (%)')
        )

        # IS Sharpe bars
        fig.add_trace(
            go.Bar(
                x=fold_nums,
                y=is_sharpes,
                name='In-Sample',
                marker_color='#2E86AB',
                hovertemplate='%{x}<br>IS Sharpe: %{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )

        # OOS Sharpe bars
        fig.add_trace(
            go.Bar(
                x=fold_nums,
                y=oos_sharpes,
                name='Out-of-Sample',
                marker_color='#E94F37',
                hovertemplate='%{x}<br>OOS Sharpe: %{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )

        # Efficiency line
        fig.add_trace(
            go.Scatter(
                x=fold_nums,
                y=efficiencies,
                mode='lines+markers',
                name='Efficiency',
                line=dict(color='#28A745', width=2),
                marker=dict(size=8),
                hovertemplate='%{x}<br>Efficiency: %{y:.0f}%<extra></extra>'
            ),
            row=2, col=1
        )

        # Add reference line at 100% efficiency
        fig.add_hline(
            y=100, line_dash="dash", line_color="gray",
            annotation_text="100%", row=2, col=1
        )

        # Determine verdict color
        verdict_colors = {
            'robust': '#28A745',
            'borderline': '#FFC107',
            'likely_overfit': '#DC3545'
        }
        verdict_color = verdict_colors.get(self.verdict, '#6C757D')

        fig.update_layout(
            title=dict(
                text=f"Walk-Forward Analysis ({self.folds} folds) - Verdict: {self.verdict.upper()}",
                x=0.5,
                xanchor='center',
                font=dict(color=verdict_color)
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            height=500,
            margin=dict(l=60, r=40, t=80, b=40),
            barmode='group'
        )

        # Add summary annotation
        summary_text = (
            f"IS Sharpe: {self.is_sharpe:.2f} | "
            f"OOS Sharpe: {self.oos_sharpe:.2f} | "
            f"Degradation: {self.oos_degradation*100:.0f}%"
        )
        fig.add_annotation(
            text=summary_text,
            xref="paper", yref="paper",
            x=0.5, y=-0.1,
            showarrow=False,
            font=dict(size=11, color='gray')
        )

        fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=1)
        fig.update_yaxes(title_text="Efficiency (%)", row=2, col=1)

        return fig

    def _repr_html_(self) -> str:
        """Rich HTML display for Jupyter notebooks."""
        if _is_jupyter() and _has_plotly():
            fig = self._plot_plotly()
            return fig.to_html(include_plotlyjs='cdn', full_html=False)
        else:
            verdict_colors = {
                'robust': 'green',
                'borderline': 'orange',
                'likely_overfit': 'red'
            }
            color = verdict_colors.get(self.verdict, 'gray')
            return f"""
            <div style="font-family: monospace; padding: 10px;">
                <h3>Walk-Forward Analysis ({self.folds} folds)</h3>
                <p><strong>Verdict:</strong> <span style="color: {color}; font-weight: bold;">{self.verdict.upper()}</span></p>
                <table>
                    <tr><td>IS Sharpe:</td><td>{self.is_sharpe:.2f}</td></tr>
                    <tr><td>OOS Sharpe:</td><td>{self.oos_sharpe:.2f}</td></tr>
                    <tr><td>OOS Degradation:</td><td>{self.oos_degradation*100:.0f}%</td></tr>
                    <tr><td>Efficiency:</td><td>{self.efficiency_ratio*100:.0f}%</td></tr>
                </table>
            </div>
            """

    def __repr__(self) -> str:
        return repr(self._rust)

    def __str__(self) -> str:
        return str(self._rust)


# =============================================================================
# Wrapper functions that return wrapped results
# =============================================================================

def backtest(
    data: Any,
    signal: Optional[np.ndarray] = None,
    strategy: Optional[str] = None,
    strategy_params: Optional[Dict[str, Any]] = None,
    config: Optional[BacktestConfig] = None,
    commission: float = 0.001,
    slippage: float = 0.001,
    size: float = 0.10,
    cash: float = 100_000.0,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    allow_short: bool = True,
    borrow_cost: float = 0.03,
    max_position: float = 1.0,
    fill_price: str = "next_open",
) -> BacktestResult:
    """
    Run a backtest on historical data with a signal array.

    See module documentation for full details.
    """
    rust_result = _backtest_raw(
        data, signal, strategy, strategy_params, config,
        commission, slippage, size, cash, stop_loss,
        take_profit, allow_short, borrow_cost, max_position, fill_price
    )
    return BacktestResult(rust_result)


def validate(
    data: Any,
    signal: np.ndarray,
    folds: int = 12,
    in_sample_ratio: float = 0.75,
    anchored: bool = True,
    config: Optional[BacktestConfig] = None,
    commission: float = 0.001,
    slippage: float = 0.001,
    size: float = 0.10,
    cash: float = 100_000.0,
) -> ValidationResult:
    """
    Run walk-forward validation on a signal-based strategy.

    See module documentation for full details.
    """
    rust_result = _validate_raw(
        data, signal, folds, in_sample_ratio, anchored,
        config, commission, slippage, size, cash
    )
    return ValidationResult(rust_result)

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
            "max_position": 1.0,
            "fill_price": "next_open",
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

    def max_position(self, fraction: float) -> "Backtest":
        """
        Set the maximum position size as a fraction of equity.

        Args:
            fraction: Maximum position size (e.g., 0.25 = 25% of equity)

        Returns:
            Self for method chaining
        """
        self._config["max_position"] = fraction
        return self

    def fill_price(self, model: str) -> "Backtest":
        """
        Set the execution price model.

        Args:
            model: Execution price model. Options:
                - "next_open" (default): Fill at next bar's open price
                - "close": Fill at bar's close price
                - "vwap": Volume-weighted average price approximation
                - "twap": Time-weighted average price approximation
                - "midpoint": Average of high and low

        Returns:
            Self for method chaining
        """
        self._config["fill_price"] = model
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
