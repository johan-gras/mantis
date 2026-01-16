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
    adjust,
    backtest as _backtest_raw,
    signal_check,
    validate as _validate_raw,
    BacktestResult as _BacktestResult,
    ValidationResult as _ValidationResult,
    FoldDetail,
    Bar,
    BacktestConfig,
    # Sensitivity analysis
    sensitivity as _sensitivity_raw,
    cost_sensitivity as _cost_sensitivity_raw,
    linear_range,
    log_range,
    discrete_range,
    centered_range,
    ParameterRange,
    SensitivityResult as _SensitivityResult,
    SensitivitySummary,
    HeatmapData,
    Cliff,
    Plateau,
    CostSensitivityResult as _CostSensitivityResult,
    CostScenario,
    # Parallel sweep
    sweep as _sweep_raw,
    SweepResult as _SweepResult,
    SweepResultItem,
)

# Try to import ONNX classes (only available when built with --features onnx)
try:
    from mantis._mantis import (
        ModelConfig,
        InferenceStats,
        OnnxModel,
        load_model,
        generate_signals,
    )
    _HAS_ONNX = True
except ImportError:
    _HAS_ONNX = False
    # Define placeholder types for when ONNX is not available
    ModelConfig = None  # type: ignore
    InferenceStats = None  # type: ignore
    OnnxModel = None  # type: ignore
    def load_model(*args, **kwargs):
        raise RuntimeError(
            "ONNX support is not enabled. "
            "Rebuild with `--features onnx` to enable ONNX model inference."
        )
    def generate_signals(*args, **kwargs):
        raise RuntimeError(
            "ONNX support is not enabled. "
            "Rebuild with `--features onnx` to enable ONNX model inference."
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

    # Benchmark comparison properties (only available if benchmark was provided)
    @property
    def alpha(self) -> Optional[float]:
        """Jensen's alpha (risk-adjusted excess return). None if no benchmark was provided."""
        return self._rust.alpha

    @property
    def beta(self) -> Optional[float]:
        """Portfolio beta (sensitivity to benchmark). None if no benchmark was provided."""
        return self._rust.beta

    @property
    def benchmark_return(self) -> Optional[float]:
        """Benchmark total return for the period. None if no benchmark was provided."""
        return self._rust.benchmark_return

    @property
    def excess_return(self) -> Optional[float]:
        """Excess return (strategy - benchmark). None if no benchmark was provided."""
        return self._rust.excess_return

    @property
    def tracking_error(self) -> Optional[float]:
        """Annualized tracking error. None if no benchmark was provided."""
        return self._rust.tracking_error

    @property
    def information_ratio(self) -> Optional[float]:
        """Information ratio (alpha / tracking error). None if no benchmark was provided."""
        return self._rust.information_ratio

    @property
    def benchmark_correlation(self) -> Optional[float]:
        """Correlation with benchmark (-1 to 1). None if no benchmark was provided."""
        return self._rust.benchmark_correlation

    @property
    def up_capture(self) -> Optional[float]:
        """Up-capture ratio. None if no benchmark was provided."""
        return self._rust.up_capture

    @property
    def down_capture(self) -> Optional[float]:
        """Down-capture ratio. None if no benchmark was provided."""
        return self._rust.down_capture

    @property
    def has_benchmark(self) -> bool:
        """Whether benchmark comparison metrics are available."""
        return self._rust.has_benchmark

    @property
    def volatility(self) -> float:
        """Annualized volatility (standard deviation of returns). E.g., 0.156 = 15.6%."""
        return self._rust.volatility

    @property
    def max_drawdown_duration(self) -> int:
        """Maximum drawdown duration in days (longest time from peak to recovery)."""
        return self._rust.max_drawdown_duration

    @property
    def avg_trade_duration(self) -> float:
        """Average trade holding period in days."""
        return self._rust.avg_trade_duration

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

    def plot(
        self,
        width: int = 40,
        show_drawdown: bool = True,
        trades: bool = False,
        benchmark: bool = False,
        save: Optional[str] = None,
        title: Optional[str] = None,
        height: Optional[int] = None,
        theme: Optional[str] = None,
    ) -> Any:
        """
        Display a visualization of the equity curve.

        In Jupyter notebooks with plotly installed, returns an interactive
        Plotly figure with equity curve and drawdown subplots.
        In terminal or without plotly, returns an ASCII sparkline string.

        Args:
            width: Width of the visualization (characters for ASCII, ignored for Plotly)
            show_drawdown: Whether to show drawdown subplot (Plotly only)
            trades: Whether to show trade entry/exit markers on the chart (Plotly only)
            benchmark: Whether to show benchmark comparison if available (Plotly only)
            save: Save the plot to a file. Supports .html, .png, .pdf, .svg extensions.
                  Requires kaleido for image export (pip install kaleido)
            title: Custom title for the plot
            height: Custom height in pixels (Plotly only)
            theme: Color theme - "light" or "dark" (Plotly only)

        Returns:
            Plotly Figure object in Jupyter with plotly, ASCII string otherwise.
            If save is provided, saves to file and returns the path.

        Example:
            >>> results = mt.backtest(data, signal)
            >>> results.plot()  # Shows interactive chart in Jupyter
            >>> results.plot(save="report.html")  # Save to HTML file
            >>> results.plot(trades=True, theme="dark")  # Show trades with dark theme
        """
        if _has_plotly():
            fig = self._plot_plotly(
                show_drawdown=show_drawdown,
                trades=trades,
                benchmark=benchmark,
                title=title,
                height=height,
                theme=theme,
            )
            if save:
                return self._save_plot(fig, save)
            return fig
        else:
            if save:
                # ASCII can only be saved as text
                ascii_plot = self._rust.plot(width)
                if save.endswith('.txt'):
                    with open(save, 'w') as f:
                        f.write(ascii_plot)
                    return save
                else:
                    raise ValueError(
                        f"Cannot save ASCII plot to {save}. "
                        "Install plotly for HTML/PNG/PDF export, or use .txt extension."
                    )
            return self._rust.plot(width)

    def _plot_plotly(
        self,
        show_drawdown: bool = True,
        trades: bool = False,
        benchmark: bool = False,
        title: Optional[str] = None,
        height: Optional[int] = None,
        theme: Optional[str] = None,
    ) -> Any:
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

        # Determine colors based on theme
        if theme == "dark":
            bg_color = '#1e1e1e'
            paper_color = '#1e1e1e'
            text_color = '#e0e0e0'
            grid_color = '#404040'
            equity_color = '#4ecdc4'
            drawdown_color = '#ff6b6b'
            drawdown_fill = 'rgba(255, 107, 107, 0.3)'
            buy_color = '#00ff00'
            sell_color = '#ff0000'
            benchmark_color = '#888888'
        else:
            bg_color = 'white'
            paper_color = 'white'
            text_color = '#333333'
            grid_color = '#e0e0e0'
            equity_color = '#2E86AB'
            drawdown_color = '#E94F37'
            drawdown_fill = 'rgba(233, 79, 55, 0.3)'
            buy_color = '#2ca02c'
            sell_color = '#d62728'
            benchmark_color = '#888888'

        # Default title
        plot_title = title or f"Backtest Results: {', '.join(self.symbols)}"

        # Determine height
        if height is None:
            fig_height = 500 if show_drawdown else 400
        else:
            fig_height = height

        # Determine if we need a legend (trades or benchmark)
        show_legend = trades or benchmark

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
                    line=dict(color=equity_color, width=2),
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
                    line=dict(color=drawdown_color, width=1.5),
                    fill='tozeroy',
                    fillcolor=drawdown_fill,
                    hovertemplate='%{x}<br>Drawdown: %{y:.1f}%<extra></extra>',
                    showlegend=False,
                ),
                row=2, col=1
            )

            # Add benchmark if requested and available
            if benchmark and self.has_benchmark:
                benchmark_equity = self._get_benchmark_equity()
                if benchmark_equity is not None:
                    fig.add_trace(
                        go.Scatter(
                            x=dates,
                            y=benchmark_equity,
                            mode='lines',
                            name='Benchmark',
                            line=dict(color=benchmark_color, width=1.5, dash='dash'),
                            hovertemplate='%{x}<br>Benchmark: $%{y:,.0f}<extra></extra>'
                        ),
                        row=1, col=1
                    )

            # Add trade markers if requested
            if trades:
                self._add_trade_markers(fig, dates, equity, buy_color, sell_color, row=1)

            # Update layout
            fig.update_layout(
                title=dict(text=plot_title, x=0.5, xanchor='center'),
                showlegend=show_legend,
                height=fig_height,
                margin=dict(l=60, r=40, t=60, b=40),
                hovermode='x unified',
                plot_bgcolor=bg_color,
                paper_bgcolor=paper_color,
                font=dict(color=text_color),
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
                font=dict(size=11, color=text_color if theme != "dark" else '#aaaaaa')
            )

            fig.update_yaxes(title_text="Equity ($)", row=1, col=1, gridcolor=grid_color)
            fig.update_yaxes(title_text="DD (%)", row=2, col=1, gridcolor=grid_color)
            fig.update_xaxes(gridcolor=grid_color)

        else:
            # Single plot without drawdown
            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=equity,
                    mode='lines',
                    name='Equity',
                    line=dict(color=equity_color, width=2),
                    hovertemplate='%{x}<br>Equity: $%{y:,.0f}<extra></extra>'
                )
            )

            # Add benchmark if requested and available
            if benchmark and self.has_benchmark:
                benchmark_equity = self._get_benchmark_equity()
                if benchmark_equity is not None:
                    fig.add_trace(
                        go.Scatter(
                            x=dates,
                            y=benchmark_equity,
                            mode='lines',
                            name='Benchmark',
                            line=dict(color=benchmark_color, width=1.5, dash='dash'),
                            hovertemplate='%{x}<br>Benchmark: $%{y:,.0f}<extra></extra>'
                        )
                    )

            # Add trade markers if requested
            if trades:
                self._add_trade_markers(fig, dates, equity, buy_color, sell_color)

            fig.update_layout(
                title=dict(text=plot_title, x=0.5, xanchor='center'),
                showlegend=show_legend,
                height=fig_height,
                margin=dict(l=60, r=40, t=60, b=40),
                hovermode='x unified',
                yaxis_title="Equity ($)",
                plot_bgcolor=bg_color,
                paper_bgcolor=paper_color,
                font=dict(color=text_color),
            )
            fig.update_yaxes(gridcolor=grid_color)
            fig.update_xaxes(gridcolor=grid_color)

        return fig

    def _add_trade_markers(
        self,
        fig: Any,
        dates: list,
        equity: Any,
        buy_color: str,
        sell_color: str,
        row: Optional[int] = None,
    ) -> None:
        """Add trade entry/exit markers to the plot."""
        import plotly.graph_objects as go
        import datetime

        trades_list = self.trades
        if not trades_list:
            return

        # Create lookup for equity at each timestamp
        ts_to_idx = {}
        timestamps = self.equity_timestamps
        for i, ts in enumerate(timestamps):
            ts_to_idx[ts] = i

        buy_dates = []
        buy_equities = []
        buy_hovers = []
        sell_dates = []
        sell_equities = []
        sell_hovers = []

        for trade in trades_list:
            # Entry marker
            entry_ts = trade.entry_time
            if entry_ts in ts_to_idx:
                idx = ts_to_idx[entry_ts]
                entry_date = datetime.datetime.fromtimestamp(entry_ts)
                is_long = trade.quantity > 0
                if is_long:
                    buy_dates.append(entry_date)
                    buy_equities.append(equity[idx])
                    buy_hovers.append(
                        f"BUY {abs(trade.quantity):.2f} @ ${trade.entry_price:.2f}"
                    )
                else:
                    sell_dates.append(entry_date)
                    sell_equities.append(equity[idx])
                    sell_hovers.append(
                        f"SHORT {abs(trade.quantity):.2f} @ ${trade.entry_price:.2f}"
                    )

            # Exit marker
            exit_ts = trade.exit_time
            if exit_ts in ts_to_idx:
                idx = ts_to_idx[exit_ts]
                exit_date = datetime.datetime.fromtimestamp(exit_ts)
                is_long = trade.quantity > 0
                if is_long:
                    sell_dates.append(exit_date)
                    sell_equities.append(equity[idx])
                    sell_hovers.append(
                        f"SELL @ ${trade.exit_price:.2f} (P&L: ${trade.pnl:+.2f})"
                    )
                else:
                    buy_dates.append(exit_date)
                    buy_equities.append(equity[idx])
                    buy_hovers.append(
                        f"COVER @ ${trade.exit_price:.2f} (P&L: ${trade.pnl:+.2f})"
                    )

        # Add buy markers
        if buy_dates:
            trace_kwargs = dict(
                x=buy_dates,
                y=buy_equities,
                mode='markers',
                name='Buy',
                marker=dict(symbol='triangle-up', size=10, color=buy_color),
                hovertext=buy_hovers,
                hoverinfo='text+x',
            )
            if row is not None:
                fig.add_trace(go.Scatter(**trace_kwargs), row=row, col=1)
            else:
                fig.add_trace(go.Scatter(**trace_kwargs))

        # Add sell markers
        if sell_dates:
            trace_kwargs = dict(
                x=sell_dates,
                y=sell_equities,
                mode='markers',
                name='Sell',
                marker=dict(symbol='triangle-down', size=10, color=sell_color),
                hovertext=sell_hovers,
                hoverinfo='text+x',
            )
            if row is not None:
                fig.add_trace(go.Scatter(**trace_kwargs), row=row, col=1)
            else:
                fig.add_trace(go.Scatter(**trace_kwargs))

    def _get_benchmark_equity(self) -> Optional[Any]:
        """
        Calculate benchmark equity curve for comparison.
        Returns None if benchmark data is not available.
        """
        if not self.has_benchmark:
            return None

        # Calculate benchmark equity from benchmark return
        # Assume benchmark return is total return over the period
        initial_equity = self.equity_curve[0]
        benchmark_total = self.benchmark_return

        # Simple linear interpolation of benchmark equity
        # (In reality, we'd need daily benchmark returns, but this is a reasonable approximation)
        n = len(self.equity_curve)
        benchmark_equity = np.linspace(initial_equity, initial_equity * (1 + benchmark_total), n)

        return benchmark_equity

    def _save_plot(self, fig: Any, path: str) -> str:
        """Save the plot to a file."""
        import os
        ext = os.path.splitext(path)[1].lower()

        if ext == '.html':
            fig.write_html(path, include_plotlyjs='cdn')
        elif ext in ('.png', '.jpg', '.jpeg', '.webp', '.svg', '.pdf'):
            # Requires kaleido: pip install kaleido
            try:
                fig.write_image(path)
            except ValueError as e:
                if 'kaleido' in str(e).lower():
                    raise ValueError(
                        f"Image export requires kaleido. Install with: pip install kaleido"
                    ) from e
                raise
        else:
            raise ValueError(
                f"Unsupported file extension: {ext}. "
                "Supported: .html, .png, .jpg, .svg, .pdf"
            )

        return path

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

    def plot_drawdown(
        self,
        save: Optional[str] = None,
        title: Optional[str] = None,
        height: Optional[int] = None,
        theme: Optional[str] = None,
    ) -> Any:
        """
        Display a dedicated drawdown visualization.

        Shows the drawdown from peak equity over time, highlighting
        periods where the strategy was underwater.

        Args:
            save: Save the plot to a file. Supports .html, .png, .pdf, .svg.
            title: Custom title for the plot.
            height: Custom height in pixels (Plotly only).
            theme: Color theme - "light" or "dark" (Plotly only).

        Returns:
            Plotly Figure object in Jupyter with plotly, ASCII string otherwise.

        Example:
            >>> results = mt.backtest(data, signal)
            >>> results.plot_drawdown()  # Interactive drawdown chart
        """
        if _has_plotly():
            fig = self._plot_drawdown_plotly(title=title, height=height, theme=theme)
            if save:
                return self._save_plot(fig, save)
            return fig
        else:
            # ASCII fallback - show drawdown as sparkline
            equity = self.equity_curve
            peak = np.maximum.accumulate(equity)
            drawdown = (equity - peak) / peak * 100
            min_dd = min(drawdown)
            return f"Drawdown: [{min(drawdown):.1f}% to 0%] Max: {self.max_drawdown*100:.1f}%"

    def _plot_drawdown_plotly(
        self,
        title: Optional[str] = None,
        height: Optional[int] = None,
        theme: Optional[str] = None,
    ) -> Any:
        """Create a dedicated Plotly drawdown figure."""
        import plotly.graph_objects as go
        import datetime

        equity = self.equity_curve
        timestamps = self.equity_timestamps
        dates = [datetime.datetime.fromtimestamp(ts) for ts in timestamps]

        # Calculate drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak * 100  # as percentage

        # Theme colors
        if theme == "dark":
            bg_color = '#1e1e1e'
            paper_color = '#1e1e1e'
            text_color = '#e0e0e0'
            grid_color = '#404040'
            dd_color = '#ff6b6b'
            fill_color = 'rgba(255, 107, 107, 0.3)'
        else:
            bg_color = 'white'
            paper_color = 'white'
            text_color = '#333333'
            grid_color = '#e0e0e0'
            dd_color = '#E94F37'
            fill_color = 'rgba(233, 79, 55, 0.3)'

        plot_title = title or f"Drawdown: {', '.join(self.symbols)}"
        fig_height = height or 400

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=drawdown,
                mode='lines',
                name='Drawdown',
                line=dict(color=dd_color, width=1.5),
                fill='tozeroy',
                fillcolor=fill_color,
                hovertemplate='%{x}<br>Drawdown: %{y:.1f}%<extra></extra>'
            )
        )

        fig.update_layout(
            title=dict(text=plot_title, x=0.5, xanchor='center'),
            showlegend=False,
            height=fig_height,
            margin=dict(l=60, r=40, t=60, b=40),
            hovermode='x unified',
            yaxis_title="Drawdown (%)",
            plot_bgcolor=bg_color,
            paper_bgcolor=paper_color,
            font=dict(color=text_color),
        )
        fig.update_yaxes(gridcolor=grid_color)
        fig.update_xaxes(gridcolor=grid_color)

        # Add annotation with max drawdown
        fig.add_annotation(
            text=f"Max Drawdown: {self.max_drawdown*100:.1f}% | Duration: {self.max_drawdown_duration} days",
            xref="paper", yref="paper",
            x=0.5, y=1.05,
            showarrow=False,
            font=dict(size=11, color=text_color if theme != "dark" else '#aaaaaa')
        )

        return fig

    def plot_returns(
        self,
        period: str = "monthly",
        save: Optional[str] = None,
        title: Optional[str] = None,
        height: Optional[int] = None,
        theme: Optional[str] = None,
    ) -> Any:
        """
        Display a returns heatmap visualization.

        Shows returns aggregated by month/year as a heatmap, making it
        easy to identify seasonal patterns and performance trends.

        Args:
            period: Aggregation period - "monthly" (default) or "daily".
            save: Save the plot to a file. Supports .html, .png, .pdf, .svg.
            title: Custom title for the plot.
            height: Custom height in pixels (Plotly only).
            theme: Color theme - "light" or "dark" (Plotly only).

        Returns:
            Plotly Figure object in Jupyter with plotly, ASCII string otherwise.

        Example:
            >>> results = mt.backtest(data, signal)
            >>> results.plot_returns()  # Monthly returns heatmap
        """
        if _has_plotly():
            fig = self._plot_returns_plotly(
                period=period, title=title, height=height, theme=theme
            )
            if save:
                return self._save_plot(fig, save)
            return fig
        else:
            # ASCII fallback
            return f"Returns plot requires plotly. Install with: pip install plotly"

    def _plot_returns_plotly(
        self,
        period: str = "monthly",
        title: Optional[str] = None,
        height: Optional[int] = None,
        theme: Optional[str] = None,
    ) -> Any:
        """Create a returns heatmap Plotly figure."""
        import plotly.graph_objects as go
        import datetime
        from collections import defaultdict

        equity = self.equity_curve
        timestamps = self.equity_timestamps

        # Calculate returns
        returns = []
        for i in range(1, len(equity)):
            ret = (equity[i] - equity[i-1]) / equity[i-1] if equity[i-1] > 0 else 0
            returns.append((timestamps[i], ret))

        # Aggregate by month/year
        if period == "monthly":
            monthly_returns = defaultdict(float)
            for ts, ret in returns:
                dt = datetime.datetime.fromtimestamp(ts)
                key = (dt.year, dt.month)
                monthly_returns[key] += ret

            # Build heatmap data
            if not monthly_returns:
                return go.Figure()

            years = sorted(set(k[0] for k in monthly_returns.keys()))
            months = list(range(1, 13))
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

            z_matrix = []
            for year in years:
                row = []
                for month in months:
                    key = (year, month)
                    if key in monthly_returns:
                        row.append(monthly_returns[key] * 100)  # Convert to percentage
                    else:
                        row.append(None)
                z_matrix.append(row)

            # Theme colors
            if theme == "dark":
                bg_color = '#1e1e1e'
                paper_color = '#1e1e1e'
                text_color = '#e0e0e0'
            else:
                bg_color = 'white'
                paper_color = 'white'
                text_color = '#333333'

            plot_title = title or f"Monthly Returns: {', '.join(self.symbols)}"
            fig_height = height or max(300, len(years) * 40 + 100)

            fig = go.Figure(data=go.Heatmap(
                z=z_matrix,
                x=month_names,
                y=[str(y) for y in years],
                colorscale='RdYlGn',
                zmid=0,
                colorbar=dict(title='Return %'),
                hovertemplate='%{y} %{x}<br>Return: %{z:.1f}%<extra></extra>'
            ))

            fig.update_layout(
                title=dict(text=plot_title, x=0.5, xanchor='center'),
                height=fig_height,
                margin=dict(l=60, r=40, t=60, b=40),
                xaxis_title="Month",
                yaxis_title="Year",
                plot_bgcolor=bg_color,
                paper_bgcolor=paper_color,
                font=dict(color=text_color),
            )

            return fig
        else:
            # Daily returns distribution
            daily_rets = [r * 100 for _, r in returns]

            if theme == "dark":
                bg_color = '#1e1e1e'
                paper_color = '#1e1e1e'
                text_color = '#e0e0e0'
                bar_color = '#4ecdc4'
            else:
                bg_color = 'white'
                paper_color = 'white'
                text_color = '#333333'
                bar_color = '#2E86AB'

            plot_title = title or f"Daily Returns Distribution: {', '.join(self.symbols)}"
            fig_height = height or 400

            fig = go.Figure(data=go.Histogram(
                x=daily_rets,
                nbinsx=50,
                name='Daily Returns',
                marker_color=bar_color,
                hovertemplate='Return: %{x:.2f}%<br>Count: %{y}<extra></extra>'
            ))

            fig.update_layout(
                title=dict(text=plot_title, x=0.5, xanchor='center'),
                height=fig_height,
                margin=dict(l=60, r=40, t=60, b=40),
                xaxis_title="Return (%)",
                yaxis_title="Frequency",
                plot_bgcolor=bg_color,
                paper_bgcolor=paper_color,
                font=dict(color=text_color),
            )

            return fig

    def plot_trades(
        self,
        save: Optional[str] = None,
        title: Optional[str] = None,
        height: Optional[int] = None,
        theme: Optional[str] = None,
    ) -> Any:
        """
        Display a trade analysis visualization.

        Shows trade entry/exit points on the equity curve with
        P&L information for each trade.

        Args:
            save: Save the plot to a file. Supports .html, .png, .pdf, .svg.
            title: Custom title for the plot.
            height: Custom height in pixels (Plotly only).
            theme: Color theme - "light" or "dark" (Plotly only).

        Returns:
            Plotly Figure object in Jupyter with plotly, ASCII string otherwise.

        Example:
            >>> results = mt.backtest(data, signal)
            >>> results.plot_trades()  # Equity curve with trade markers
        """
        # This is essentially plot() with trades=True
        return self.plot(
            trades=True,
            show_drawdown=False,
            save=save,
            title=title or f"Trades: {', '.join(self.symbols)}",
            height=height,
            theme=theme,
        )

    def rolling_sharpe(
        self,
        window: int = 252,
        annualization_factor: float = 252.0,
    ) -> np.ndarray:
        """
        Calculate rolling Sharpe ratio over a sliding window.

        Args:
            window: Number of periods for rolling calculation (default: 252 for daily data)
            annualization_factor: Factor to annualize returns (default: 252.0 for daily)

        Returns:
            Numpy array of rolling Sharpe ratios with same length as equity curve.
        """
        return self._rust.rolling_sharpe(window, annualization_factor)

    def rolling_drawdown(self, window: Optional[int] = None) -> np.ndarray:
        """
        Calculate rolling drawdown from peak equity.

        Args:
            window: Optional maximum lookback window for peak (None = all history)

        Returns:
            Numpy array of drawdown values (0 at peaks, negative otherwise).
        """
        return self._rust.rolling_drawdown(window)

    def rolling_max_drawdown(self, window: int = 252) -> np.ndarray:
        """
        Calculate the worst drawdown within each rolling window.

        Args:
            window: Rolling window size in periods (default: 252 for 1 year of daily data)

        Returns:
            Numpy array of worst drawdown values for each window.
        """
        return self._rust.rolling_max_drawdown(window)

    def rolling_volatility(
        self,
        window: int = 21,
        annualization_factor: float = 252.0,
    ) -> np.ndarray:
        """
        Calculate rolling volatility from equity returns.

        Args:
            window: Number of periods for rolling calculation (default: 21 for monthly)
            annualization_factor: Factor to annualize volatility (default: 252.0)

        Returns:
            Numpy array of annualized volatility values.
        """
        return self._rust.rolling_volatility(window, annualization_factor)

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
    stop_loss: Optional[Union[float, str]] = None,
    take_profit: Optional[Union[float, str]] = None,
    allow_short: bool = True,
    fractional: bool = False,
    borrow_cost: float = 0.03,
    max_position: float = 1.0,
    fill_price: str = "next_open",
    benchmark: Optional[Any] = None,
    freq: Optional[str] = None,
    trading_hours_24: Optional[bool] = None,
    max_volume_participation: Optional[float] = None,
    order_type: str = "market",
    limit_offset: float = 0.0,
) -> BacktestResult:
    """
    Run a backtest on historical data with a signal array.

    See module documentation for full details.

    Args:
        stop_loss: Optional stop loss. Can be:
            - float: percentage (e.g., 0.05 for 5%)
            - str: ATR-based (e.g., "2atr" for 2x ATR), trailing (e.g., "5trail")
        take_profit: Optional take profit. Can be:
            - float: percentage (e.g., 0.10 for 10%)
            - str: ATR-based (e.g., "3atr" for 3x ATR), risk-reward (e.g., "2rr")
        fractional: Allow fractional shares (default: False for whole shares).
            Set to True for crypto or fractional brokers.
        benchmark: Optional benchmark data (from load()) for performance comparison.
                   When provided, the result will include alpha, beta, benchmark_return,
                   excess_return, and other benchmark comparison metrics.
        freq: Data frequency override ("1min", "5min", "1h", "1d", etc.). Auto-detected if None.
        trading_hours_24: Whether to use 24/7 trading hours (crypto). Auto-detected if None.
        max_volume_participation: Maximum volume participation rate (e.g., 0.10 = 10%).
                                  Prevents unrealistic fills in illiquid markets. None = no limit.
        order_type: Order type for signal-generated orders. Options:
            - "market" (default): Execute at market price
            - "limit": Place limit orders at offset from close price
        limit_offset: Limit order offset as fraction of close price (e.g., 0.01 = 1%).
            For buys: limit_price = close * (1 - limit_offset) (below close)
            For sells: limit_price = close * (1 + limit_offset) (above close)
            Only used when order_type="limit".
    """
    rust_result = _backtest_raw(
        data, signal, strategy, strategy_params, config,
        commission, slippage, size, cash, stop_loss,
        take_profit, allow_short, fractional, borrow_cost, max_position, fill_price,
        benchmark, freq, trading_hours_24, max_volume_participation,
        order_type, limit_offset
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
    "CompareResult",
    "sweep",
    # Sensitivity analysis
    "sensitivity",
    "cost_sensitivity",
    "linear_range",
    "log_range",
    "discrete_range",
    "centered_range",
    "ParameterRange",
    "SensitivityResult",
    "SensitivitySummary",
    "HeatmapData",
    "Cliff",
    "Plateau",
    "CostSensitivityResult",
    "CostScenario",
]


class CompareResult:
    """
    Result of comparing multiple backtest strategies.

    In Jupyter notebooks with plotly installed, automatically displays an
    interactive equity curve chart with all strategies overlaid. Also provides
    access to comparison metrics as a dictionary.

    Attributes:
        metrics: Dictionary of strategy names to their metrics
        results: List of BacktestResult objects
        names: List of strategy names
    """

    def __init__(
        self,
        results: List[BacktestResult],
        names: List[str],
        metrics: Dict[str, Any],
    ):
        self._results = results
        self._names = names
        self._metrics = metrics

    @property
    def metrics(self) -> Dict[str, Any]:
        """Get comparison metrics as a dictionary."""
        return self._metrics

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return self._metrics

    def plot(
        self,
        title: Optional[str] = None,
        height: Optional[int] = None,
        theme: Optional[str] = None,
        save: Optional[str] = None,
    ) -> Any:
        """
        Display a visualization comparing all strategies.

        In Jupyter notebooks with plotly installed, returns an interactive
        Plotly figure with equity curves for all strategies overlaid.
        In terminal or without plotly, returns an ASCII summary string.

        Args:
            title: Chart title (default: "Strategy Comparison")
            height: Chart height in pixels (default: 500)
            theme: "light" or "dark" theme
            save: Optional path to save chart (HTML, PNG, PDF supported)

        Returns:
            Plotly Figure object in Jupyter with plotly, string otherwise.

        Example:
            >>> comparison = mt.compare([lstm, baseline], names=["LSTM", "Baseline"])
            >>> comparison.plot()  # Interactive chart in Jupyter
            >>> comparison.plot(save="comparison.html")  # Save to file
        """
        if _has_plotly():
            fig = self._plot_plotly(title=title, height=height, theme=theme)
            if save:
                return self._save_plot(fig, save)
            return fig
        else:
            return self._ascii_summary()

    def _plot_plotly(
        self,
        title: Optional[str] = None,
        height: Optional[int] = None,
        theme: Optional[str] = None,
    ) -> Any:
        """Create an interactive Plotly figure comparing strategies."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import datetime

        # Create figure with subplots: equity curves and metrics table
        fig = make_subplots(
            rows=2,
            cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=("Equity Curves", "Performance Metrics"),
            vertical_spacing=0.15,
        )

        # Color palette for different strategies
        colors = [
            "#1f77b4",  # blue
            "#ff7f0e",  # orange
            "#2ca02c",  # green
            "#d62728",  # red
            "#9467bd",  # purple
            "#8c564b",  # brown
            "#e377c2",  # pink
            "#7f7f7f",  # gray
        ]

        # Plot equity curves for each strategy
        for i, (result, name) in enumerate(zip(self._results, self._names)):
            equity = result.equity_curve
            timestamps = result.equity_timestamps
            dates = [datetime.datetime.fromtimestamp(ts) for ts in timestamps]
            color = colors[i % len(colors)]

            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=equity,
                    mode="lines",
                    name=name,
                    line=dict(color=color, width=2),
                    hovertemplate=f"{name}<br>Date: %{{x}}<br>Equity: $%{{y:,.0f}}<extra></extra>",
                ),
                row=1,
                col=1,
            )

        # Create metrics table
        headers = ["Strategy", "Return", "Sharpe", "Max DD", "Win Rate", "Trades"]
        cells = [[], [], [], [], [], []]

        for name in self._names:
            m = self._metrics[name]
            cells[0].append(name)
            cells[1].append(f"{m['total_return']*100:+.1f}%")
            cells[2].append(f"{m['sharpe']:.2f}")
            cells[3].append(f"{m['max_drawdown']*100:.1f}%")
            cells[4].append(f"{m['win_rate']*100:.1f}%")
            cells[5].append(str(m["total_trades"]))

        fig.add_trace(
            go.Table(
                header=dict(
                    values=headers,
                    fill_color="paleturquoise",
                    align="center",
                    font=dict(size=12, color="black"),
                ),
                cells=dict(
                    values=cells,
                    fill_color="lavender",
                    align="center",
                    font=dict(size=11),
                ),
            ),
            row=2,
            col=1,
        )

        # Theme settings
        if theme == "dark":
            template = "plotly_dark"
            paper_bgcolor = "#1e1e1e"
            plot_bgcolor = "#1e1e1e"
        else:
            template = "plotly_white"
            paper_bgcolor = "white"
            plot_bgcolor = "white"

        fig_title = title or "Strategy Comparison"
        fig_height = height or 600

        fig.update_layout(
            title=fig_title,
            height=fig_height,
            template=template,
            paper_bgcolor=paper_bgcolor,
            plot_bgcolor=plot_bgcolor,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
            ),
            hovermode="x unified",
        )

        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Equity ($)", row=1, col=1)

        return fig

    def _save_plot(self, fig: Any, path: str) -> str:
        """Save Plotly figure to file."""
        import os

        ext = os.path.splitext(path)[1].lower()

        if ext == ".html":
            fig.write_html(path, include_plotlyjs="cdn")
        elif ext in (".png", ".jpg", ".jpeg", ".webp", ".svg", ".pdf"):
            try:
                fig.write_image(path)
            except ValueError as e:
                if "kaleido" in str(e).lower():
                    raise ImportError(
                        f"Saving to {ext} requires kaleido. Install with: pip install kaleido"
                    ) from e
                raise
        else:
            raise ValueError(
                f"Unsupported file format: {ext}. Use .html, .png, .jpg, .svg, or .pdf"
            )

        return path

    def _ascii_summary(self) -> str:
        """Generate ASCII summary for terminal display."""
        lines = ["Strategy Comparison", "=" * 60]
        header = f"{'Strategy':<15} {'Return':>10} {'Sharpe':>8} {'Max DD':>8} {'Win Rate':>10}"
        lines.append(header)
        lines.append("-" * 60)

        for name in self._names:
            m = self._metrics[name]
            row = (
                f"{name:<15} "
                f"{m['total_return']*100:>+9.1f}% "
                f"{m['sharpe']:>8.2f} "
                f"{m['max_drawdown']*100:>7.1f}% "
                f"{m['win_rate']*100:>9.1f}%"
            )
            lines.append(row)

        return "\n".join(lines)

    def _repr_html_(self) -> str:
        """Rich HTML display for Jupyter notebooks."""
        if _is_jupyter() and _has_plotly():
            fig = self._plot_plotly()
            return fig.to_html(include_plotlyjs="cdn", full_html=False)
        else:
            # Return HTML table
            rows = []
            for name in self._names:
                m = self._metrics[name]
                rows.append(
                    f"<tr><td>{name}</td>"
                    f"<td>{m['total_return']*100:+.1f}%</td>"
                    f"<td>{m['sharpe']:.2f}</td>"
                    f"<td>{m['max_drawdown']*100:.1f}%</td>"
                    f"<td>{m['win_rate']*100:.1f}%</td>"
                    f"<td>{m['total_trades']}</td></tr>"
                )
            return f"""
            <div style="font-family: monospace; padding: 10px;">
                <h3>Strategy Comparison</h3>
                <table border="1" style="border-collapse: collapse;">
                    <tr>
                        <th>Strategy</th>
                        <th>Return</th>
                        <th>Sharpe</th>
                        <th>Max DD</th>
                        <th>Win Rate</th>
                        <th>Trades</th>
                    </tr>
                    {"".join(rows)}
                </table>
            </div>
            """

    def __repr__(self) -> str:
        return self._ascii_summary()

    def __str__(self) -> str:
        return self._ascii_summary()


def compare(
    results: List[BacktestResult],
    names: Optional[List[str]] = None,
) -> CompareResult:
    """
    Compare multiple backtest results side by side.

    In Jupyter notebooks with plotly installed, automatically displays an
    interactive equity curve chart with all strategies overlaid.

    Args:
        results: List of BacktestResult objects to compare
        names: Optional list of names for each result

    Returns:
        CompareResult object with metrics dict and plot() method.
        In Jupyter, automatically renders as interactive chart.

    Example:
        >>> lstm_results = mt.backtest(data, lstm_signal)
        >>> baseline = mt.backtest(data, baseline_signal)
        >>> comparison = mt.compare([lstm_results, baseline], names=["LSTM", "Baseline"])
        >>> comparison.metrics  # Get dict of metrics
        >>> comparison.plot()  # Show interactive chart
        >>> comparison.plot(save="comparison.html")  # Save to file
    """
    if names is None:
        names = [f"Strategy {i+1}" for i in range(len(results))]

    metrics = {}
    for name, result in zip(names, results):
        metrics[name] = {
            "total_return": result.total_return,
            "sharpe": result.sharpe,
            "max_drawdown": result.max_drawdown,
            "win_rate": result.win_rate,
            "total_trades": result.total_trades,
            "profit_factor": result.profit_factor,
        }

    return CompareResult(results, names, metrics)


def sweep(
    data: Any,
    signal_fn: callable,
    params: Dict[str, List[Any]],
    n_jobs: int = -1,
    parallel: bool = True,
    **backtest_kwargs,
) -> "SweepResult":
    """
    Run a parameter sweep, testing multiple parameter combinations.

    Uses Rust's rayon for parallel execution across all CPU cores.

    Args:
        data: Data dictionary from load() or file path
        signal_fn: Function that takes params and returns a signal array
        params: Dictionary of parameter names to lists of values to test
        n_jobs: Number of parallel jobs (-1 for all cores). Deprecated, use `parallel` instead.
        parallel: Whether to use parallel execution (default True)
        **backtest_kwargs: Additional arguments passed to backtest()

    Returns:
        SweepResult object with results for each parameter combination.
        Use .best() to get the best result, .to_dict() for dict format.

    Example:
        >>> def generate_signal(threshold):
        ...     return (predictions > threshold).astype(int)
        >>>
        >>> sweep_results = mt.sweep(
        ...     data,
        ...     generate_signal,
        ...     params={"threshold": [0.1, 0.2, 0.3, 0.4, 0.5]}
        ... )
        >>> best = sweep_results.best()  # Get best by Sharpe
        >>> print(best.params, best.result.sharpe)
    """
    from itertools import product

    # Generate all parameter combinations and pre-compute signals
    param_names = list(params.keys())
    param_values = list(params.values())
    combinations = list(product(*param_values))

    # Pre-compute all signals in Python (signal_fn can't be called from Rust)
    signals_list = []
    for combo in combinations:
        param_dict = dict(zip(param_names, (float(v) for v in combo)))

        # Generate signal with these parameters
        if len(param_names) == 1:
            signal = signal_fn(combo[0])
        else:
            signal = signal_fn(**dict(zip(param_names, combo)))

        # Convert to numpy array and then to list for Rust
        if hasattr(signal, 'tolist'):
            signal = signal.tolist()
        elif hasattr(signal, 'to_list'):
            signal = signal.to_list()

        signals_list.append((param_dict, signal))

    # Determine parallelism - use parallel flag, n_jobs is for backwards compat
    use_parallel = parallel and (n_jobs != 0)

    # Extract backtest config from kwargs
    commission = backtest_kwargs.pop('commission', 0.001)
    slippage = backtest_kwargs.pop('slippage', 0.001)
    size = backtest_kwargs.pop('size', 0.10)
    cash = backtest_kwargs.pop('cash', 100_000.0)
    stop_loss = backtest_kwargs.pop('stop_loss', None)
    take_profit = backtest_kwargs.pop('take_profit', None)
    allow_short = backtest_kwargs.pop('allow_short', True)
    borrow_cost = backtest_kwargs.pop('borrow_cost', 0.03)
    max_position = backtest_kwargs.pop('max_position', 1.0)
    fill_price = backtest_kwargs.pop('fill_price', 'next_open')

    # Call the Rust parallel sweep function
    rust_result = _sweep_raw(
        data,
        signals_list,
        parallel=use_parallel,
        commission=commission,
        slippage=slippage,
        size=size,
        cash=cash,
        stop_loss=stop_loss,
        take_profit=take_profit,
        allow_short=allow_short,
        borrow_cost=borrow_cost,
        max_position=max_position,
        fill_price=fill_price,
    )

    # Wrap in Python SweepResult for enhanced functionality
    return SweepResult(rust_result)


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


class SweepResult:
    """
    Result of a parallel parameter sweep.

    Wraps the Rust SweepResult with additional Python functionality
    including Plotly visualization support in Jupyter.

    Attributes:
        num_combinations: Total number of parameter combinations tested
        parallel: Whether parallel execution was used

    Example:
        >>> results = mt.sweep(data, signal_fn, params={"threshold": [0.1, 0.2, 0.3]})
        >>> best = results.best()  # Get best by Sharpe
        >>> print(best.params, best.result.sharpe)
        >>> results.plot()  # Interactive heatmap in Jupyter
    """

    def __init__(self, rust_result: _SweepResult):
        """Initialize from a Rust SweepResult."""
        self._inner = rust_result

    @property
    def num_combinations(self) -> int:
        """Total number of parameter combinations tested."""
        return self._inner.num_combinations

    @property
    def parallel(self) -> bool:
        """Whether parallel execution was used."""
        return self._inner.parallel

    def items(self) -> List[SweepResultItem]:
        """Get all results as a list of SweepResultItem objects."""
        return self._inner.items()

    def to_dict(self) -> Dict[str, "BacktestResult"]:
        """Get all results as a dictionary mapping param strings to BacktestResult."""
        result = self._inner.to_dict()
        # Wrap results in BacktestResult class
        return {k: BacktestResult(v) if hasattr(v, 'sharpe') else v for k, v in result.items()}

    def best(self, metric: str = "sharpe", maximize: bool = True) -> Optional[SweepResultItem]:
        """
        Get the best result by a given metric.

        Args:
            metric: Metric to optimize ("sharpe", "sortino", "return", "calmar", "profit_factor")
            maximize: Whether to maximize (default True) or minimize the metric

        Returns:
            The SweepResultItem with the best metric value, or None if empty.
        """
        return self._inner.best(metric, maximize)

    def best_params(self, metric: str = "sharpe", maximize: bool = True) -> Optional[Dict[str, float]]:
        """
        Get the best parameters by a given metric.

        Args:
            metric: Metric to optimize
            maximize: Whether to maximize or minimize

        Returns:
            Dictionary of parameter values for the best combination.
        """
        return self._inner.best_params(metric, maximize)

    def sorted_by(self, metric: str = "sharpe", descending: bool = True) -> List[SweepResultItem]:
        """Get results sorted by a metric."""
        return self._inner.sorted_by(metric, descending)

    def top(self, n: int = 10, metric: str = "sharpe") -> List[SweepResultItem]:
        """Get top N results by a metric."""
        return self._inner.top(n, metric)

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics across all parameter combinations."""
        return self._inner.summary()

    def plot(self, x_param: Optional[str] = None, y_param: Optional[str] = None, metric: str = "sharpe"):
        """
        Plot sweep results.

        In Jupyter with plotly installed, shows an interactive heatmap.
        Otherwise, prints a summary table.

        Args:
            x_param: Parameter for X axis (auto-detected if not provided)
            y_param: Parameter for Y axis (auto-detected if not provided)
            metric: Metric to visualize (default "sharpe")

        Returns:
            Plotly Figure in Jupyter, None otherwise.
        """
        if _is_jupyter() and _has_plotly():
            return self._plot_plotly(x_param, y_param, metric)
        else:
            # ASCII fallback - print summary
            summary = self.summary()
            best = self.best(metric)
            print(f"Sweep Results: {summary['num_combinations']} combinations")
            print(f"  Parallel: {summary['parallel']}")
            print(f"  Sharpe range: {summary['sharpe_min']:.2f} to {summary['sharpe_max']:.2f}")
            print(f"  Return range: {summary['return_min']*100:.1f}% to {summary['return_max']*100:.1f}%")
            if best:
                print(f"  Best params: {best.params}")
                print(f"  Best sharpe: {best.result.sharpe:.4f}")
            return None

    def _plot_plotly(self, x_param: Optional[str], y_param: Optional[str], metric: str):
        """Create interactive Plotly visualization."""
        import plotly.graph_objects as go

        items = self.items()
        if not items:
            return None

        # Extract parameter names
        param_names = list(items[0].params.keys())

        if len(param_names) == 1:
            # 1D plot - line chart
            x_param = param_names[0]
            x_vals = [item.params[x_param] for item in items]
            y_vals = [getattr(item.result, metric, item.result.sharpe) for item in items]

            fig = go.Figure(data=go.Scatter(x=x_vals, y=y_vals, mode='lines+markers'))
            fig.update_layout(
                title=f"Parameter Sweep: {metric.title()} vs {x_param}",
                xaxis_title=x_param,
                yaxis_title=metric.title(),
            )
            return fig

        elif len(param_names) >= 2:
            # 2D plot - heatmap
            if x_param is None:
                x_param = param_names[0]
            if y_param is None:
                y_param = param_names[1]

            # Get unique values for each param
            x_vals = sorted(set(item.params[x_param] for item in items))
            y_vals = sorted(set(item.params[y_param] for item in items))

            # Build heatmap matrix
            z_matrix = [[None for _ in x_vals] for _ in y_vals]
            for item in items:
                try:
                    xi = x_vals.index(item.params[x_param])
                    yi = y_vals.index(item.params[y_param])
                    z_matrix[yi][xi] = getattr(item.result, metric, item.result.sharpe)
                except (ValueError, AttributeError):
                    pass

            fig = go.Figure(data=go.Heatmap(
                z=z_matrix,
                x=[str(v) for v in x_vals],
                y=[str(v) for v in y_vals],
                colorscale='RdYlGn',
                colorbar=dict(title=metric.title()),
            ))
            fig.update_layout(
                title=f"Parameter Sweep: {metric.title()}",
                xaxis_title=x_param,
                yaxis_title=y_param,
            )
            return fig

        return None

    def __len__(self) -> int:
        return self.num_combinations

    def __repr__(self) -> str:
        return repr(self._inner)

    def _repr_html_(self) -> str:
        """Rich display for Jupyter notebooks."""
        summary = self.summary()
        best = self.best()
        best_info = f"Best Sharpe: {best.result.sharpe:.4f}" if best else "No results"
        return f"""
        <div style="font-family: monospace;">
            <strong>SweepResult</strong><br>
            Combinations: {summary['num_combinations']}<br>
            Parallel: {summary['parallel']}<br>
            Sharpe range: {summary['sharpe_min']:.2f} - {summary['sharpe_max']:.2f}<br>
            {best_info}
        </div>
        """


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
            "max_volume_participation": None,
            "order_type": "market",
            "limit_offset": 0.0,
            "max_leverage": 2.0,
            "target_vol": None,
            "vol_lookback": None,
            "base_size": None,
            "risk_per_trade": None,
            "stop_atr": None,
            "atr_period": None,
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

    def commission_per_share(self, rate: float) -> "Backtest":
        """
        Set the per-share commission rate.

        Use this with commission=0 for per-share pricing models.

        Args:
            rate: Commission per share (e.g., 0.005 = $0.005/share)

        Returns:
            Self for method chaining

        Example:
            >>> # Per-share commission of $0.005/share
            >>> results = mt.Backtest(data, signal).commission(0).commission_per_share(0.005).run()
        """
        self._config["commission_per_share"] = rate
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

    def size(self, fraction: Union[float, str]) -> "Backtest":
        """
        Set the position sizing method.

        Args:
            fraction: Position size specification. Can be:
                - float: percentage of equity (e.g., 0.10 = 10%)
                - "volatility": volatility-targeted sizing
                - "signal": signal-scaled sizing
                - "risk": risk-based sizing with ATR

        Returns:
            Self for method chaining

        Example:
            >>> mt.Backtest(data, signal).size(0.10)           # 10% of equity
            >>> mt.Backtest(data, signal).size("volatility")   # Volatility-targeted
        """
        self._config["size"] = fraction
        return self

    def max_leverage(self, leverage: float) -> "Backtest":
        """
        Set the maximum leverage allowed.

        Args:
            leverage: Maximum leverage (default 2.0)

        Returns:
            Self for method chaining
        """
        self._config["max_leverage"] = leverage
        return self

    def target_vol(self, target: float) -> "Backtest":
        """
        Set target volatility for volatility-targeted sizing.

        Only used when size="volatility".

        Args:
            target: Target annualized volatility (e.g., 0.15 for 15%)

        Returns:
            Self for method chaining
        """
        self._config["target_vol"] = target
        return self

    def vol_lookback(self, lookback: int) -> "Backtest":
        """
        Set lookback period for volatility calculation.

        Only used when size="volatility".

        Args:
            lookback: Lookback period in bars (default 20)

        Returns:
            Self for method chaining
        """
        self._config["vol_lookback"] = lookback
        return self

    def base_size(self, size: float) -> "Backtest":
        """
        Set base position size for signal-scaled sizing.

        Only used when size="signal".

        Args:
            size: Base position size as fraction of equity (default 0.10)

        Returns:
            Self for method chaining
        """
        self._config["base_size"] = size
        return self

    def risk_per_trade(self, risk: float) -> "Backtest":
        """
        Set risk per trade for risk-based sizing.

        Only used when size="risk".

        Args:
            risk: Risk per trade as fraction of equity (e.g., 0.01 for 1%)

        Returns:
            Self for method chaining
        """
        self._config["risk_per_trade"] = risk
        return self

    def stop_atr(self, multiplier: float) -> "Backtest":
        """
        Set stop loss distance in ATR multiples for risk-based sizing.

        Only used when size="risk".

        Args:
            multiplier: ATR multiplier for stop distance (e.g., 2.0 for 2x ATR)

        Returns:
            Self for method chaining
        """
        self._config["stop_atr"] = multiplier
        return self

    def atr_period(self, period: int) -> "Backtest":
        """
        Set ATR period for risk-based sizing.

        Only used when size="risk".

        Args:
            period: ATR period in bars (default 14)

        Returns:
            Self for method chaining
        """
        self._config["atr_period"] = period
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

    def stop_loss(self, value: Union[float, str]) -> "Backtest":
        """
        Set a stop loss.

        Args:
            value: Stop loss specification. Can be:
                - float: percentage (e.g., 0.05 for 5%)
                - str: ATR-based (e.g., "2atr" for 2x ATR)
                - str: trailing (e.g., "5trail" for 5% trailing)

        Returns:
            Self for method chaining

        Example:
            >>> mt.Backtest(data, signal).stop_loss(0.05)  # 5% stop
            >>> mt.Backtest(data, signal).stop_loss("2atr")  # 2x ATR stop
        """
        self._config["stop_loss"] = value
        return self

    def take_profit(self, value: Union[float, str]) -> "Backtest":
        """
        Set a take profit.

        Args:
            value: Take profit specification. Can be:
                - float: percentage (e.g., 0.10 for 10%)
                - str: ATR-based (e.g., "3atr" for 3x ATR)
                - str: risk-reward (e.g., "2rr" for 2:1 R:R ratio)

        Returns:
            Self for method chaining

        Example:
            >>> mt.Backtest(data, signal).take_profit(0.10)  # 10% target
            >>> mt.Backtest(data, signal).take_profit("3atr")  # 3x ATR target
            >>> mt.Backtest(data, signal).take_profit("2rr")  # 2:1 risk-reward
        """
        self._config["take_profit"] = value
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

    def benchmark(self, data: Any) -> "Backtest":
        """
        Set benchmark data for performance comparison.

        When benchmark is provided, the result will include alpha, beta,
        benchmark_return, excess_return, and other comparison metrics.

        Args:
            data: Benchmark data dictionary from load() (e.g., SPY data)

        Returns:
            Self for method chaining

        Example:
            >>> spy = mt.load("SPY.csv")
            >>> results = mt.Backtest(data, signal).benchmark(spy).run()
            >>> print(results.alpha, results.beta)
            0.05 1.2
        """
        self._config["benchmark"] = data
        return self

    def freq(self, frequency: str) -> "Backtest":
        """
        Override the data frequency for metric annualization.

        By default, data frequency is auto-detected from bar timestamps.
        Use this method to explicitly specify the frequency when auto-detection
        fails or when you want to override the detected value.

        Args:
            frequency: Data frequency. Options:
                - "1s", "5s", "10s", "15s", "30s": Second frequencies
                - "1min", "5min", "15min", "30min": Minute frequencies
                - "1h", "4h": Hourly frequencies
                - "1d" or "daily": Daily frequency
                - "1w" or "weekly": Weekly frequency
                - "1mo" or "monthly": Monthly frequency

        Returns:
            Self for method chaining

        Example:
            >>> # 5-minute data
            >>> results = mt.Backtest(data, signal).freq("5min").run()
        """
        self._config["freq"] = frequency
        return self

    def trading_hours_24(self, enabled: bool = True) -> "Backtest":
        """
        Enable 24/7 trading hours for metric annualization (crypto markets).

        By default, the system auto-detects whether data is from a 24/7 market
        by checking for weekend bars. Use this method to explicitly specify
        the market type.

        When enabled (True), metrics are annualized using 365 days/year.
        When disabled (False), metrics use 252 trading days/year (traditional markets).

        Args:
            enabled: Whether to use 24/7 market annualization (default True)

        Returns:
            Self for method chaining

        Example:
            >>> # Bitcoin backtest with proper annualization
            >>> results = mt.Backtest(btc_data, signal).trading_hours_24().run()
        """
        self._config["trading_hours_24"] = enabled
        return self

    def max_volume_participation(self, rate: float) -> "Backtest":
        """
        Set the maximum volume participation rate.

        This limits the maximum order size to a fraction of the bar's volume,
        preventing unrealistic fills in illiquid markets. For example, setting
        this to 0.10 means orders cannot exceed 10% of the bar's total volume.

        Args:
            rate: Maximum participation rate as a fraction (e.g., 0.10 = 10%)

        Returns:
            Self for method chaining

        Example:
            >>> # Limit fills to 5% of bar volume
            >>> results = mt.Backtest(data, signal).max_volume_participation(0.05).run()
        """
        self._config["max_volume_participation"] = rate
        return self

    def order_type(self, order_type: str) -> "Backtest":
        """
        Set the order type for signal-generated orders.

        By default, signals are converted to market orders. Use this method
        to place limit orders instead, which can provide more realistic
        execution simulation for passive strategies.

        Args:
            order_type: Order type. Options:
                - "market" (default): Execute at market price
                - "limit": Place limit orders at offset from close price

        Returns:
            Self for method chaining

        Example:
            >>> # Use limit orders with 0.5% offset
            >>> results = (
            ...     mt.Backtest(data, signal)
            ...     .order_type("limit")
            ...     .limit_offset(0.005)
            ...     .run()
            ... )
        """
        self._config["order_type"] = order_type
        return self

    def limit_offset(self, offset: float) -> "Backtest":
        """
        Set the limit order offset from close price.

        Only used when order_type="limit". The offset determines how far
        from the close price the limit order is placed.

        For buy orders: limit_price = close * (1 - offset) (below close)
        For sell orders: limit_price = close * (1 + offset) (above close)

        Args:
            offset: Offset as a fraction of close price (e.g., 0.01 = 1%)

        Returns:
            Self for method chaining

        Example:
            >>> # Limit orders 1% away from close
            >>> results = (
            ...     mt.Backtest(data, signal)
            ...     .order_type("limit")
            ...     .limit_offset(0.01)
            ...     .run()
            ... )
        """
        self._config["limit_offset"] = offset
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


# =============================================================================
# Sensitivity Analysis Wrapper Classes
# =============================================================================

class SensitivityResult:
    """
    Wrapper around the Rust SensitivityResult that adds Plotly visualization support.

    In Jupyter notebooks with plotly installed, heatmap plotting returns an interactive
    Plotly figure. In other environments, it falls back to ASCII visualization.
    """

    def __init__(self, rust_result: _SensitivityResult):
        self._rust = rust_result

    @property
    def strategy_name(self) -> str:
        return self._rust.strategy_name

    @property
    def symbol(self) -> str:
        return self._rust.symbol

    @property
    def num_combinations(self) -> int:
        return self._rust.num_combinations

    def best_params(self) -> Optional[Dict[str, float]]:
        """Get the best performing parameter set."""
        return self._rust.best_params()

    def stability_score(self) -> float:
        """Get the overall stability score (0-1, higher is more stable)."""
        return self._rust.stability_score()

    def parameter_stability(self, param_name: str) -> Optional[float]:
        """Get stability score for a specific parameter."""
        return self._rust.parameter_stability(param_name)

    def is_fragile(self, threshold: float = 0.5) -> bool:
        """Check if the strategy is fragile (high sensitivity to parameters)."""
        return self._rust.is_fragile(threshold)

    def heatmap(self, x_param: str, y_param: str) -> Optional[HeatmapData]:
        """Get 2D heatmap data for two parameters."""
        return self._rust.heatmap(x_param, y_param)

    def parameter_importance(self) -> List[tuple]:
        """Get parameter importance ranking."""
        return self._rust.parameter_importance()

    def cliffs(self) -> List[Cliff]:
        """Get detected cliffs (sharp performance drops)."""
        return self._rust.cliffs()

    def plateaus(self) -> List[Plateau]:
        """Get detected plateaus (stable performance regions)."""
        return self._rust.plateaus()

    def summary(self) -> SensitivitySummary:
        """Get summary statistics."""
        return self._rust.summary()

    def to_csv(self) -> str:
        """Export results to CSV format."""
        return self._rust.to_csv()

    def plot_heatmap(self, x_param: str, y_param: str) -> Any:
        """
        Display a heatmap visualization for two parameters.

        In Jupyter notebooks with plotly installed, returns an interactive
        Plotly heatmap. Otherwise returns ASCII representation.

        Args:
            x_param: Parameter name for X axis.
            y_param: Parameter name for Y axis.

        Returns:
            Plotly Figure object in Jupyter with plotly, string otherwise.
        """
        heatmap = self.heatmap(x_param, y_param)
        if heatmap is None:
            raise ValueError(f"Parameters '{x_param}' and/or '{y_param}' not found.")

        if _is_jupyter() and _has_plotly():
            return self._plot_heatmap_plotly(heatmap, x_param, y_param)
        else:
            return heatmap.to_csv()

    def _plot_heatmap_plotly(self, heatmap: HeatmapData, x_param: str, y_param: str) -> Any:
        """Create an interactive Plotly heatmap."""
        import plotly.graph_objects as go

        x_vals = list(heatmap.x_values())
        y_vals = list(heatmap.y_values())
        z_vals = heatmap.values()

        fig = go.Figure(data=go.Heatmap(
            z=z_vals,
            x=x_vals,
            y=y_vals,
            colorscale='RdYlGn',
            hovertemplate=f'{x_param}: %{{x}}<br>{y_param}: %{{y}}<br>Metric: %{{z:.3f}}<extra></extra>'
        ))

        fig.update_layout(
            title=dict(
                text=f"Parameter Sensitivity: {self.strategy_name}",
                x=0.5,
                xanchor='center'
            ),
            xaxis_title=x_param,
            yaxis_title=y_param,
            height=500,
            margin=dict(l=60, r=40, t=60, b=60)
        )

        return fig

    def __repr__(self) -> str:
        return repr(self._rust)

    def __str__(self) -> str:
        return str(self._rust)


class CostSensitivityResult:
    """
    Wrapper around the Rust CostSensitivityResult that adds Plotly visualization support.

    In Jupyter notebooks with plotly installed, plot() returns an interactive
    Plotly figure. In other environments, it falls back to text report.
    """

    def __init__(self, rust_result: _CostSensitivityResult):
        self._rust = rust_result

    @property
    def symbol(self) -> str:
        return self._rust.symbol

    @property
    def strategy_name(self) -> str:
        return self._rust.strategy_name

    def scenarios(self) -> List[CostScenario]:
        """Get all scenarios."""
        return self._rust.scenarios()

    def scenario_at(self, multiplier: float) -> Optional[CostScenario]:
        """Get scenario at specific multiplier."""
        return self._rust.scenario_at(multiplier)

    def baseline(self) -> Optional[CostScenario]:
        """Get baseline (1x) scenario."""
        return self._rust.baseline()

    def zero_cost(self) -> Optional[CostScenario]:
        """Get zero-cost scenario (theoretical upper bound)."""
        return self._rust.zero_cost()

    def sharpe_degradation_at(self, multiplier: float) -> Optional[float]:
        """Calculate Sharpe ratio degradation percentage at given multiplier."""
        return self._rust.sharpe_degradation_at(multiplier)

    def return_degradation_at(self, multiplier: float) -> Optional[float]:
        """Calculate return degradation percentage at given multiplier."""
        return self._rust.return_degradation_at(multiplier)

    def is_robust(self, threshold_sharpe: float = 0.5) -> bool:
        """Check if strategy passes robustness threshold at 5x costs."""
        return self._rust.is_robust(threshold_sharpe)

    def cost_elasticity(self) -> Optional[float]:
        """Calculate cost elasticity (% change in return per % change in costs)."""
        return self._rust.cost_elasticity()

    def breakeven_multiplier(self) -> Optional[float]:
        """Calculate breakeven cost multiplier (where returns become zero/negative)."""
        return self._rust.breakeven_multiplier()

    def report(self) -> str:
        """Generate formatted summary report."""
        return self._rust.report()

    def plot(self) -> Any:
        """
        Display a visualization of cost sensitivity.

        In Jupyter notebooks with plotly installed, returns an interactive
        Plotly figure showing Sharpe and returns at different cost levels.
        Otherwise returns a text report.

        Returns:
            Plotly Figure object in Jupyter with plotly, string otherwise.
        """
        if _is_jupyter() and _has_plotly():
            return self._plot_plotly()
        else:
            return self.report()

    def _plot_plotly(self) -> Any:
        """Create an interactive Plotly figure for cost sensitivity."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        scenarios = self.scenarios()
        multipliers = [s.multiplier for s in scenarios]
        sharpes = [s.sharpe for s in scenarios]
        returns = [s.total_return for s in scenarios]

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Sharpe Ratio vs Cost Multiplier', 'Return vs Cost Multiplier')
        )

        # Sharpe ratio plot
        fig.add_trace(
            go.Scatter(
                x=multipliers,
                y=sharpes,
                mode='lines+markers',
                name='Sharpe',
                line=dict(color='#2E86AB', width=2),
                marker=dict(size=8),
                hovertemplate='%{x}x costs<br>Sharpe: %{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )

        # Return plot
        fig.add_trace(
            go.Scatter(
                x=multipliers,
                y=returns,
                mode='lines+markers',
                name='Return',
                line=dict(color='#28A745', width=2),
                marker=dict(size=8),
                hovertemplate='%{x}x costs<br>Return: %{y:.1f}%<extra></extra>'
            ),
            row=1, col=2
        )

        # Add robustness threshold line
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
                     annotation_text="Robustness threshold", row=1, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray",
                     annotation_text="Breakeven", row=1, col=2)

        robust = self.is_robust()
        verdict_color = '#28A745' if robust else '#DC3545'
        verdict_text = 'ROBUST' if robust else 'FRAGILE'

        fig.update_layout(
            title=dict(
                text=f"Cost Sensitivity: {self.strategy_name} ({self.symbol}) - {verdict_text}",
                x=0.5,
                xanchor='center',
                font=dict(color=verdict_color)
            ),
            showlegend=False,
            height=400,
            margin=dict(l=60, r=40, t=60, b=40)
        )

        fig.update_xaxes(title_text="Cost Multiplier", row=1, col=1)
        fig.update_xaxes(title_text="Cost Multiplier", row=1, col=2)
        fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=1)
        fig.update_yaxes(title_text="Return (%)", row=1, col=2)

        return fig

    def _repr_html_(self) -> str:
        """Rich HTML display for Jupyter notebooks."""
        if _is_jupyter() and _has_plotly():
            fig = self._plot_plotly()
            return fig.to_html(include_plotlyjs='cdn', full_html=False)
        else:
            robust = self.is_robust()
            verdict_color = 'green' if robust else 'red'
            verdict_text = 'ROBUST' if robust else 'FRAGILE'
            return f"""
            <div style="font-family: monospace; padding: 10px;">
                <h3>Cost Sensitivity: {self.strategy_name} ({self.symbol})</h3>
                <p><strong>Verdict:</strong> <span style="color: {verdict_color}; font-weight: bold;">{verdict_text}</span></p>
                <table>
                    <tr><th>Multiplier</th><th>Return</th><th>Sharpe</th></tr>
                    {''.join(f"<tr><td>{s.multiplier}x</td><td>{s.total_return:.1f}%</td><td>{s.sharpe:.2f}</td></tr>" for s in self.scenarios())}
                </table>
            </div>
            """

    def __repr__(self) -> str:
        return repr(self._rust)

    def __str__(self) -> str:
        return str(self._rust)


# =============================================================================
# Sensitivity Analysis Wrapper Functions
# =============================================================================

def sensitivity(
    data: Any,
    strategy: str,
    params: Dict[str, ParameterRange],
    metric: str = "sharpe",
    commission: float = 0.001,
    slippage: float = 0.001,
    cash: float = 100_000.0,
    parallel: bool = True,
) -> SensitivityResult:
    """
    Run parameter sensitivity analysis on a built-in strategy.

    Tests how strategy performance varies across different parameter values.
    This helps identify:
    - Fragile strategies that only work with specific parameters
    - Robust strategies that perform well across parameter ranges
    - Cliffs where performance drops sharply
    - Plateaus where performance is stable

    Args:
        data: Data dictionary from load() or path to CSV/Parquet file
        strategy: Name of built-in strategy ("sma-crossover", "momentum",
                  "mean-reversion", "rsi", "macd", "breakout")
        params: Dictionary mapping parameter names to ParameterRange objects
        metric: Metric to analyze ("sharpe", "sortino", "return", "calmar",
                "profit_factor", "win_rate", "max_drawdown")
        commission: Commission rate (default 0.001 = 0.1%)
        slippage: Slippage rate (default 0.001 = 0.1%)
        cash: Initial capital (default 100,000)
        parallel: Run parameter combinations in parallel (default True)

    Returns:
        SensitivityResult with analysis results.

    Example:
        >>> data = mt.load_sample("AAPL")
        >>> result = mt.sensitivity(
        ...     data,
        ...     strategy="sma-crossover",
        ...     params={
        ...         "fast_period": mt.linear_range(5, 20, 4),
        ...         "slow_period": mt.linear_range(20, 60, 5),
        ...     },
        ...     metric="sharpe"
        ... )
        >>> print(result.stability_score())
        0.72
        >>> print(result.best_params())
        {'fast_period': 10.0, 'slow_period': 40.0}
    """
    rust_result = _sensitivity_raw(
        data, strategy, params, metric,
        commission, slippage, cash, parallel
    )
    return SensitivityResult(rust_result)


def cost_sensitivity(
    data: Any,
    signal: Optional[np.ndarray] = None,
    strategy: Optional[str] = None,
    strategy_params: Optional[Dict[str, Any]] = None,
    multipliers: Optional[List[float]] = None,
    commission: float = 0.001,
    slippage: float = 0.001,
    cash: float = 100_000.0,
    include_zero_cost: bool = True,
) -> CostSensitivityResult:
    """
    Run cost sensitivity analysis to test strategy robustness to transaction costs.

    Tests the same strategy at multiple cost levels (e.g., 1x, 2x, 5x, 10x) to see
    how performance degrades. A robust strategy should maintain acceptable returns
    even with significantly higher costs.

    Args:
        data: Data dictionary from load() or path to CSV/Parquet file
        signal: Signal array (1=long, -1=short, 0=flat), or None for built-in strategy
        strategy: Name of built-in strategy if signal is None
        strategy_params: Parameters for built-in strategy
        multipliers: List of cost multipliers to test (default [0.0, 1.0, 2.0, 5.0, 10.0])
        commission: Base commission rate (default 0.001 = 0.1%)
        slippage: Base slippage rate (default 0.001 = 0.1%)
        cash: Initial capital (default 100,000)
        include_zero_cost: Include zero-cost scenario (default True)

    Returns:
        CostSensitivityResult with analysis results.

    Example:
        >>> data = mt.load_sample("AAPL")
        >>> signal = (data['close'] > data['close'].mean()).astype(int)
        >>> result = mt.cost_sensitivity(data, signal)
        >>> print(result.sharpe_degradation_at(5.0))
        45.2
        >>> print(result.is_robust())
        True
    """
    rust_result = _cost_sensitivity_raw(
        data, signal, strategy, strategy_params,
        multipliers, commission, slippage, cash, include_zero_cost
    )
    return CostSensitivityResult(rust_result)
