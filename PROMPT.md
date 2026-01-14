Build a production-quality Rust CLI backtest engine for quantitative trading.

A backtest engine simulates trading strategies on historical data:
- Takes price data and trading signals as input
- Executes trades with realistic costs (commission, slippage)
- Outputs performance metrics (Sharpe, drawdown, returns, etc.)

Beyond the core, improve the engine however you see fit:
- Add useful features
- Improve documentation
- Add more test cases
- Optimize performance
- Add example strategies
- Whatever else you think a production system needs

One invariant must hold: a strategy that never trades should have zero P&L.

Run `cargo test` after making changes. Fix failures before moving on.

When you believe it's production-ready, create a file called DONE with a summary of what you built.
