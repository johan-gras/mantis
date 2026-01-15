## Build & Run

Build: `cargo build --release`
Run CLI: `cargo run -- --help`

## Validation

Run these after implementing to get immediate feedback:

- Tests: `cargo test`
- Lint: `cargo clippy -- -D warnings`
- Format check: `cargo fmt --check`
- Format fix: `cargo fmt`

## Operational Notes

- Project is named `mantis` (Rust crate name)
- CLI binary is also named `mantis`
- Uses chrono for timestamps, serde for serialization
- Tests are in unit (src/), integration (tests/), and doc-tests

### Codebase Patterns

- Strategies implement the `Strategy` trait
- Multi-asset strategies implement `PortfolioStrategy`
- Data loading via `load_csv()` in `src/data.rs`
- Result export via `Exporter` in `src/export.rs`
- CLI `mantis run` supports `--asset-class` (equity/future/crypto/forex/option) plus parameters
  like `--multiplier`, `--tick-size`, `--margin-requirement`, etc., to configure symbol metadata.
- `mantis run`/`mantis walk-forward` also accept `--lot-selection fifo|lifo|highest-cost|lowest-cost` to control default tax-lot consumption (per-order overrides via `Order::with_lot_selection`).
- Execution realism can be tuned via `--execution-price`, `--fill-probability`, and `--limit-order-ttl` flags on `mantis run`.
- Walk-forward optimization is available via `mantis walk-forward -d data.csv --folds 5 --strategy sma-crossover` with optional `--anchored` windows and `--metric profit-factor`.
