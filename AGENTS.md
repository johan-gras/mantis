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
