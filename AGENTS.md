## Build & Run

Build: `cargo build --release`
Run CLI: `cargo run -- --help`
Run with ONNX: `cargo build --release --features onnx`

## Validation

Run these after implementing to get immediate feedback:

- Tests: `cargo test --features onnx`
- Lint: `cargo clippy --features onnx -- -D warnings`
- Format check: `cargo fmt --check`
- Format fix: `cargo fmt`
- Benchmarks: `cargo bench --features onnx`

## Python Tests

Python tests require pytest and the mantis package installed:
```bash
pip install maturin pytest
maturin develop --features onnx
pytest tests/python/
```

## Operational Notes

- The ONNX feature requires `--features onnx` flag for build/test/clippy
- ONNX test models are in `data/models/` and can be regenerated with `python scripts/generate_test_onnx.py`
- Python bindings are built with maturin
- Use `python3` for scripts when `python` is not available (e.g., `python3 scripts/test_doc_examples.py`)
- **cffi conflict**: If cffi is installed, `maturin develop` creates `python/mantis/_mantis/` which shadows the PyO3 module. Fix: `rm -rf python/mantis/_mantis/`

### Codebase Patterns

- Core engine: `src/engine.rs` - backtest execution loop
- Python bindings: `src/python/` - PyO3 wrappers
- Strategies: `src/strategies/` - built-in trading strategies
- Tests: inline in src files + `tests/` directory
