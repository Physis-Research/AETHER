# AETHER

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Language: Rust](https://img.shields.io/badge/Language-Rust-orange.svg)](https://www.rust-lang.org/)

**AETHER** is a minimalist, high-performance quantitative research engine. It utilizes **Genetic Programming (GP)** over physics-inspired stationary market features to evolve robust, non-linear mathematical agents for market research.

This project is provided for research and educational purposes only. It is not financial advice.

## Core Methodology

AETHER operates on a **Physics Layer** to ensure stationarity and prevent signal decay:

- **Kaufman Efficiency Ratio:** Quantifying signal-to-noise and trend persistence.
- **Market Friction:** Analyzing price displacement relative to volume intensity (Liquidity Exhaustion).
- **Log-Return Vol-of-Vol:** Detecting regime shifts before they manifest in price levels.

## Performance (Out-of-Sample Validation)

Results vary by dataset, interval, and seed. The snippet below reflects a recent run using `fetch --interval 4h --days 365` followed by `train` and `validate` with default parameters:

```text
--- Results ---
ROI:    412.76%
Sharpe: 8.63
MDD:    7.68%

--- Summary ---
Mean Test ROI: -7.09%
Baseline B&H:  -48.90%
```

Training metrics are in-sample and should not be interpreted as forward performance. Use `validate` on your own CSV for out-of-sample results.

## Getting Started

```bash
cargo build --release
./target/release/aether fetch --interval 4h --days 365
./target/release/aether audit data/market_data.csv
./target/release/aether train data/market_data.csv --seed 42
./target/release/aether validate data/market_data.csv --folds 3 --seed 42
```

## Data Format

Input CSV must include the following columns:

```
date,close,volume,ticker
```

## Notes on Reproducibility

- The `fetch` command pulls fresh data and overwrites `data/market_data.csv`, so results will vary run-to-run.
- `fetch` uses the Binance public API, which may be unavailable in some regions.
- For stable benchmarks, supply a fixed CSV and pin the random seed.

## Architecture

- **Regime-Awareness:** Specialized agents for Bull and Bear market states.
- **Asymmetric Co-Evolution:** Adversarial loop for Long/Short optimization.
- **L1-Optimized Interpreter:** Ultra-high-speed RPN evaluation kernel.

## License

MIT.
