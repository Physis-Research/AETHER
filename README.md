# AETHER

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Language: Rust](https://img.shields.io/badge/Language-Rust-orange.svg)](https://www.rust-lang.org/)

**AETHER** is a minimalist, high-performance quantitative research engine. It utilizes **Genetic Programming (GP)** over physics-inspired stationary market features to evolve robust, non-linear mathematical agents capable of harvesting alpha from market entropy.

## Core Methodology

AETHER operates on a **Physics Layer** to ensure stationarity and prevent signal decay:

- **Kaufman Efficiency Ratio:** Quantifying signal-to-noise and trend persistence.
- **Market Friction:** Analyzing price displacement relative to volume intensity (Liquidity Exhaustion).
- **Log-Return Vol-of-Vol:** Detecting regime shifts before they manifest in price levels.

## Performance (Out-of-Sample Validation)

The results below are illustrative examples from a prior run and are **not** reproducible from this repo alone without the original dataset and exact runtime conditions:

- **Market Benchmark (B&H):** -49.59%
- **AETHER Performance:** -10.09%
- **Net Alpha:** **+39.50%** relative outperformance.
- **Peak Sharpe:** 8.72

If you want reproducible numbers, run `validate` on your own CSV and report the outputs produced by the current code.

## Getting Started

```bash
cargo build --release
./target/release/aether fetch --interval 4h --days 365
./target/release/aether audit data/market_data.csv
./target/release/aether train data/market_data.csv --seed 42
./target/release/aether validate data/market_data.csv --folds 3 --seed 42
```

## Reproducibility

- The `fetch` command pulls fresh data and overwrites `data/market_data.csv`, so results will vary run-to-run.
- For stable benchmarks, supply a fixed CSV (date, close, volume, ticker) and pin the random seed.

## Architecture

- **Regime-Awareness:** Specialized agents for Bull and Bear market states.
- **Asymmetric Co-Evolution:** Adversarial loop for Long/Short optimization.
- **L1-Optimized Interpreter:** Ultra-high-speed RPN evaluation kernel.

## License

MIT.
