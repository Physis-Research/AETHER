# AETHER

**AETHER** is a high-performance quantitative research engine utilizing **Regime-Aware Genetic Programming** to evolve robust mathematical agents.

## Core Methodology

- **Physics Layer:** Ensures stationarity via Efficiency Ratio, Market Friction, and Hurst exponents.
- **Regime Awareness:** Automatically switches between bull/bear models based on volatility.
- **Robustness Focus:** 70% of fitness weight is allocated to out-of-sample validation to prevent overfitting.
- **Vectorized Interpreter:** High-throughput RPN stack evaluation for massive evolution speed.

## Usage

```bash
cargo build --release
# Fetch and audit data
./target/release/aether fetch --tickers BTCUSDT,ETHUSDT --days 365 --audit
# Validate out-of-sample performance
./target/release/aether validate data/market_data.csv --folds 3 --seed 42
```

## Performance Metrics

- **ROI / Sharpe / Sortino:** Risk-adjusted return assessment.
- **MDD:** Maximum Drawdown penalty integrated into fitness.
- **Diversity Penalty:** Encourages decorrelated long/short strategies.

## License

MIT.
