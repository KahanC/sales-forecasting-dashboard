# Multi-Channel Sales Forecasting Dashboard

Interactive six-panel Plotly dashboard comparing daily sales in November and December (2022–2024) with an illustrative 2025 baseline and an optional Prophet forecast with an error margin of the number.

**[View the interactive dashboard](https://KahanC.github.io/sales-forecasting-dashboard/)**

## Privacy and data

All data is **synthetically generated**. It does not represent actual company sales, revenue, customers, or proprietary information. The prebuilt dashboard uses an **illustrative baseline**, not Prophet. Its ±28% bands are visual guides, **not statistically calibrated prediction intervals**.

## Implementation notes

- Synthetic complete daily observations for three channels, November–December 2022–2024.
- US Black Friday and Cyber Monday dates are calculated dynamically.
- Prophet uses weekly seasonality and holiday indicators; yearly seasonality is disabled because the dataset covers only two months per year.
- The illustrative baseline extrapolates same-calendar-day growth; its interval is heuristic and is not a substitute for validated forecasting.
- For production-quality modeling, use rolling backtests and measure MAE/WAPE against a holdout period; verify that zero-sales days are distinct from missing data.
