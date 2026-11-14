# Multi-Channel Sales Forecasting Dashboard

Interactive six-panel Plotly dashboard comparing daily sales in November and December (2022–2024) with an illustrative 2025 baseline or an optional Prophet forecast.

**[View the interactive dashboard](https://KahanC.github.io/sales-forecasting-dashboard/)** — replace `YOUR_USERNAME` after publishing.

## Privacy and data

All data is **synthetically generated**. It does not represent actual company sales, revenue, customers, or proprietary information. The prebuilt dashboard uses an **illustrative baseline**, not Prophet. Its ±28% bands are visual guides, **not statistically calibrated prediction intervals**.

## Run locally

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py --model prophet
```

Open `docs/index.html` in a browser. For a quick run without installing Prophet, install `pandas numpy plotly` and run `python main.py --model baseline`.

## Publish on GitHub Pages

1. Push this folder to a public GitHub repository.
2. In **Settings → Pages**, select **Deploy from a branch**, `main`, `/docs`.
3. Your interactive dashboard will appear at `https://YOUR_USERNAME.github.io/REPOSITORY_NAME/`.
4. To publish actual Prophet results, run `python main.py --model prophet`, commit the updated `docs/index.html`, and push again.

GitHub Pages serves HTML/JavaScript only; it does not run Python on page load.

## Implementation notes

- Synthetic complete daily observations for three channels, November–December 2022–2024.
- US Black Friday and Cyber Monday dates are calculated dynamically.
- Prophet uses weekly seasonality and holiday indicators; yearly seasonality is disabled because the dataset covers only two months per year.
- The illustrative baseline extrapolates same-calendar-day growth; its interval is heuristic and is not a substitute for validated forecasting.
- For production-quality modeling, use rolling backtests and measure MAE/WAPE against a holdout period; verify that zero-sales days are distinct from missing data.
