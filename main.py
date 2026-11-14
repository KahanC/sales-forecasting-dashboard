"""Build a static, interactive Plotly dashboard from synthetic sales data."""
import argparse
from pathlib import Path
import calendar
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent
CATEGORIES = ['WebSales', 'DropShip', 'USA']
COLORS = {2022: '#93C5FD', 2023: '#FCA5A5', 2024: '#34D399'}

def holiday_dates(year):
    """Return Black Friday and Cyber Monday for a given year."""
    nov1 = pd.Timestamp(year, 11, 1)
    first_thursday = nov1 + pd.Timedelta(days=(3 - nov1.weekday()) % 7)
    thanksgiving = first_thursday + pd.Timedelta(weeks=3)
    black_friday = thanksgiving + pd.Timedelta(days=1)
    return black_friday, black_friday + pd.Timedelta(days=3)

def load_data(path):
    df = pd.read_csv(path, parse_dates=['Date'])
    required = {'Date', 'Category', 'Sales'}
    if not required.issubset(df.columns):
        raise ValueError(f'CSV must contain {sorted(required)}')
    df['Sales'] = pd.to_numeric(df['Sales'], errors='raise')
    df = df.groupby(['Date', 'Category'], as_index=False)['Sales'].sum()
    # This synthetic dataset contains one row per calendar day and category.
    for category in CATEGORIES:
        subset = df[df.Category == category]
        for year in [2022, 2023, 2024]:
            for month in [11, 12]:
                expected = pd.date_range(f'{year}-{month:02d}-01', periods=calendar.monthrange(year, month)[1])
                actual = subset[(subset.Date.dt.year == year) & (subset.Date.dt.month == month)].Date
                if not expected.isin(actual).all():
                    raise ValueError(f'Missing daily records for {category} {year}-{month:02d}')
    return df

def forecast_baseline(history, future):
    """Transparent illustrative seasonal baseline, NOT Prophet."""
    values = []
    for dt in future:
        prior = history[(history.Date.dt.month == dt.month) & (history.Date.dt.day == dt.day)]
        prior = prior.sort_values('Date')
        x = prior.Date.dt.year.to_numpy()
        y = prior.Sales.to_numpy()
        slope = np.polyfit(x - 2022, np.log(np.maximum(y, 1)), 1)[0]
        slope = float(np.clip(slope, -.15, .30))
        estimate = float(y[-1] * np.exp(slope))
        # Align major promotion spikes to their moving dates.
        for yr in (2024, 2025):
            bf, cm = holiday_dates(yr)
            if yr == 2025 and dt.normalize() in [bf, cm]:
                ref = history[history.Date == (holiday_dates(2024)[0] if dt == bf else holiday_dates(2024)[1])]
                if not ref.empty:
                    estimate = float(ref.Sales.iloc[0] * np.exp(slope))
            elif yr == 2025:
                old_bf, old_cm = holiday_dates(2024)
                old_day = dt.replace(year=2024)
                if old_day in [old_bf, old_cm]:
                    estimate /= 1.7
        values.append(estimate)
    out = pd.DataFrame({'ds': future, 'yhat': values})
    out['yhat_lower'] = out.yhat * .72
    out['yhat_upper'] = out.yhat * 1.28
    return out

def forecast_prophet(history, future):
    try:
        from prophet import Prophet
    except ImportError as exc:
        raise SystemExit('Prophet is not installed. Run: pip install -r requirements.txt') from exc
    holidays = []
    for year in [2022, 2023, 2024, 2025]:
        bf, cm = holiday_dates(year)
        holidays.extend([{'holiday': 'black_friday', 'ds': bf}, {'holiday': 'cyber_monday', 'ds': cm}])
    model = Prophet(holidays=pd.DataFrame(holidays), yearly_seasonality=False,
                    weekly_seasonality=True, daily_seasonality=False,
                    interval_width=.80)
    # The dataset contains Nov-Dec only. Yearly seasonality is deliberately disabled.
    model.fit(history.rename(columns={'Date': 'ds', 'Sales': 'y'})[['ds', 'y']])
    return model.predict(pd.DataFrame({'ds': future}))

def build_dashboard(df, model_type, output):
    fig = make_subplots(rows=3, cols=2, subplot_titles=[f'{c} · {m}' for c in CATEGORIES for m in ['November', 'December']],
                        horizontal_spacing=.08, vertical_spacing=.11)
    for i, category in enumerate(CATEGORIES, start=1):
        history = df[df.Category == category].copy()
        for j, month in enumerate([11, 12], start=1):
            for year in [2022, 2023, 2024]:
                part = history[(history.Date.dt.year == year) & (history.Date.dt.month == month)].sort_values('Date')
                fig.add_trace(go.Scatter(x=part.Date.dt.day, y=part.Sales, mode='lines', name=str(year),
                                         legendgroup=str(year), showlegend=(i==1 and j==1),
                                         line=dict(color=COLORS[year], width=2), hovertemplate=f'Day %{{x}}<br>$%{{y:,.0f}}<extra>{year}</extra>'), row=i, col=j)
            dates = pd.date_range(f'2025-{month:02d}-01', periods=calendar.monthrange(2025, month)[1])
            fc = forecast_prophet(history, dates) if model_type == 'prophet' else forecast_baseline(history, dates)
            days = fc.ds.dt.day.to_list()
            fig.add_trace(go.Scatter(x=days + days[::-1], y=fc.yhat_upper.to_list() + fc.yhat_lower.to_list()[::-1],
                                     fill='toself', fillcolor='rgba(167,139,250,.17)', line=dict(color='rgba(0,0,0,0)'),
                                     name='Model interval' if model_type=='prophet' else 'Illustrative band (±28%)',
                                     legendgroup='band', showlegend=(i==1 and j==1), hoverinfo='skip'), row=i, col=j)
            fig.add_trace(go.Scatter(x=days, y=fc.yhat, name='2025 forecast' if model_type=='prophet' else '2025 illustrative baseline',
                                     legendgroup='forecast', showlegend=(i==1 and j==1),
                                     mode='lines', line=dict(color='#C4B5FD', width=3),
                                     hovertemplate='Day %{x}<br>$%{y:,.0f}<extra>2025</extra>'), row=i, col=j)
            fig.update_xaxes(title_text='Day of month', dtick=5, row=i, col=j, gridcolor='#273449')
            fig.update_yaxes(title_text='Sales ($)', tickprefix='$', tickformat=',.0f', row=i, col=j, gridcolor='#273449')
    subtitle = ('Prophet holiday-aware forecast' if model_type=='prophet' else 'Illustrative seasonal baseline (not Prophet)')
    fig.update_layout(template='plotly_dark', paper_bgcolor='#0B1220', plot_bgcolor='#101B2C',
                      font=dict(family='Arial', color='#E5E7EB'), height=1250,
                      title=dict(text=f'Sales Forecasting Dashboard<br><sup>Synthetic data · {subtitle} · 2022–2025</sup>', x=.04),
                      margin=dict(l=80, r=50, t=145, b=80), legend=dict(orientation='h', y=1.06, x=.03), hovermode='x unified')
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output), include_plotlyjs=True, full_html=True,
                   config={'responsive': True, 'displaylogo': False})
    print(f'Dashboard created: {output}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['baseline', 'prophet'], default='baseline',
                        help='baseline works without Prophet; prophet fits holiday-aware models')
    parser.add_argument('--output', type=Path, default=ROOT/'docs'/'index.html')
    args = parser.parse_args()
    build_dashboard(load_data(ROOT/'data'/'sample_sales.csv'), args.model, args.output)
