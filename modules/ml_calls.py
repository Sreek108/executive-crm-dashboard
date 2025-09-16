# modules/ml_calls.py
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def daily_call_series(calls: pd.DataFrame, date_col='CallDateTime') -> pd.Series:
    ts = calls.copy()
    ts[date_col] = pd.to_datetime(ts[date_col], errors='coerce')
    ds = ts.set_index(date_col).sort_index().resample('D').size()
    return ds.asfreq('D').fillna(0)

def forecast_calls(ds: pd.Series, periods=30) -> pd.DataFrame:
    model = ExponentialSmoothing(ds, seasonal='add', seasonal_periods=7, trend='add', initialization_method='estimated')
    fit = model.fit(optimized=True)
    f = fit.forecast(periods)
    resid = ds - fit.fittedvalues
    sigma = float(np.nanstd(resid))
    out = pd.DataFrame({'date': f.index, 'forecast': f.values})
    out['lo'] = out['forecast'] - 1.96*sigma
    out['hi'] = out['forecast'] + 1.96*sigma
    return out

def optimal_call_windows(calls: pd.DataFrame) -> pd.DataFrame:
    df = calls.copy()
    if 'CallHour' not in df.columns:
        df['CallHour'] = pd.to_datetime(df['CallDateTime'], errors='coerce').dt.hour
    grp = df.groupby('CallHour').agg(total=('IsSuccessful','count'), success=('IsSuccessful','sum'))
    grp['success_rate'] = (grp['success'] / grp['total']).fillna(0.0)
    return grp.reset_index().sort_values('success_rate', ascending=False)
