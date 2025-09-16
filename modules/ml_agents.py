# modules/ml_agents.py
import pandas as pd
import numpy as np

def availability_heatmap(forecast_df: pd.DataFrame, aht_sec=300, agents=10, interval_minutes=60) -> pd.DataFrame:
    hourly_weight = np.array([0.03,0.02,0.02,0.02,0.03,0.05,0.07,0.08,0.10,0.10,0.08,0.06,
                              0.05,0.05,0.05,0.05,0.04,0.03,0.02,0.02,0.02,0.02,0.01,0.01], dtype=float)
    hourly_weight /= hourly_weight.sum()
    rows = []
    for _, r in forecast_df.iterrows():
        day = pd.to_datetime(r['date']).date()
        daily_calls = float(r['forecast'])
        for h in range(24):
            calls_h = daily_calls * hourly_weight[h]
            offered_sec = calls_h * aht_sec
            staffed_sec = agents * (interval_minutes*60)
            util = offered_sec / staffed_sec if staffed_sec > 0 else 0.0
            rows.append({'date': day, 'hour': h, 'utilization': util})
    heat = pd.DataFrame(rows)
    heat['status'] = pd.cut(heat['utilization'], bins=[-0.01,0.7,0.9,10], labels=['Free','Busy','Overloaded'])
    return heat
