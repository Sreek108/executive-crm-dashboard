# modules/geo_priority.py
import pandas as pd
import numpy as np

def market_priority(leads: pd.DataFrame) -> pd.DataFrame:
    df = leads.copy()
    df['rev'] = pd.to_numeric(df.get('ExpectedRevenue', df.get('RevenuePotential', 0)), errors='coerce').fillna(0.0)
    df['prop'] = pd.to_numeric(df.get('PropensityToConvert', 0), errors='coerce').fillna(0.0)
    df['cycle'] = pd.to_numeric(df.get('LeadVelocity', np.nan), errors='coerce')
    g = df.groupby('CountryId').agg(
        leads=('FullName','count'),
        exp_rev=('rev','sum'),
        mean_rev=('rev','mean'),
        mean_prop=('prop','mean'),
        mean_cycle=('cycle','mean')
    ).reset_index()
    for c in ['exp_rev','mean_rev','mean_prop']:
        g[c+'_n'] = (g[c] - g[c].min())/(g[c].max()-g[c].min()+1e-9)
    g['mean_cycle_n'] = 1 - (g['mean_cycle'] - g['mean_cycle'].min())/(g['mean_cycle'].max()-g['mean_cycle'].min()+1e-9)
    g['PriorityScore'] = 0.35*g['exp_rev_n'] + 0.25*g['mean_prop_n'] + 0.25*g['mean_rev_n'] + 0.15*g['mean_cycle_n']
    return g.sort_values('PriorityScore', ascending=False)
