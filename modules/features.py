# modules/features.py
import pandas as pd
import numpy as np

def derive_engagement(leads: pd.DataFrame, calls: pd.DataFrame, days=30) -> pd.Series:
    c = calls.copy()
    c['CallDateTime'] = pd.to_datetime(c['CallDateTime'], errors='coerce')
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
    recent = c[c['CallDateTime'] >= cutoff]
    agg = recent.groupby('LeadId').agg(
        conn=('IsSuccessful','sum'),
        calls=('IsSuccessful','count'),
        dur=('DurationSeconds','mean')
    )
    score = (0.5*agg['conn'].fillna(0) +
             0.3*agg['calls'].fillna(0) +
             0.2*(agg['dur'].fillna(0)/300)).clip(0)
    score = 100*(score - score.min())/(score.max() - score.min() + 1e-9)
    return leads['LeadId'].map(score).fillna(0)

def attach_engagement_and_values(leads: pd.DataFrame, calls: pd.DataFrame) -> pd.DataFrame:
    df = leads.copy()
    df['EngagementScore'] = derive_engagement(df, calls)
    if 'RevenuePotential' not in df.columns:
        stage_weight = {1:0.3, 2:0.6, 3:0.8, 4:1.0}
        df['RevenuePotential'] = 100000 * df['LeadStageId'].map(stage_weight).fillna(0.3) * (0.5 + df['EngagementScore']/200)
    return df
