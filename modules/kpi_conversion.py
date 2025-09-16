# modules/kpi_conversion.py
import pandas as pd
import numpy as np

def propensity_weighted_pipeline(leads: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = leads.copy()
    rev = pd.to_numeric(df.get('ExpectedRevenue', df.get('RevenuePotential', 0)), errors='coerce').fillna(0.0)
    prop = pd.to_numeric(df.get('PropensityToConvert', 0), errors='coerce').fillna(0.0)
    df['ExpectedValue'] = rev * prop
    kpis = {
        'TotalPipeline': float(rev.sum()),
        'ExpectedRevenuePW': float(df['ExpectedValue'].sum()),
        'PipelineEfficiencyPct': float((df['ExpectedValue'].sum() / rev.sum() * 100) if rev.sum() > 0 else 0.0)
    }
    top10 = df.assign(ExpectedRevenue=rev).sort_values(['PropensityToConvert','ExpectedRevenue'], ascending=[False,False]).head(10)
    return pd.DataFrame([kpis]), top10
