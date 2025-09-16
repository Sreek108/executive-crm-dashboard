# modules/ml_leads.py
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

def build_lead_features(leads: pd.DataFrame) -> tuple[pd.DataFrame, list, list]:
    df = leads.copy()
    num_cols = [c for c in ['EngagementScore','LeadVelocity','RevenuePotential','ExpectedRevenue'] if c in df.columns]
    cat_cols = [c for c in ['LeadStageId','LeadScoringId','BehavioralSegment','CountryId','CityRegionId'] if c in df.columns]
    if 'EngagementScore' in df.columns and 'LeadVelocity' in df.columns:
        df['EngagePerDay'] = pd.to_numeric(df['EngagementScore'], errors='coerce') / (pd.to_numeric(df['LeadVelocity'], errors='coerce') + 1)
        num_cols.append('EngagePerDay')
    return df, sorted(set(num_cols)), sorted(set(cat_cols))

def train_lead_model(leads_labeled: pd.DataFrame, label_col='Converted') -> tuple[Pipeline, float]:
    df, num_cols, cat_cols = build_lead_features(leads_labeled)
    y = df[label_col].astype(int).values
    X = df[num_cols + cat_cols]
    num_tf = Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler(with_mean=False))])
    cat_tf = Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))])
    pre = ColumnTransformer([('num', num_tf, num_cols), ('cat', cat_tf, cat_cols)])
    base = Pipeline([('pre', pre), ('gb', GradientBoostingClassifier(random_state=42))])
    clf = CalibratedClassifierCV(base, method='isotonic', cv=3)
    clf.fit(X, y)
    auc = roc_auc_score(y, clf.predict_proba(X)[:,1])
    return clf, auc

def score_leads(model: Pipeline, leads: pd.DataFrame) -> pd.DataFrame:
    df, num_cols, cat_cols = build_lead_features(leads)
    proba = model.predict_proba(df[num_cols + cat_cols])[:,1]
    out = leads.copy()
    out['PropensityToConvert'] = proba.clip(0,1)
    return out

def next_best_action(row: pd.Series) -> tuple[str,str,float,str]:
    p = float(row.get('PropensityToConvert', 0.0))
    eng = float(pd.to_numeric(row.get('EngagementScore', 0), errors='coerce'))
    rev = float(pd.to_numeric(row.get('ExpectedRevenue', row.get('RevenuePotential', 0)), errors='coerce'))
    temp = str(row.get('TemperatureTrend', 'Unknown'))
    reason = []
    if p >= 0.75 and eng >= 60:
        reason.append('High intent + engagement')
        return ('Schedule Demo in 48h','High', min(0.95, 0.70 + 0.25*p), '; '.join(reason))
    if p >= 0.70 and rev > 0 and temp == 'Cooling Down':
        reason.append('Valuable lead cooling')
        return ('Retention Call in 24h','High', min(0.95, 0.70 + 0.25*p), '; '.join(reason))
    if p >= 0.60 and eng >= 50:
        reason.append('Moderate intent + good engagement')
        return ('Upsell Offer','Medium', 0.65, '; '.join(reason))
    if p >= 0.50:
        reason.append('Moderate intent—nurture')
        return ('Nurture Sequence','Medium', 0.60, '; '.join(reason))
    reason.append('Low intent—light touch')
    return ('Check-in Email','Low', 0.50, '; '.join(reason))

def attach_nba(scored: pd.DataFrame) -> pd.DataFrame:
    df = scored.copy()
    acts = df.apply(next_best_action, axis=1, result_type='expand')
    df[['NBA_Action','NBA_Priority','NBA_Confidence','NBA_Reason']] = acts
    return df
