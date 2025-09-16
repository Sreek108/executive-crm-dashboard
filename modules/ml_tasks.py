# modules/ml_tasks.py
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

def build_task_features(df: pd.DataFrame):
    X = df.copy()
    num, cat = [], []
    if 'DaysUntilDue' in X.columns:
        X['DaysUntilDue'] = pd.to_numeric(X['DaysUntilDue'], errors='coerce')
        num.append('DaysUntilDue')
    for c in ['PriorityId','TaskStatusId','TaskTypeId','AssignedAgentId']:
        if c in X.columns: cat.append(c)
    return X, num, cat

def train_sla_model(tasks_labeled: pd.DataFrame, label_col='IsBreach') -> Pipeline:
    X, num, cat = build_task_features(tasks_labeled)
    y = tasks_labeled[label_col].astype(int).values
    pre = ColumnTransformer([
        ('num', Pipeline([('impute', SimpleImputer(strategy='median'))]), num),
        ('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                          ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat)
    ])
    lr = LogisticRegression(max_iter=1000, class_weight='balanced')
    pipe = Pipeline([('pre', pre), ('lr', lr)])
    pipe.fit(X[num+cat], y)
    return pipe

def score_sla(pipe: Pipeline, tasks: pd.DataFrame) -> pd.DataFrame:
    X, num, cat = build_task_features(tasks)
    p = pipe.predict_proba(X[num+cat])[:,1]
    out = tasks.copy()
    out['SLA_BreachRisk'] = p.clip(0,1)
    out['SLA_RiskLevel'] = pd.cut(out['SLA_BreachRisk'], bins=[-0.01,0.33,0.66,1.0], labels=['Low','Medium','High'])
    return out
