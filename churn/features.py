import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ['CreditScore', 'Age', 'Balance', 'NumOfProducts', 'HasCrCard',
           'IsActiveMember', 'EstimatedSalary']
CATEGORICAL = ['Geography', 'Gender']
DATES = ['baseDate', 'accountOpeningDate']


class Features(TransformerMixin, BaseEstimator):
    """Use an allowlist: names, IDs, labels and unexpected columns never enter X."""
    def __init__(self, engineered=True):
        self.engineered = engineered

    def fit(self, X, y=None):
        self.transform(X)
        return self

    def transform(self, X):
        required = NUMERIC + CATEGORICAL + (DATES if self.engineered else [])
        missing = sorted(set(required) - set(X.columns))
        if missing:
            raise ValueError('Missing input columns: ' + ', '.join(missing))
        result = X[NUMERIC + CATEGORICAL].copy()
        for name in NUMERIC:
            result[name] = pd.to_numeric(result[name], errors='raise').astype(float)
            if np.isinf(result[name]).any():
                raise ValueError('Infinite input values: ' + name)
        for name in CATEGORICAL:
            result[name] = result[name].astype(object).where(result[name].notna(), np.nan)
        if self.engineered:
            base = pd.to_datetime(X['baseDate'], errors='coerce')
            opened = pd.to_datetime(X['accountOpeningDate'], errors='coerce')
            duration = (base - opened).dt.days.astype(float)
            result['accountDuration_days'] = duration.where(duration >= 0)
            result['salaryPerAge'] = result['EstimatedSalary'].div(result['Age'].where(result['Age'] > 0))
        return result


def pipeline(estimator, engineered=True):
    numeric = Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
                        ('scale', StandardScaler())])
    categorical = Pipeline([('impute', SimpleImputer(strategy='most_frequent', keep_empty_features=True)),
                            ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    transform = ColumnTransformer([
        ('numeric', numeric, make_column_selector(dtype_include=np.number)),
        ('categorical', categorical, make_column_selector(dtype_exclude=np.number)),
    ])
    return Pipeline([('features', Features(engineered)), ('preprocess', transform), ('model', estimator)])
