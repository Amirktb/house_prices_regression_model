from typing import List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class TemporalVariableTransformer(BaseEstimator, TransformerMixin):
    """Temporal elapsed time transformer."""

    def __init__(self, variables: List[str], reference_var: str):

        if not isinstance(variables, list):
            raise ValueError("variables should be a list")
        
        self.variables = variables
        self.reference_var = reference_var

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # to not overwrite the original dataframe
        X = X.copy()

        for feature in self.variables:
            X[feature] = X[self.reference_var] - X[feature]
        
        return X
    
class Mapper(BaseEstimator, TransformerMixin):
    """Categorical variable mapper."""

    def __init__(self, variables: List[str], mappings: dict):
        
        if not isinstance(variables, list):
            raise ValueError("Varibales should be a list")
        
        self.variables = variables
        self.mappings = mappings

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].map(self.mappings)

        return X