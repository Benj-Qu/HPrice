


import pandas as pd
import numpy as np
import os
import plotly.express as px

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor



def create_pipeline():
    """Create a machine learning pipeline"""
    
    preproc = ColumnTransformer(
        transformers=[
            ('log_trans', FunctionTransformer(np.log), ['Land Square Feet']),
            # ('categorical_cols', OneHotEncoder(drop='first'), ['group'])
        ],
        remainder='passthrough'
    )
    pipeline = Pipeline([
        ('preprocessor', preproc), 
        ('lin-reg', LinearRegression()),
    ])
    return pipeline