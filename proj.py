


import pandas as pd
import numpy as np
import os

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

def categoricals():
    categoricals = ['Modeling Group', 'Apartments', 'Wall Material', 'Roof Material', 
        'Basement', 'Basement Finish', 'Central Heating', 'Other Heating', 
        'Central Air', 'Attic Type', 'Attic Finish', 'Design Plan', 
        'Cathedral Ceiling', 'Construction Quality', 'Site Desirability', 
        'Garage 1 Size', 'Garage 1 Material', 'Garage 1 Attachment', 'Garage 1 Area', 
        'Garage 2 Size', 'Garage 2 Material', 'Garage 2 Attachment', 'Garage 2 Area', 
        'Porch', 'Repair Condition', 'Multi Code', 'Use']
    return categoricals

def create_pipeline():
    """Create a machine learning pipeline"""
    preproc = ColumnTransformer(
        transformers=[
            ('log_trans', FunctionTransformer(np.log), ['Land Square Feet']),
            ('categorical_cols', OneHotEncoder(drop='first'), categoricals())
        ],
        remainder='passthrough'
    )
    pipeline = Pipeline([
        ('category_encoding', FunctionTransformer(substitute_categorical_variables)),
        ('one_hot_encoding', FunctionTransformer(one_hot_encode)),
        ('preprocessor', preproc), 
        ('lin-reg', LinearRegression()),
    ])
    return pipeline

def add_total_bedrooms(data):
    with_rooms = data.copy()
    with_rooms["Bedrooms"] = with_rooms["Description"].str.findall(".*(\d+) of which are bedrooms.*").str[0].fillna(0).astype(int)
    with_rooms["Rooms"] = with_rooms["Description"].str.findall(".*(\d+) rooms.*").str[0].fillna(0).astype(int)
    with_rooms["Bathrooms"] = with_rooms["Description"].str.findall(".*(\d+) of which are bathrooms.*").str[0].fillna(0).astype(int)
    return with_rooms

def one_hot_encode(data):
    def custom_combiner(feature, category):
        return feature + " " + str(category)
    for categorical in categoricals():
        enc = OneHotEncoder(drop='first', handle_unknown='ignore')
        enc.fit(data[[categorical]])
        new_cols = pd.DataFrame(enc.transform(data[[categorical]]).todense(),
            columns=enc.get_feature_names_out(),
            index=data.index)
        data = data.join(new_cols) 
    return data

def ohe_wall_material(data):
    enc = OneHotEncoder()
    enc.fit(data[["Wall Material"]])
    new_cols = pd.DataFrame(enc.transform(data[["Wall Material"]]).todense(), 
        columns = enc.get_feature_names(),
        index = data.index)
    return data.join(new_cols)                      

def substitute_categorical_variables(data):
    categoricals = ['Apartments', 'Wall Material', 'Roof Material', 
        'Basement', 'Basement Finish', 'Central Heating', 'Other Heating', 
        'Central Air', 'Attic Type', 'Attic Finish', 'Design Plan', 
        'Cathedral Ceiling', 'Construction Quality', 'Site Desirability', 
        'Garage 1 Size', 'Garage 1 Material', 'Garage 1 Attachment', 'Garage 1 Area', 
        'Garage 2 Size', 'Garage 2 Material', 'Garage 2 Attachment', 'Garage 2 Area', 
        'Porch', 'Repair Condition', 'Multi Code', 'Use']
    for categorical in categoricals:
        data[categorical] = data[categorical].astype(int)
    return data