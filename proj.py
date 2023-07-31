


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

def add_total_bedrooms(data):
    with_rooms = data.copy()
    with_rooms["Bedrooms"] = with_rooms["Description"].str.findall(".*(\d+) of which are bedrooms.*").str[0].fillna(0).astype(int)
    with_rooms["Rooms"] = with_rooms["Description"].str.findall(".*(\d+) rooms.*").str[0].fillna(0).astype(int)
    with_rooms["Bathrooms"] = with_rooms["Description"].str.findall(".*(\d+) of which are bathrooms.*").str[0].fillna(0).astype(int)
    return with_rooms

def ohe_roof_material(data):
    enc = OneHotEncoder()
    enc.fit(data[['Roof Material']])
    new_cols = pd.DataFrame(enc.transform(data[['Roof Material']]).todense(),
        columns=enc.get_feature_names(),
        index=data.index)
    return data.join(new_cols) 

def ohe_wall_material(data):
    enc = OneHotEncoder()
    enc.fit(data[["Wall Material"]])
    new_cols = pd.DataFrame(enc.transform(data[["Wall Material"]]).todense(), 
        columns = enc.get_feature_names(),
        index = data.index)
    return data.join(new_cols)                      
    
def select_columns(data, *columns):
    """Select only columns passed as arguments."""
    return data.loc[:, columns]

def logTransform(data):
    data['Log Building Square Feet'] = np.log(data['Building Square Feet'])
    return data

def rmse(predicted, actual):
    return np.sqrt(np.mean((actual - predicted)**2))

def process_data_gm(data, pipeline_functions):
    """Process the data for a guided model."""
    for function, arguments, keyword_arguments in pipeline_functions:
        if keyword_arguments and (not arguments):
            data = data.pipe(function, **keyword_arguments)
        elif (not keyword_arguments) and (arguments):
            data = data.pipe(function, *arguments)
        else:
            data = data.pipe(function)
    return data

def substitute_categorical_variables(data):
    data = data.replace({
        'Apartments': {
            2: 'Two',
            3: 'Three',
            4: 'Four',
            5: 'Five',
            6: 'Six',
            0: 'None',
        },
        'Wall Material': {
            1: 'Wood',
            2: 'Masonry',
            3: 'Wood&Masonry',
            4: 'Stucco',
        },
        'Roof Material': {
            1: 'Shingle/Asphalt',
            2: 'Tar&Gravel',
            3: 'Slate',
            4: 'Shake',
            5: 'Tile',
            6: 'Other',
        },
        'Basement': {
            1: 'Full',
            2: 'Slab',
            3: 'Partial',
            4: 'Crawl',
        },
        'Basement Finish': {
            1: 'Formal rec room',
            2: 'Apartment',
            3: 'Unfinished',
        },
        'Central Heating': {
            1: 'Warm air',
            2: 'Hot water steam',
            3: 'Electric',
            4: 'Other',
        },
        'Other Heating': {
            1: 'Floor furnace',
            2: 'Unit heater',
            3: 'Stove',
            4: 'Solar',
            5: 'none',
        },
        'Central Air': {
            1: 'yes',
            2: 'no',
        },
        'Attic Type': {
            1: 'Full',
            2: 'partial',
            3: 'none',
        },
        'Attic Finish': {
            1: 'Living area',
            2: 'Apartment',
            3: 'unfinished',
        },
        'Design Plan': {
            1: 'architect',
            2: 'stock plan',
        },
        'Cathedral Ceiling': {
            1: 'yes',
            2: 'no',
        },
        'Construction Quality': {
            1: 'Deluxe',
            2: 'Average',
            3: 'Poor',
            4: 'Other',
        },
        'Site Desirability': {
            1: 'Beneficial to Value',
            2: 'Not relevant to Value',
            3: 'Detracts from Value',
        },
        'Garage 1 Size': {
            1: '1 car',
            2: '1.5 car',
            3: '2 car',
            4: '2.5 cars',
            5: '3 cars',
            6: '3.5 cars',
            7: 'none',
            8: '4 cars',
        },
        'Garage 1 Material': {
            1: 'Frame',
            2: 'Masonry',
            3: 'Frame/Masonry',
            4: 'Stucco',
        },
        'Garage 1 Attachment': {
            1: 'Yes',
            2: 'No',
        },
        'Garage 1 Area': {
            1: 'Yes',
            2: 'No',
        },
        'Garage 2 Size': {
            1: '1 car',
            2: '1.5 car',
            3: '2 car',
            4: '2.5 cars',
            5: '3 cars',
            6: '3.5 cars',
            7: 'none',
            8: '4 cars',
        },
        'Garage 2 Material': {
            1: 'Frame',
            2: 'Masonry',
            3: 'Frame/Masonry',
            4: 'Stucco',
        },
        'Garage 2 Attachment': {
            1: 'Yes',
            2: 'No',
        },
        'Garage 2 Area': {
            1: 'Yes',
            2: 'No',
        },
        'Porch': {
            1: 'Frame',
            2: 'Masonry',
            3: 'None',
        },
        'Repair Condition': {
            1: 'Above average',
            2: 'Average',
            3: 'Below average',
        },
        'Multi Code': {
            2: '1 building',
            3: '2 buildings',
            4: '3 buildings',
            5: '4 buildings',
            6: '5 buildings',
            7: '6 buildings',
        },
        'Use': {
            1: 'single family',
            2: 'multi-family',
        },
        
    })
    return data