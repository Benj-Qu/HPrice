


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
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import TransformerMixin
from sklearn.neighbors import LocalOutlierFactor

def categoricals():
    categoricals = ['Modeling Group', 'Apartments', 'Wall Material', 'Roof Material', 
        'Basement', 'Basement Finish', 'Central Heating', 'Other Heating', 
        'Central Air', 'Attic Type', 'Attic Finish', 'Design Plan', 
        'Cathedral Ceiling', 'Construction Quality', 
        'Garage 1 Size', 'Garage 1 Material', 'Garage 1 Attachment', 'Garage 1 Area', 
        'Garage 2 Size', 'Garage 2 Material', 'Garage 2 Attachment', 'Garage 2 Area', 
        'Porch', 'Repair Condition', 'Multi Code', 'Use', 
        'Property Class', 'Story']
    return categoricals

def create_pipeline():
    """Create a machine learning pipeline"""
    preproc = ColumnTransformer(
        transformers=[
            ('log_trans', FunctionTransformer(np.log), ['Land Square Feet', 'Building Square Feet', 'Age']),
            ('categorical_cols', OneHotEncoder(drop='first', handle_unknown='ignore'), categoricals()),
        ],
        remainder='passthrough'
    )
    pipeline = Pipeline([
        ('extract_expense_neighbor', FunctionTransformer(find_expensive_neighborhoods)),
        ('extract_expense_town', FunctionTransformer(find_expensive_towns)),
        ('extract_description', FunctionTransformer(extract_description)),
        ('category_encoding', FunctionTransformer(substitute_categorical_variables)),
        ('drop_cols', FunctionTransformer(drop_columns)),
        ('preprocessor', preproc), 
        ('lin-reg', RandomForestRegressor()),
    ])
    return pipeline

def drop_columns(data):
    return data.drop(
        ['Other Improvements', 'Neighborhood Code', 'Town Code', 'Address', 'Longitude', 'Latitude', 'Site Desirability'], 
    axis=1)

def extract_description(data):
    with_rooms = data.copy()
    with_rooms['Sold Year'] = with_rooms['Description'].str.findall(".*sold on \d+/\d+/(\d+).*").str[0].fillna(2000).astype(int)
    with_rooms['Sold Month'] = with_rooms['Description'].str.findall(".*sold on (\d+)/\d+/\d+.*").str[0].fillna(1).astype(int)
    with_rooms['Sold Day'] = with_rooms['Description'].str.findall(".*sold on \d+/(\d+)/\d+.*").str[0].fillna(1).astype(int)
    with_rooms['Story'] = with_rooms['Description'].str.findall(".*(.*) houeshold.*").str[0].fillna('')
    with_rooms['Address'] = with_rooms['Description'].str.findall(".*located at (.*).*").str[0].fillna('')
    with_rooms['Rooms'] = with_rooms['Description'].str.findall(".*(\d+) rooms.*").str[0].fillna(1).astype(int)
    with_rooms['Bedrooms'] = with_rooms['Description'].str.findall(".*(\d+) of which are bedrooms.*").str[0].fillna(1).astype(int)
    with_rooms['Bathrooms'] = with_rooms['Description'].str.findall(".*(\d+) of which are bathrooms.*").str[0].fillna(0).astype(int)
    return with_rooms.drop('Description', axis=1)

def find_expensive_neighborhoods(data):
    expensive = [106,580,117,67,94,93,96,64,48,400,461,95,116,83,44,18,143,74,25,166]
    data['Expensive Neighborhood'] = data['Neighborhood Code'].apply(lambda x: int(x) in expensive)
    return data

def find_expensive_towns(data, n=3, metric=np.median):
    expensive = [23, 74, 73, 33, 25, 10, 17, 27, 19, 75]
    data['Expensive Town'] = data['Town Code'].apply(lambda x: int(x) in expensive)
    return data

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
    # data = data.replace({
    #     'Apartments': {
    #         2: 'Two',
    #         3: 'Three',
    #         4: 'Four',
    #         5: 'Five',
    #         6: 'Six',
    #         0: 'None',
    #     },
    #     'Wall Material': {
    #         1: 'Wood',
    #         2: 'Masonry',
    #         3: 'Wood&Masonry',
    #         4: 'Stucco',
    #     },
    #     'Roof Material': {
    #         1: 'Shingle/Asphalt',
    #         2: 'Tar&Gravel',
    #         3: 'Slate',
    #         4: 'Shake',
    #         5: 'Tile',
    #         6: 'Other',
    #     },
    #     'Basement': {
    #         1: 'Full',
    #         2: 'Slab',
    #         3: 'Partial',
    #         4: 'Crawl',
    #     },
    #     'Basement Finish': {
    #         1: 'Formal rec room',
    #         2: 'Apartment',
    #         3: 'Unfinished',
    #     },
    #     'Central Heating': {
    #         0: 'Warm air',
    #         1: 'Hot water steam',
    #         2: 'Electric',
    #         3: 'Other',
    #     },
    #     'Other Heating': {
    #         1: 'Floor furnace',
    #         2: 'Unit heater',
    #         3: 'Stove',
    #         4: 'Solar',
    #         5: 'none',
    #     },
    #     'Central Air': {
    #         1: 'yes',
    #         0: 'no',
    #     },
    #     'Attic Type': {
    #         1: 'Full',
    #         2: 'partial',
    #         3: 'none',
    #     },
    #     'Attic Finish': {
    #         1: 'Living area',
    #         2: 'Apartment',
    #         0: 'unfinished',
    #     },
    #     'Design Plan': {
    #         1: 'architect',
    #         2: 'stock plan',
    #     },
    #     'Cathedral Ceiling': {
    #         1: 'yes',
    #         2: 'no',
    #     },
    #     'Construction Quality': {
    #         1: 'Deluxe',
    #         2: 'Average',
    #         3: 'Poor',
    #         4: 'Other',
    #     },
    #     'Site Desirability': {
    #         1: 'Beneficial to Value',
    #         2: 'Not relevant to Value',
    #         3: 'Detracts from Value',
    #     },
    #     'Garage 1 Size': {
    #         1: '1 car',
    #         2: '1.5 car',
    #         3: '2 car',
    #         4: '2.5 cars',
    #         5: '3 cars',
    #         6: '3.5 cars',
    #         7: 'none',
    #         8: '4 cars',
    #     },
    #     'Garage 1 Material': {
    #         1: 'Frame',
    #         2: 'Masonry',
    #         3: 'Frame/Masonry',
    #         4: 'Stucco',
    #     },
    #     'Garage 1 Attachment': {
    #         1: 'Yes',
    #         2: 'No',
    #     },
    #     'Garage 1 Area': {
    #         1: 'Yes',
    #         2: 'No',
    #     },
    #     'Garage 2 Size': {
    #         1: '1 car',
    #         2: '1.5 car',
    #         3: '2 car',
    #         4: '2.5 cars',
    #         5: '3 cars',
    #         6: '3.5 cars',
    #         7: 'none',
    #         8: '4 cars',
    #     },
    #     'Garage 2 Material': {
    #         1: 'Frame',
    #         2: 'Masonry',
    #         3: 'Frame/Masonry',
    #         4: 'Stucco',
    #     },
    #     'Garage 2 Attachment': {
    #         1: 'Yes',
    #         2: 'No',
    #     },
    #     'Garage 2 Area': {
    #         1: 'Yes',
    #         2: 'No',
    #     },
    #     'Porch': {
    #         1: 'Frame',
    #         2: 'Masonry',
    #         3: 'None',
    #     },
    #     'Repair Condition': {
    #         1: 'Above average',
    #         2: 'Average',
    #         3: 'Below average',
    #     },
    #     'Multi Code': {
    #         2: '1 building',
    #         3: '2 buildings',
    #         4: '3 buildings',
    #         5: '4 buildings',
    #         6: '5 buildings',
    #         7: '6 buildings',
    #     },
    #     'Use': {
    #         1: 'single family',
    #         2: 'multi-family',
    #     },
        
    # })
    return data

def remove_outliers(data, variable, lower=-np.inf, upper=np.inf):
    return data[(data[variable] >= lower) & (data[variable] <= upper)]