# Cook County Housing Prices Prediction

This project mainly focuses on [Exploratory Data Analysis](#exploratory-data-analysis), [Feature Engineering](#feature-engineering), and [Modeling](#modeling) to predict the housing prices in Cook County.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Feature Engineering](#feature-engineering)
- [Modeling](#modeling)

## Overview

The main task of this project is to predict the housing prices in Cook County. By conducting [Exploratory Data Analysis](#exploratory-data-analysis), we transformed the data and filtered out outliers. By conducting [Feature Engineering](#feature-engineering), we extracted information from texts, performed one-hot-encoding to categorical features, and briefly looked into the data. In [Modeling](#modeling), we fitted a few linear regression models and evaluated the performance.

## Dataset

The dataset comes from the Cook County Assessor’s Office (CCAO) in Illinois. The CCAO dataset consists of over 500 thousand records describing houses sold in Cook County in recent years with 61 features in total. The features of the dataset includes `Property Class`, `Land Square Feet`, `Apartments`, `Wall Material`, `Roof Material`, `Basement` and so on.

## Exploratory Data Analysis

As the targeted value `Sale Price` is heavily right-skewed, we performed a log-trasformation on the value, and it turns out the `Log Sale Price` is slightly left skewed. Similarly, we conduct log-transformation on `Building Square Feet`, and it turns out there is a significant correlation between `Log Sale Price` and `Building Square Feet`. Also, we filtered out some houses with extremely large Building Square Feet with border value `8000`, which are possibly outliers.

## Feature Engineering

From feature `Description`, we extracted information including `the date the property was sold on`, `the number of stories the property contains`, `the address of the property`, `the total number of rooms inside the property`, `the total number of bedrooms inside the property`, and `the total number of bathrooms inside the property`. It turns out that there is a significant correlation between `Log Sale Price` and `the total number of bedrooms inside the property`. Also, we discovered that there is some relationship between the neighborhood and `Log Sale Price`. Overall speaking, neighborhoods with more buildings may hold a lower `Log Sale Price`.

## Modeling

With the transformation done, we fitted two different linear regression models (`Log Sale Price` ~ `Bedrooms` and `Log Sale Price` ~ `Bedrooms` + `Log Building Square Feet`) based on the data. The rooted mean squared error for both models are quite large, but the fitted values show a significant correlation with the actual values.

Then we apply all the features provided to the model. We first conduct one-hot-encoding on all categorical variables, then compared the training error of models `Log Sale Price` ~ `Log Building Square Feet + ...`, `Sale Price` ~ `Building Square Feet + ...`, and `Sale Price / Building Square Feet` ~ `...`, where `...` includes all the features provided other than `Building Square Feet`. Among these models, `Sale Price / Building Square Feet` ~ `...` behaves the best. The test error eventually reaches about 65k.