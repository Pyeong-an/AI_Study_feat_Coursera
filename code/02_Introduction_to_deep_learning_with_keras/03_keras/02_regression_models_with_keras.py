import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import numpy as np
import keras

import warnings
warnings.simplefilter('ignore', FutureWarning)

# 1. Load data
filepath='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv'
concrete_data = pd.read_csv(filepath)

# load test
# print(concrete_data.head())
# Let's check how many data points we have
# print(concrete_data.shape)
# describe
# print(concrete_data.describe())
# is null sum
# print(concrete_data.isnull().sum())

# 2. preprocessing
concrete_data_columns = concrete_data.columns
predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']] # all columns except Strength
target = concrete_data['Strength'] # Strength column

# split test
# print(predictors.head())
# print(target.head())

predictors_norm = (predictors - predictors.mean()) / predictors.std()

# normalize test
# print(predictors_norm.head())

n_cols = predictors_norm.shape[1] # number of predictors

# 2. Build a Neural Network
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input

# define regression model
def regression_model():
    # create model
    model = Sequential()
    model.add(Input(shape=(n_cols,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))

    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

model = regression_model()
model.fit(predictors_norm, target, validation_split=0.1, epochs=100, verbose=2)