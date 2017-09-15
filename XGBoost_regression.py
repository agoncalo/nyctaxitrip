#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 22:15:08 2017

@author: Christian R. F. Gomes
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#--------------------------------------------------------------------------

# BASES DE DADOS
train = pd.read_csv('~/Documentos/Ciencia da Computacao/Machine Learning/New York City Taxi Trip Duration - Kaggle Competition/Data/train.csv', header=0, names=['ID', 'VENDOR_ID', 'P_DATETIME', 'D_DATETIME', 'PASSAGENR_COUNT', 'P_LONGITUDE', 'P_LATITUDE', 'D_LONGITUDE', 'D_LATITUDE', 'FLAG', 'T_DURATION'], parse_dates=[2, 3])
test = pd.read_csv('~/Documentos/Ciencia da Computacao/Machine Learning/New York City Taxi Trip Duration - Kaggle Competition/Data/test.csv', header=0, names=['ID', 'VENDOR_ID', 'P_DATETIME', 'PASSAGENR_COUNT', 'P_LONGITUDE', 'P_LATITUDE', 'D_LONGITUDE', 'D_LATITUDE', 'FLAG'], parse_dates=[2])

#--------------------------------------------------------------------------

# PRE-PROCESSAMENTO

train.dtypes
# One Hot Encoding
train = pd.get_dummies(train, columns=['FLAG'])
test = pd.get_dummies(test, columns=['FLAG'])

# DATA
type(train['P_DATETIME'])
train['P_DATETIME'] = train['P_DATETIME'].dt.strftime('%j')
train.drop(train.columns[[0, 3]], axis=1, inplace=True)
train = train[['VENDOR_ID', 'P_DATETIME', 'PASSAGENR_COUNT', 'P_LONGITUDE', 'P_LATITUDE', 'D_LONGITUDE', 'D_LATITUDE', 'FLAG', 'T_DURATION']]

test['P_DATETIME'] = test['P_DATETIME'].dt.strftime('%j')
y_ids = test['ID']      # Para arquivo de submissao
test.drop(test.columns[[0]], axis=1, inplace=True)

#==============================================================================
# # Analise basica sem as colunas id, pickup_datetime e dropoff_datetime
# train.drop(train.columns[[0, 2, 3]], axis=1, inplace=True)
# train = train[['vendor_id', 'passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'frag_N', 'frag_Y', 'trip_duration']]
# y_ids = test['id']      # Para arquivo de submissao
# test.drop(test.columns[[0, 2]], axis=1, inplace=True)
#==============================================================================

X_train = train.iloc[:, 0:9].values
y_train = train.iloc[:, 9].values
X_test = test.iloc[:, 0:8].values

#-------------------------------------------------------------------------

# MODELO
from xgboost import XGBRegressor
regressor = XGBRegressor(n_estimators=1000, silent=0, nthread=6, seed=2, max_depth=10, objective='reg:gamma')
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

#-------------------------------------------------------------------------

# SUBMISSION FILE
df = pd.DataFrame({"id": y_ids, "trip_duration": y_pred})
df.to_csv('XGBoost_submission.csv', index=False)

# valores negativos
df[(df < 0).all(1)]
df.loc[(df['trip_duration'] < 0), 'trip_duration'] = 100


