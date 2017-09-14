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

train = pd.read_csv('~/Documentos/Ciencia da Computacao/Machine Learning/New York City Taxi Trip Duration - Kaggle Competition/Data/train.csv', header=0)
test = pd.read_csv('~/Documentos/Ciencia da Computacao/Machine Learning/New York City Taxi Trip Duration - Kaggle Competition/Data/test.csv', header=0)

#--------------------------------------------------------------------------

# PRE-PROCESSAMENTO

train.dtypes
# One Hot Encoding
train = pd.get_dummies(train, columns=['store_and_fwd_flag'], prefix=['frag'])
test = pd.get_dummies(test, columns=['store_and_fwd_flag'], prefix=['frag'])

# Analise basica sem as colunas id, pickup_datetime e dropoff_datetime
train.drop(train.columns[[0, 2, 3]], axis=1, inplace=True)
train = train[['vendor_id', 'passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'frag_N', 'frag_Y', 'trip_duration']]

y_ids = test['id']      # Para arquivo de submissao
test.drop(test.columns[[0, 2]], axis=1, inplace=True)

X_train = train.iloc[:, 0:8].values
y_train = train.iloc[:, 8].values

X_test = test.iloc[:, 0:8].values

#-------------------------------------------------------------------------

# MODELO

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=12)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

#-------------------------------------------------------------------------

# SUBMISSION FILE
df = pd.DataFrame({"id": y_ids, "trip_duration": y_pred})
df.to_csv('submission.csv', index=False)