#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 21:42:33 2017

@author: Christian R. F. Gomes
@title: Support Vector Regression
"""

# Importando as bibliotecas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#--------------------------------------------------------------------------

# BASES DE DADOS

train = pd.read_csv('~/Documentos/Ciencia da Computacao/Machine Learning/New York City Taxi Trip Duration - Kaggle Competition/Data/train.csv', header=0)
test = pd.read_csv('~/Documentos/Ciencia da Computacao/Machine Learning/New York City Taxi Trip Duration - Kaggle Competition/Data/test.csv', header=0)

#--------------------------------------------------------------------------

# PRE-PROCESSAMENTO

# Tipos de cada coluna
train.dtypes

# One Hot Encoding
train = pd.get_dummies(train, columns=['store_and_fwd_flag'], prefix=['frag'])

# Analise basica sem as colunas id, pickup_datetime e dropoff_datetime
train.drop(train.columns[[0, 2, 3]], axis=1, inplace=True)

train = train[['vendor_id', 'passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'frag_N', 'frag_Y', 'trip_duration']]

# Separacao em atributos (X) e classe (y)
X = train.iloc[:, 0:8].values
y = train.iloc[:, 8].values


# Feature Scaling

#--------------------------------------------------------------------------

# CRIACAO E AVALIACAO DO MODELO

# Criacao do modelo SVR
from sklearn.svm import SVR
svr_regressor = SVR(kernel='rbf')
svr_regressor.fit(X, y)

