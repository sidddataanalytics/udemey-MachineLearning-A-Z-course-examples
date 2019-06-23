# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 09:00:30 2019

@author: Siddhartha Banerjee
"""
# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(X[:, 1:3])

# Taking care of categorical data and converth them to numerical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:,0]=labelencoder_x.fit_transform(x[:, 0])
onehotencoder_x = OneHotEncoder(categorical_features =[0])
x = onehotencoder_x.fit_transform(x).toarray()
#convert the Y value too
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Split the data into tain & test data sets 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, train_size = 0.8, random_state = 0)


# Feature scaling to normalize the values of different variables
# The formula used are x = (x - min X) / ( Max X - Min X)
from sklearn.preprocessing import StandardScaler
stand_x = StandardScaler()
x_train = stand_x.fit_transform(x_train)
# note fit & trasnform is not used, as the fir is already used in the previous function, so the stand_x knows what to use..the max & mon values of the test data
x_test = stand_x.transform(x_test)




