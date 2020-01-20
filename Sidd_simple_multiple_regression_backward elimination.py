# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:11:40 2019

@author: 320001866
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r'C:\Sid Data\BITS\4th Sem\Udemey\Part 2 - Regression\Section 5 - Multiple Linear Regression\50_Startups.csv')
x_data = dataset.iloc[:, :-1].values
print (x_data)
x_data1 = dataset.iloc[:, :-1].values

row,columns = dataset.shape
column_index = columns-1
y_data =dataset.iloc[:,column_index].values
print(y_data)

# Data preprocesing
'''from sklearn.preprocessing import Imputer
pre_proessing = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)x_data_upd1 = np.append(arr = np.ones((row,1)).astype(int), values = x_data, axis = 1)

pre_proessing = pre_proessing.fit(x_data[:,1:3])
x_data[:,1:3] =  pre_proessing.transform(x_data[:,1:3])
print (x_data)'''

#Label Encoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#Label Encoder only changes text to numberic
x_encoder = LabelEncoder()
x_data[:,3] = x_encoder.fit_transform(x_data[:,3])
print (x_data)

#dummy encoding using Onehtencoder to transfortm them into rows of values
x_hot_encoder = OneHotEncoder(categorical_features =[3])
x_data = x_hot_encoder.fit_transform(x_data).toarray()
print (x_data)

#Remove one of the dummy variable
x_data = x_data[:, 1:]

#Split the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.2, train_size = 0.8, random_state = 0)




'''#Encode the Y value using the label encoder
y_encoder = LabelEncoder()
y_data = y_encoder.fit_transform(y_data)




#scaling of the data for normalization
from sklearn.preprocessing import StandardScaler
scaling_x = StandardScaler()
x_train_scale = scaling_x.fit_transform(x_train)
x_test_scale = scaling_x.transform(x_test)  '''

# Linear regression
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train, y_train)

#Transform the X_test
y_pred = reg.predict(x_test)

#Backward elimination
import statsmodels.api as sm
x_data = np.append(arr = np.ones((row, 1)).astype(int) , values =x_data , axis =1)

# Find featues that has high impact on the value Y
x_opt = x_data[:, :]
# Ordinary Least square
reg_ols = sm.OLS(endog = y_data,exog = x_opt).fit()

print(reg_ols.summary())





#visulzation - Train set
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, reg.predict(x_train))
plt.title('Training Set Plotting Training')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')


#Visualzation Test Set
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, reg.predict(x_train))
plt.title('Training Set Plotting - Test Set')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')











    








# New version of python...not working ------------------------------------
# Data preprocesing
'''from sklearn.impute import SimpleImputer
pre_proessing1 = SimpleImputer(missing_values = 'NaN', strategy = 'mean')
pre_proessing1 = pre_proessing1.fit(x_data1[:,1:3])

#dummy encoding using Onehtencoder to transfortm them into rows of values
x_hot_encoder = OneHotEncoder(categorical_features =[0])
x_data = x_hot_encoder.fit_transform(x_data).toarray()'''



