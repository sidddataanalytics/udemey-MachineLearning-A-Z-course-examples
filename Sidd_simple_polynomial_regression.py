# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:11:40 2019

@author: 320001866
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r'C:\Sid Data\BITS\4th Sem\Udemey\Part 2 - Regression\Section 6 - Polynomial Regression\Position_Salaries.csv')
x_data = dataset.iloc[:, 1:2].values
print (x_data)
#x_data1 = dataset.iloc[:, :-1].values

row,columns = dataset.shape
column_index = columns-1
y_data =dataset.iloc[:,column_index].values
print(y_data)

# Data preprocesing
'''from sklearn.preprocessing import Imputer
pre_proessing = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
pre_proessing = pre_proessing.fit(x_data[:,1:3])
x_data[:,1:3] =  pre_proessing.transform(x_data[:,1:3])
print (x_data)

#Label Encoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#Label Encoder only changes text to numberic
x_encoder = LabelEncoder()
x_data[:,0] = x_encoder.fit_transform(x_data[:,0])
print (x_data)

#dummy encoding using Onehtencoder to transfortm them into rows of values
x_hot_encoder = OneHotEncoder(categorical_features =[0])
x_data = x_hot_encoder.fit_transform(x_data).toarray()

#Encode the Y value using the label encoder
y_encoder = LabelEncoder()
y_data = y_encoder.fit_transform(y_data)'''


#Split the data
#from sklearn.model_selection import train_test_split
#x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.2, train_size = 0.8, random_state = 0)

#scaling of the data for normalization
'''from sklearn.preprocessing import StandardScaler
scaling_x = StandardScaler()
x_train_scale = scaling_x.fit_transform(x_train)
x_test_scale = scaling_x.transform(x_test)  '''

# Linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x_data, y_data)
#Transform the X_test
#y_pred = lin_reg.predict(x_test)



# Polynomia regression to add polyynomical features
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)  # Degree 3
x_poly = poly_reg.fit_transform(x_data)
poly_reg.fit(x_poly, y_data)
lin_poly_reg = LinearRegression()
lin_poly_reg.fit(x_poly, y_data)
#Transform the X_test
y_pred_poly = lin_poly_reg.predict(x_poly)


# Polynomia regression to add polyynomical features
from sklearn.preprocessing import PolynomialFeatures
poly_reg_4 = PolynomialFeatures(degree = 3)
x_poly_4 = poly_reg_4.fit_transform(x_data)
poly_reg_4.fit(x_poly_4, y_data)
lin_poly_reg_4 = LinearRegression()
lin_poly_reg_4.fit(x_poly_4, y_data)
#Transform the X_test
y_pred_poly_4 = lin_poly_reg_4.predict(x_poly_4)


#visulzation - Linear regression
plt.scatter(x_data, y_data, color = 'red')
plt.plot(x_data, lin_reg.predict(x_data), color ='blue')
plt.title('Training Set Plotting Training - Linear')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()



#Visualzation Test Set
plt.scatter(x_data, y_data, color = 'red')
plt.plot(x_data, y_pred_poly_4, color = 'blue')
#plt.plot(x_data, y_pred_poly_1, color = 'blue')
plt.title('Training Set Plotting Poly')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


#Visualzation Test Set
plt.scatter(x_data, y_data, color = 'red')
plt.plot(x_data, y_pred_poly, color = 'blue')
#plt.plot(x_data, y_pred_poly_1, color = 'blue')
plt.title('Training Set Plotting Poly')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Test with different preduction

print (lin_poly_reg.predict(poly_reg.fit_transform(6,2)))









    








# New version of python...not working ------------------------------------
# Data preprocesing
'''from sklearn.impute import SimpleImputer
pre_proessing1 = SimpleImputer(missing_values = 'NaN', strategy = 'mean')
pre_proessing1 = pre_proessing1.fit(x_data1[:,1:3])

#dummy encoding using Onehtencoder to transfortm them into rows of values
x_hot_encoder = OneHotEncoder(categorical_features =[0])
x_data = x_hot_encoder.fit_transform(x_data).toarray()'''



