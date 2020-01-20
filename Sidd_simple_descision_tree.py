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
y_data =dataset.iloc[:,column_index:].values
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
from sklearn.tree import DecisionTreeRegressor
reg_tree = DecisionTreeRegressor(random_state =0)
reg_tree.fit(x_data,  y_data)
y_sample_test = reg_tree.predict(np.array([[4.2]]))
y_pred_tree = reg_tree.predict(x_data)


#Visualzation Descision Tree
plt.scatter(x_data, y_data, color = 'red')
plt.plot(x_data, y_pred_tree, color = 'blue')
plt.title('Descision Tree')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Visualzation Descision Tree with actual categorization
x_grid = np.arange(min(x_data), max(x_data), 0.01)
x_grid = x_grid.reshape(len(x_grid),1)
plt.scatter(x_data, y_data, color = 'red')
plt.plot(x_data, y_pred_tree, color = 'red')
plt.plot(x_grid, reg_tree.predict(x_grid), color = 'blue')
plt.title('Descision Tree')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()







#SVR - fitting data on the SVR to the data set
from sklearn.svm import SVR
reg_svr = SVR(kernel = 'rbf')
reg_svr.fit(x_data, y_data)
y_pred = reg_svr.predict(x_data)



#Plot the Normal SVR without scaling
#Visualzation Test Set
#x_grid = np.arranger(min(x_data), max(x_data), 0.1)
#x_grid = x_grid.reschape(len(x_grid),1)
plt.scatter(x_data, y_data, color = 'red')
plt.plot(x_data, y_pred, color = 'blue')
#plt.plot(x_data, y_pred_poly_1, color = 'blue')
plt.title('TSVR without scaling')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#With scaling----------------
#scaling of the data for normalization
from sklearn.preprocessing import StandardScaler
scaling_x = StandardScaler()
scaling_y = StandardScaler()
x_scale = scaling_x.fit_transform(x_data)
y_scale = scaling_y.fit_transform(y_data)



from sklearn.svm import SVR
reg_svr_s = SVR(kernel = 'rbf')
reg_svr_s.fit(x_scale, y_scale)
y_pred_s = reg_svr_s.predict(x_scale)



plt.scatter(x_scale, y_scale, color = 'red')
plt.plot(x_scale, y_pred_s, color = 'blue')
plt.title('TSVR with scaling')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Apply inverse transform method to get the salary
print (scaling_y.inverse_transform(reg_svr_s.predict(scaling_x.transform(np.array([[6.5]])))))

# Step wise - To get a salary of6.5 years
# Step 1- Convert the experience to array
a = np.array([[6.5]])
#Step - Scale
b = scaling_x.transform(a)
print (scaling_x)
#Step 3 apply regression to predict
c = reg_svr_s.predict(b)
print (reg_svr_s)

#Step 3 - inverse transform the predictin
print (scaling_y.inverse_transform(c))







