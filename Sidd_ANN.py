# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 00:13:19 2020

@author: 320001866
"""

#import the libraires
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#imort the data set
dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values

#encode the categorical variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_geo = LabelEncoder()
labelencoder_gender = LabelEncoder()

x[:,1] = labelencoder_geo.fit_transform(x[:,1])
x[:,2] = labelencoder_gender.fit_transform(x[:,2])

onehotencoder = OneHotEncoder(categorical_features = [1])
x = onehotencoder.fit_transform(x).toarray()

#Remove the first column as it is not needed
x = x[:,1:]

#Split the train Test data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

#Start the ANN
import keras
from keras.models import Sequential
from keras.layers import Dense

#intilize the ANN
classifer = Sequential()

#Add the input layer
classifer.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
#add the 1st layer
classifer.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
#Add the 2nd Layer
classifer.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
#Add the output layer
classifer.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

#Compile the ANN
classifer.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifer.fit(x_train,y_train, batch_size = 10, nb_epoch =100 )

y_pred = classifer.predict(x_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

















