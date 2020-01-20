# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:11:40 2019

@author: 320001866
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r'C:\Sid Data\BITS\4th Sem\Udemey\Part 3 - Classification\Section 14 - Logistic Regression\Social_Network_Ads.csv')
# Ned to get age & salary correspondence as X
x_data = dataset.iloc[:, [2,3]].values
print (x_data)
#x_data1 = dataset.iloc[:, :-1].values

row,columns = dataset.shape
column_index = columns-1
y_data =dataset.iloc[:,column_index].values
print(y_data)

# Split Train & test set
from sklearn.model_selection import train_test_split
x_data_train, x_data_test, y_data_train,y_data_test = train_test_split(x_data, y_data, test_size =0.25, random_state = 0)    

#scale the data for running algorithms
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_data_train = sc_x.fit_transform(pd.DataFrame(x_data_train))
x_data_test = sc_x.fit_transform(pd.DataFrame(x_data_test))
print (x_data)
print (x_data_train)

#Logistic Regression-------------------------------------
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(random_state =0)
log_reg.fit(x_data_train, y_data_train)
#Predict test results
y_data_pred = log_reg.predict(x_data_test)


#KNN classifer -------------------------------------------------
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors =5, metric = 'minkowski', p=2 )
knn.fit(x_data_train, y_data_train)

#SVM classifer -------------------------------------------------
from sklearn.svm import SVC
svm = SVC(kernel = 'rbf', random_state = 0)
svm.fit(x_data_train, y_data_train)

#Predict test results
y_data_pred = svm.predict(x_data_test)

#Confusion Matrix to check the precision & recall
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_data_test, y_data_pred)
print (cm)

#-------------- Naive Bayes classifer ---> p(a\b) =(P(b\a) X P(a))/ P(b)
from sklearn.naive_bayes import GaussianNB
naiveb = GaussianNB()
naiveb.fit(x_data_train, y_data_train)

#Predict test results
y_data_pred = naiveb.predict(x_data_test)

#Confusion Matrix to check the precision & recall
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_data_test, y_data_pred)
print (cm)


#-------------- Descision Tree classifer
from sklearn.tree import DecisionTreeClassifier
dsctree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
dsctree.fit(x_data_train, y_data_train)

#Predict test results
y_data_pred = dsctree.predict(x_data_test)

#Confusion Matrix to check the precision & recall
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_data_test, y_data_pred)
print (cm)
    
#visualize the results
# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = x_data_train, y_data_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
#---------------**********&&&&&&&$$$$$$$$$$$$$$$$ Change the predict metod&&&&$#################*************_____
plt.contourf(X1, X2, dsctree.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
   plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('NaiveBayes Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

#----------------------------------------------------------------
X_set, y_set = x_data_test, y_data_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, dsctree.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()









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












