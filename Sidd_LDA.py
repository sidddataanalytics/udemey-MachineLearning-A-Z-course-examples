# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 21:43:17 2020

@author: 320001866
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r'C:\Sid Data\BITS\4th Sem\Udemey\Part 9 - Dimensionality Reduction\Section 43 - Principal Component Analysis (PCA)\Wine.csv')
# Ned to get age & salary correspondence as X
x_data = dataset.iloc[:, 0:13].values
#print (x_data)
#x_data1 = dataset.iloc[:, :-1].values

row,columns = dataset.shape
column_index = columns-1
y_data =dataset.iloc[:,column_index].values
#print(y_data)

# Split Train & test set
from sklearn.model_selection import train_test_split
x_data_train, x_data_test, y_data_train,y_data_test = train_test_split(x_data, y_data, test_size =0.2, random_state = 0)    

#scale the data for running algorithms
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_data_train = sc_x.fit_transform(pd.DataFrame(x_data_train))
x_data_test = sc_x.fit_transform(pd.DataFrame(x_data_test))
#print (x_data)
#print (x_data_train)

#Applying LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda =  LDA(n_components =2)
x_data_train = lda.fit_transform(x_data_train, y_data_train)
x_data_test = lda.transform(x_data_test)


#Logistic Regression
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(random_state =0)
log_reg.fit(x_data_train, y_data_train)

#Predict test results
y_data_pred = log_reg.predict(x_data_test)

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
plt.contourf(X1, X2, log_reg.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
   plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

#----------------------------------------------------------------
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, log_reg.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()