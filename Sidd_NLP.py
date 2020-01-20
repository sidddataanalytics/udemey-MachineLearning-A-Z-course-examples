# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import datase
datset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

#clean the data
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#keep only letters a to z
review = re.sub('[^a-zA-z]',' ', datset['Review'][0]);
#convert to lower case
review = review.lower()

#remove the stop words

review = review.split() # this will convert review into list with diffrent wo4ds\
print (review)
ps = PorterStemmer()
review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
#join the words into a string seperated by space
review = ' '.join(review)

#do for all the reviews
corpus = []
for i in  range(0,1000):
    review = re.sub('[^a-zA-z]',' ', datset['Review'][i]);
    #convert to lower case
    review = review.lower()

    #remove the stop words

    review = review.split() # this will convert review into list with diffrent wo4ds\
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    #join the words into a string seperated by space
    review = ' '.join(review)   
    #add it back tot he review
    corpus.append(review)
    
#bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
matrix = cv.fit_transform(corpus).toarray()

# as the matrix is very sparse...remove the non relevant words
cv = CountVectorizer(max_features = 1500)
matrix_x = cv.fit_transform(corpus).toarray()

# add depednet varuabe whihc is likedor not
y = datset.iloc[:,1].values

#use naive bayes
#split the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(matrix_x, y, test_size = 0.2, random_state = 0)

#use naive bayes
from sklearn.naive_bayes import GaussianNB
classifer = GaussianNB()
classifer.fit(x_train, y_train)

y_pred = classifer.predict(x_test)

from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test, y_pred)

#Use descison tree
from sklearn.tree import DecisionTreeClassifier
descTree = DecisionTreeClassifier()
descTree.fit(x_train, y_train)
y_pred = descTree.predict(x_test)
cm = confusion_matrix(y_test, y_pred)

from sklearn.ensemble import RandomForestClassifier
randTree = RandomForestClassifier()
randTree.fit(x_train, y_train)
y_pred = randTree.predict(x_test)
cm2 = confusion_matrix(y_test, y_pred)

print (cm)
print (cm1)
print (cm2)





    
    
    







