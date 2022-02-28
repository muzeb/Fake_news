# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 13:03:34 2021

@author: MUZI Manzui
"""

import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

#Read the data
df=pd.read_csv('D:\\DataFlair\\news.csv')
df.head()
df["label"].head()
df['label'] = df.label.map({'FAKE':0, 'REAL':1})
df["label"].head()
df=df.iloc[0:,[2,3]]
df.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['text'],
                                                    df['label'],
                                                    random_state=1)
from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer() #set the variable
train = count_vector.fit_transform(X_train)
test = count_vector.transform(X_test)
from sklearn.naive_bayes import MultinomialNB
naive_bayes = MultinomialNB() #call the method
naive_bayes.fit(train, y_train) #train the classifier on the training set
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)



val='Daniel Greenfield, a Shillman Journalism Fellow at the Freedom Center, is a New York writer focusing on radical Islam.'
dt=pd.DataFrame({"value":[val]})
dt=count_vector.transform(dt)



from sklearn.naive_bayes import MultinomialNB
naive_bayes = MultinomialNB() #call the method
naive_bayes.fit(train, y_train) #train the classifier on the training set
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
predictions = naive_bayes.predict(dt)
predictions

