# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 13:43:00 2021

@author: MUZI Manzui
"""


import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

#Read the data
df=pd.read_csv('C:\\Users\\MUZI Manzui\\Desktop\\FND\\news.csv')
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


from tkinter import *
root = Tk()
root.title('FAKE NEWS DETECTION ')
root.geometry('530x300+200+80')
root.configure(bg='light green')
root.iconbitmap('C:/Users/MUZI Manzui/Desktop/vv.ico')
#############################################################  Labels
IntroLabel = Label(root,text='FAKE NEWS DETECTION',font=('new roman',30,'italic bold'),bg='white',width=22)
IntroLabel.place(x=0,y=0)

EntryLabel = Label(root,text=' Statement:-  ',font=('arial',20,'italic bold'),bg='light green')
EntryLabel.place(x=10,y=70)

FormatLabel = Label(root,text='   RESULT:-  ',font=('arial',20,'italic bold'),bg='light green')
FormatLabel.place(x=10,y=150)



############################################################# Entry
countrydata = StringVar()
ent1 = Entry(root,textvariable=countrydata,font=('arial',20,'italic bold'),relief=RIDGE,bd=2,width=20)
ent1.place(x=220,y=70)
############################################################  Buttons

def submit():
    global pr
    X_test=countrydata.get()
    dt=pd.DataFrame({"value":[X_test]})
    dt=count_vector.transform(dt)
    predictions = naive_bayes.predict(dt)
    pr=predictions[0]
    ref={0:"FAKE",1:"REAL"}
    Label(root,text=str(ref[pr]),font=15).place(x=240,y=150)
    

    

Real = Button(root,text='Result',bg='red',font=('arial',15,'italic bold'),relief=RIDGE,activebackground='green',
              activeforeground='white',command=submit,
                bd=5,width=10)
Real.place(x=110,y=250)
root.mainloop()
