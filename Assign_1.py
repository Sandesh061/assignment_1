import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

data=pd.read_csv('cell_samples.csv')

data=data[pd.to_numeric(data['BareNuc'], errors='coerce').notnull()]
data['BareNuc']=data['BareNuc'].astype('int')

features_df=data[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize',
       'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
X=np.asarray(features_df)
y=np.asarray(data['Class'])

X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.2, random_state=4)

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


class Classifier:

    def support_vect():
        svm = SVC()
        svm.fit(X_train, y_train)
        p = svm.predict(X_test)
        print(f"Accuracy of SVM: {accuracy_score(y_test, p)*100}")

    def gausian():
        gauss = GaussianNB()
        gauss.fit(X_train, y_train)
        p = gauss.predict(X_test)
        print(f"Accuracy of Naive Bayes: {accuracy_score(y_test, p)*100}")

    def knearest(): 
        knn =  KNeighborsClassifier(n_neighbors=30)
        knn.fit(X_train,y_train)
        p = knn.predict(X_test)
        print(f"Accuracy of Knearest Neighbor: {accuracy_score(y_test, p)*100}")

while True:
    n = int(input("\nChoices for Classifier:\n \t1) Support Vector Machine\n\t2) Naive Bayes\n\t3) Knearest Neighbors\n\t:= "))
    if n == 'q':
        break 

    if n == 1:
        Classifier.support_vect()
    elif n == 2:
        Classifier.gausian()
    
    elif n == 3:
        Classifier.knearest()
    else:
        print("\tinvalid choice")
    