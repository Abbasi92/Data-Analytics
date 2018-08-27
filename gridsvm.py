import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


parameters = {'kernel': ('linear', 'rbf','poly'), 'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000],'gamma': [1e-7, 1e-3, 1e-5, 1e-6, 1e-2, 1e-4]}
clf = GridSearchCV(SVR(), parameters)

df=pd.read_csv('merge8.csv')

X=df.drop(['Values'],1)
y=np.array(df['Values'])



data = pd.get_dummies(X)
x=data.iloc[:,:].values



X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2)


clf.fit(X_train,y_train)
score=clf.best_params_
print (score)