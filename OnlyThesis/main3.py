import pandas as pd
from sklearn import datasets, linear_model
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

mydataset = pd.read_csv('data.csv')
x = mydataset.iloc[:-1,:-1].values
y = mydataset.iloc[:-1,3:].values
y=y.astype('int')
print(y.shape)
print(y)
print(x)
print(x.shape)
# fit a model
lm = linear_model.LinearRegression()

y_test = [[189],[294],[2218]]

x_test = [[1,2,1],[1,2,5],[1,9,10]]

model = lm.fit(x, y)
predictions = lm.predict(x_test)


predictions=predictions.astype('float')



print(predictions)
#decision tree
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=3, min_samples_leaf=5)
model = clf_gini.fit(x, y)

predict = clf_gini.predict(x_test)

print(predict)
#knn
knn = KNeighborsClassifier()

result = knn.fit(x,y.ravel())

predict_knn = knn.predict(x_test)

print (predict_knn)

print "Accuracy is ", accuracy_score(y_test, predictions)*100