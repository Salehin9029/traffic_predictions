import pandas as pd
from sklearn import datasets, linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
mydataset = pd.read_csv('data.csv')
x = [[1,3],[1,3],[1,3],[1,3]]
y = [0,10,2,3]


print(y)
# fit a model
#lm = linear_model.LinearRegression()

#model = lm.fit(x, y)
#predictions = lm.predict(1,2)
model = LogisticRegression()

model.fit(x, y)
model.score(x, y)
#Equation coefficient and Intercept
print('Coefficient: \n', model.coef_)
print('Intercept: \n', model.intercept_)
#Predict Output
predicted= model.predict([[2,3]])