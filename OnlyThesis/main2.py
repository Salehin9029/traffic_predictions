import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


url = "data.csv"
names = ['sourceid', 'dstid', 'hod', 'standard_deviation_travel_time', 'geometric_mean_travel_time', 'geometric_standard_deviation_travel_time',
]
data = pd.read_csv(url, names=names)
print(data.shape)
df = pd.DataFrame(data.mean_travel_time, columns=names)
y = data.target
# create training and testing vars
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)
print X_train.shape, y_train.shape
print X_test.shape, y_test.shape