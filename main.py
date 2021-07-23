import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, plot_confusion_matrix, plot_precision_recall_curve, precision_recall_curve
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

train= pd.read_csv('data/train.csv')
train.drop(['PassengerId', 'Name', 'Cabin'],axis= 1, inplace=True)
train['Age'].fillna(train['Age'].mean(), inplace= True)
train['Embarked'].fillna('S', inplace= True)

x_col= [x for x in train.columns if x not in ['Survived']]
x= train[x_col]
y= train['Survived']
from sklearn.preprocessing import LabelEncoder

l_enc= LabelEncoder()

obj_col= x.select_dtypes('object').columns

for col in obj_col:
    x[col]= l_enc.fit_transform(x[col])

x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.2)


dt_reg= DecisionTreeClassifier()
dt_reg.fit(x_train, y_train)

plot_confusion_matrix(dt_reg, x_test, y_test, cmap= plt.cm.Blues)
plt.savefig('conf_matrix.png')
plot_precision_recall_curve(dt_reg, x_test, y_test)
plt.savefig('pr_curve.png')


y_pred= dt_reg.predict(x_test)

print(accuracy_score(y_pred, y_test))
