
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import explained_variance_score, mean_squared_error, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier



data = pd.read_csv('breastCancer.csv')
data = data[data['bare_nucleoli']!='?']
data['bare_nucleoli'] = data['bare_nucleoli'].astype(int)
data.drop(['mitoses', 'id'], axis=1, inplace=True)
df_x = data[['clump_thickness', 'size_uniformity', 'shape_uniformity',
       'marginal_adhesion', 'epithelial_size', 'bare_nucleoli',
       'bland_chromatin', 'normal_nucleoli']] 

df_y = data["class"]
########################### splitting data into train and test ###############################
Xtrain, Xtest, ytrain, ytest = train_test_split(df_x.values, df_y.values,random_state=1, )
model = LogisticRegression(fit_intercept = True)
model.fit(Xtrain,ytrain)
pred_y = model.predict(Xtest)
acc = accuracy_score(ytest, pred_y)
print("accuracy of the model: "+ str(acc))
print("score of the cross validation: " + str(cross_val_score(model,df_x,df_y, cv=10).mean()))
print("Matrix of confusion: \n" + str(confusion_matrix(ytest, pred_y)))

clf = GradientBoostingClassifier(n_estimators=300, learning_rate=0.1, max_depth=4, random_state=0).fit(Xtrain, ytrain)
pred_y = clf.predict(Xtest)
print("The clf score with Gradient boost  is", clf.score(Xtest, ytest))
print("Matrix of confusion: \n" + str(confusion_matrix(ytest, pred_y)))
