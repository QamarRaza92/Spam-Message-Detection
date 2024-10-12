import pandas as pd
import numpy as np

pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
df = pd.read_csv("C:\\Users\\moham\\Downloads\\JaipurFinalCleanData.csv")
print(df.shape)


x = [1 if df.iloc[x+1,37]>0 else 0 for x in range(0,df.shape[0]-1)]
x.append(0)
df['RainToday'] = x
print(df.head(20))


import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


X_train,X_test,y_train,y_test = train_test_split(df.drop(columns=['RainToday','date'],axis=1),df['RainToday'],random_state=0,test_size=0.2)


scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = SVC()
model.fit(X_train,y_train)
ypred = model.predict(X_test)
print('accuracy score is : ',accuracy_score(y_test,ypred))


from sklearn.decomposition import PCA
import time
start = time.time()
# for k in range(1,40):
#     model = PCA(n_components=k)
#     X_train_copy = model.fit_transform(X_train)
#     X_test_copy = model.transform(X_test)
#     model2 = RandomForestClassifier()
#     model2.fit(X_train_copy,y_train)
#     ypred = model2.predict(X_test_copy)
#     print('accuracy score for PCA {a} is : {b}'.format(a=k,b=accuracy_score(y_test,ypred)))
# print(f"eigen value : {model.explained_variance_}\neigen vector : {model.components_}")
# print("time taken : ",time.time()-start)


model = PCA(n_components=7)
X_train_copy = model.fit_transform(X_train)
X_test_copy = model.transform(X_test)
model2 = RandomForestClassifier()
model2.fit(X_train_copy,y_train)
ypred = model2.predict(X_test_copy)
print('accuracy score after PCA is : {b}%'.format(b=np.round(100*(accuracy_score(y_test,ypred)),2)))


print(pd.DataFrame(X_train_copy).columns.values)


