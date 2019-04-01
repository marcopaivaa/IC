import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier



df_dataset = pd.read_csv('iris.csv', sep=',', index_col=None)
df_dataset.shape
df_dataset.head(n=5)
df_dataset.describe()

X = df_dataset.iloc[:,:-1].values
y = df_dataset["classe"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=2019) # 70% treino e 30% teste

#####################################################################################################################

print('# PADRÃO')

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("\tAcurácia:", metrics.accuracy_score(y_test, y_pred))
print("\tTamanho do dataset: ",len(X_train))


X_train = X_train.tolist()
y_train = y_train.tolist()

#####################################################################################################################

print('# INSERÇÃO')

new_X = []
new_y = []

for index,value in enumerate(X_train):
    if(len(new_X) < 5 ):
        new_X.append(X_train[index])
        new_y.append(y_train[index])
    else:
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(new_X, new_y)
        y_pred = knn.predict([X_train[index]])
        if y_pred[0] != y_train[index]:
            new_X.append(X_train[index])
            new_y.append(y_train[index])

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(new_X, new_y)
y_pred = knn.predict(X_test)
print("\tAcurácia:", metrics.accuracy_score(y_test, y_pred))
print("\tTamanho do dataset: ",len(new_X))

#####################################################################################################################

print('# REMOÇÃO')

new_X = X_train
new_y = y_train

for index,value in enumerate(X_train):
    if(len(new_X) >= 5 ):
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(new_X, new_y)
        y_pred = knn.predict([X_train[index]])
        if y_pred[0] == y_train[index]:
            new_X.remove(X_train[index])
            new_y.remove(y_train[index])

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(new_X, new_y)
y_pred = knn.predict(X_test)
print("\tAcurácia:", metrics.accuracy_score(y_test, y_pred))
print("\tTamanho do dataset: ",len(new_X))