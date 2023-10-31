# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages.
2. Analyse the data.
3. Use modelselection and Countvectorizer to preditct the values. 
4. Find the accuracy and display the result.

## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: A.Anbuselvam
RegisterNumber:212222240009

import chardet
file = '/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result

import pandas as pd 
data = pd.read_csv("/content/spam.csv",encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values

y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train,y_train)

y_pred = svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy  
```
## Output:
Result:

![image](https://github.com/anbuselvamA/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119559871/a4d6763b-0093-4ada-bd4d-2944b6e2947e)

Data.head():

![image](https://github.com/anbuselvamA/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119559871/9e3b27d0-9e5f-4a93-9181-14cdc9f3b2a5)

data.info():

![image](https://github.com/anbuselvamA/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119559871/27afdb72-6b5a-4223-a893-fa7eee5a463e)

data.isnull().sum():

![image](https://github.com/anbuselvamA/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119559871/1e5fb1fd-4b81-4a85-8327-d639c050235a)

Y prediction value:

![image](https://github.com/anbuselvamA/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119559871/f5c410a8-a9ad-4cc9-b912-a1b646084dd0)

Accuracy value:

![image](https://github.com/anbuselvamA/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119559871/84660e82-ab4b-4be4-94c9-7daea04c17d1)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
