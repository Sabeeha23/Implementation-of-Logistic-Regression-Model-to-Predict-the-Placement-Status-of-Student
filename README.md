# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the Dataset

2.Create a Copy of the Original Data

3.Drop Irrelevant Columns (sl_no, salary)

4.Check for Missing Values

5.Check for Duplicate Rows

6.Encode Categorical Features using Label Encoding

7.Split Data into Features (X) and Target (y)

8.Split Data into Training and Testing Sets

9.Initialize and Train Logistic Regression Model

10.Make Predictions on Test Set

11.Evaluate Model using Accuracy Score

12.Generate and Display Confusion Matrix

13.Generate and Display Classification Report

14.Make Prediction on a New Sample Input

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Sabeeha Shaik
RegisterNumber:  212223230176
*/
```
```
import pandas as pd
df=pd.read_csv('Placement_Data.csv')
df.head()
```
```
d1=df.copy()
d1=d1.drop(["sl_no","salary"],axis=1)
d1.head()
```
```
d1.isnull().sum()
```
```
d1.duplicated().sum()
```
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
d1['gender']=le.fit_transform(d1["gender"])
d1["ssc_b"]=le.fit_transform(d1["ssc_b"])
d1["hsc_b"]=le.fit_transform(d1["hsc_b"])
d1["hsc_s"]=le.fit_transform(d1["hsc_s"])
d1["degree_t"]=le.fit_transform(d1["degree_t"])
d1["workex"]=le.fit_transform(d1["workex"])
d1["specialisation"]=le.fit_transform(d1["specialisation"])
d1["status"]=le.fit_transform(d1["status"])
d1
```
```
x=d1.iloc[:, : -1]
x
```
```
y=d1["status"]
y
```
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=45)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(solver="liblinear")
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_pred
```
```
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy=accuracy_score(y_test,y_pred)
accuracy
```
```
confusion=confusion_matrix(y_test,y_pred)
confusion
```
```
from sklearn.metrics import classification_report
classification_report=classification_report(y_test,y_pred)
print(classification_report)

print("Sabeeha Shaik")
print(212223230176)
```
## Output:
## data.head()
![image](https://github.com/user-attachments/assets/318c781a-735f-4d3e-b91d-d988e813e1b6)

## data1.head()
![image](https://github.com/user-attachments/assets/f1dcf64b-1bd5-49b7-97a0-c33ef471f897)

## isnull()
![image](https://github.com/user-attachments/assets/8415e44c-7161-4821-9492-17779cefebe0)

## duplicated()
![image](https://github.com/user-attachments/assets/21083706-9610-41d6-85de-37a39232bd02)

## data1
![image](https://github.com/user-attachments/assets/807edf14-d8cf-4b4b-a5bb-39e54cd5c3e7)

## X
![image](https://github.com/user-attachments/assets/eaf43efa-be27-4763-993b-9cc66d778ef9)

## y
![image](https://github.com/user-attachments/assets/9bda52a1-094f-4bd7-9cee-b7d8e218c3b1)

## y_pred
![image](https://github.com/user-attachments/assets/fce17131-757c-434f-a2e4-b885d515ad63)

## confusion matrix
![image](https://github.com/user-attachments/assets/3b245a8a-3820-4c46-9947-959466b2d1e8)

## classification report
![image](https://github.com/user-attachments/assets/282485ca-a147-45ed-94b0-7aec07a30baf)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
