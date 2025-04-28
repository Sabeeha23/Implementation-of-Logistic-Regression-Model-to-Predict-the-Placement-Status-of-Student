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

## Output:
```
import pandas as pd
df=pd.read_csv('Placement_Data.csv')
df.head()
```
![image](https://github.com/user-attachments/assets/8f39864c-6a96-4e84-9685-02d45a6f91de)

```
d1=df.copy()
d1=d1.drop(["sl_no","salary"],axis=1)
d1.head()
```

![image](https://github.com/user-attachments/assets/f4014dc5-709e-49a9-8f51-70b79b183ae2)

```
d1.isnull().sum()
```
![image](https://github.com/user-attachments/assets/7a08c4fd-fd1d-4337-96dd-929e9d19d7f7)

```
d1.duplicated().sum()
```
![image](https://github.com/user-attachments/assets/5117141f-5935-405f-b4e8-212918684637)

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
![image](https://github.com/user-attachments/assets/c893eddb-3d5e-415a-b114-1bea73a1620d)

```
x=d1.iloc[:, : -1]
x
```
![image](https://github.com/user-attachments/assets/12e98d4c-f7ca-4790-bed7-f488e4620679)

```
y=d1["status"]
y
```

![image](https://github.com/user-attachments/assets/cfe4f2eb-204b-4b25-8c3d-8b01f93d2d4d)
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=45)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(solver="liblinear")
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_pred
```

![image](https://github.com/user-attachments/assets/4a446cb6-bf8c-457e-ad9b-bd5182e157c2)
```
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy=accuracy_score(y_test,y_pred)
accuracy
```
![image](https://github.com/user-attachments/assets/6cf7030a-e268-4df3-9e70-59afb70ed0e1)
```
confusion=confusion_matrix(y_test,y_pred)
confusion
```
![image](https://github.com/user-attachments/assets/2cfaf0d2-9116-45aa-aa7f-0bb27f253290)
```
from sklearn.metrics import classification_report
classification_report=classification_report(y_test,y_pred)
print(classification_report)

print("Sabeeha Shaik")
print(212223230176)
```
![image](https://github.com/user-attachments/assets/89df9b8b-b24a-4dfe-b207-f5a05127f6e1)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
