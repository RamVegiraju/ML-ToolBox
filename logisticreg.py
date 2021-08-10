import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

df = pd.read_csv("data/titanic.csv")

#Data Preprocessing
def imputeAge(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age
df['Age'] = df[['Age','Pclass']].apply(imputeAge,axis=1)
df.drop('Cabin',axis=1,inplace=True)
df.dropna(inplace=True)
sex = pd.get_dummies(df['Sex'], drop_first = True)
embark = pd.get_dummies(df['Embarked'], drop_first = True)
df = pd.concat([df,sex,embark],axis=1)
df.drop(['Sex','Embarked','Name','Ticket','PassengerId'], axis=1, inplace=True)

#Split Data
X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

#Create Model + Predictions
logModel = LogisticRegression() 
logModel.fit(X_train,y_train) 
predictions = logModel.predict(X_test)
print(confusion_matrix(y_test,predictions))

