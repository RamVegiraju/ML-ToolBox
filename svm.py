import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('data/mushrooms.csv')

#Preprocessing
def encodeCategorical(data):
    labelencoder=LabelEncoder()
    for col in data.columns:
        data[col] = labelencoder.fit_transform(data[col])
    return data

def prepareData(df):
    X = df.iloc[:,1:23]
    y = df.iloc[:, 0]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
    return X_train, X_test, y_train, y_test
df = encodeCategorical(df)
X_train, X_test, y_train, y_test = prepareData(df)

#Model Building & Predictions
def SVM(X_train, X_test, y_train, y_test, kernelType):
    svclassifier = SVC(kernel=kernelType)
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)
    confusionMatrix = confusion_matrix(y_test,y_pred)
    classificationReport = classification_report(y_test,y_pred)
    return confusionMatrix, classificationReport

#Linear
confusionMatrixSVM, classificationReportSVM = SVM(X_train, X_test, y_train, y_test, 'linear')
print("--------------")
print("Linear Kernel")
print("--------------")
print(confusionMatrixSVM)
print(classificationReportSVM)

#Polynomial
print("--------------")
print("Polynomial Kernel")
print("--------------")
confusionMatrixSVM, classificationReportSVM = SVM(X_train, X_test, y_train, y_test, 'poly')
print(confusionMatrixSVM)
print(classificationReportSVM)


#RBF
print("--------------")
print("RBF Kernel")
print("--------------")
confusionMatrixSVM, classificationReportSVM = SVM(X_train, X_test, y_train, y_test, 'rbf')
print(confusionMatrixSVM)
print(classificationReportSVM)