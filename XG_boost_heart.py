import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
#Reading the file
df= pd.read_csv('heart.csv')
print(df.head())

#Checking if there is null
print(df.isnull().sum())
data = df.dropna()

X = df.drop(['output'], axis=1)
y = df['output']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
#Fit function used
model = XGBClassifier() 
model.fit(X_train, y_train)
y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)

#Prediction for the given data
prediction = model.predict([[67,1,0,160,286,0,0,108,1,1.5,1,3,2]])  
print("Prediction:", prediction)

#Accuracy Score
score = accuracy_score(y_train, y_pred_train)
print("Model Accuracy Train:", score)
score_test = accuracy_score(y_test, y_pred_test)
print("Model Accuracy Test:", score_test)

