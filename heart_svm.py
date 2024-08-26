import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
#Reading the file
df= pd.read_csv('heart.csv')
print(df.head())

#Checking if there is null
print(df.isnull().sum())
data = df.dropna()

X = df.drop(['output'], axis=1)
y = df['output']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#Fit function used
model = SVC() # Initialize the XGBClassifier model
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

plt.figure(figsize=(10, 6))
plt.scatter(X_test['age'], X_test['chol'], c=y_pred_test, cmap='coolwarm', edgecolors='k', s=100)
plt.xlabel('Age')
plt.ylabel('Cholesterol')
plt.title('Scatter Plot of Age vs Cholesterol Colored by SVM Predictions')
plt.show()


