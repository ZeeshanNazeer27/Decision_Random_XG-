import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
#Reading the file
df= pd.read_csv('t20.csv')
print(df.head())

#Checking if there is null
print(df.isnull().sum())
data = df.dropna()

df.head()
df.tail()

df.dtypes
df["Winner"].unique()
df["Winner"].value_counts()
label_encoder = LabelEncoder()
print(label_encoder)
# Fit the LabelEncoder on all unique values of 'Winner' to ensure consecutive labels
#all_winners = df["Winner"].unique()
#label_encoder.fit(all_winners) 

df["Winner"] = label_encoder.fit_transform(df["Winner"])
df["Date"] = label_encoder.fit_transform(df["Date"])
df["Venue"] = label_encoder.fit_transform(df["Venue"])
df["Bat_First"] = label_encoder.fit_transform(df["Bat_First"])
df["Bat_Second"] = label_encoder.fit_transform(df["Bat_Second"])



X = df[['Date','Venue','Bat_First','Bat_Second']]
y = df['Winner']

y_new = y.dropna()
X_new = X.dropna()

X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.2, random_state=10)
#Fit function used
model = XGBClassifier() # Initialize the XGBClassifier model
model.fit(X_new, y_new)
y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)

#Prediction for the given data
# Adjust the input for prediction to match the number of features (4 in this case)
prediction = model.predict([[108, 150, 22, 0]])  
print("Prediction:", prediction)

#Accuracy Score
score = accuracy_score(y_train, y_pred_train)
print("Model Accuracy Train:", score)
score_test = accuracy_score(y_test, y_pred_test)
print("Model Accuracy Test:", score_test)

