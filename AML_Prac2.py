# %%
#Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

# %%
#Create an Imbalanced Dataset
# Sample dataset

data = {
'Age': [25,30,35,40,45,50,22,27,29,31,33,37,39,42,48,52,55,60,24,26],
'Income': [30000,40000,50000,60000,70000,80000,28000,35000,42000,46000,
52000,58000,62000,68000,75000,82000,90000,100000,31000,36000],
'Gender': ['M','F','M','F','M','F','M','F','M','F',
'M','F','M','F','M','F','M','F','M','F'],
'Purchased': [0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,1,0,1]

}
df = pd.DataFrame(data)
print(df.head())

# %%
print(df['Purchased'].value_counts())

# %%
#Feature Engineering
#Encode Categorical Feature eg: M=1, F=0
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

# %%
#Create New Features
# Income per age ratio
df['Income_Age_Ratio'] = df['Income'] / df['Age']

# Age group
df['Age_Group'] = pd.cut(
df['Age'],

bins=[20,30,40,50,70],
labels=[1,2,3,4]
)

df['Age_Group'] = df['Age_Group'].astype(int)

print(df.head())

# %%
#Define Features and Target
X = df.drop('Purchased', axis=1)
y = df['Purchased']

#Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.2,
random_state=42,
stratify=y
)

# %%
#Apply SMOTE
from imblearn.over_sampling import SMOTE
print("Before SMOTE:")
print(y_train.value_counts())

smote = SMOTE(
random_state=42,
k_neighbors=1
)

X_train_smote, y_train_smote = smote.fit_resample(X_train,y_train)
print("\nAfter SMOTE:")
print(pd.Series(y_train_smote).value_counts())

# %%
#Train Model
model = RandomForestClassifier(
n_estimators=100,
random_state=42
)
model.fit(X_train_smote, y_train_smote)
#Prediction
y_pred = model.predict(X_test)

#Evaluation
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred, zero_division=0))



