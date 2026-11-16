# %%
#XGBoost-XGBoost builds a series of decision trees sequentially, where each new tree tries tocorrect the errors made by the previous trees. machine learning algorithm based on the GradientBoosting technique. It is widely used for classification, regression, and ranking problems

!pip install xgboost
import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from xgboost import XGBClassifier

# %%
#load dataset
iris = load_iris()

X = pd.DataFrame(
iris.data,
columns=iris.feature_names
)

y = iris.target

print("Dataset shape:", X.shape)
display(X.head())

# %%
#Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.20,
random_state=42,
stratify=y
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# %%
#Create XGBoost Model
model = XGBClassifier(
n_estimators=100,
learning_rate=0.1,

max_depth=3,
subsample=0.8,
colsample_bytree=0.8,
random_state=42,
eval_metric='mlogloss'
)

# %%
#Train the Model
model.fit(X_train, y_train)

print("XGBoost model trained successfully!")

# %%
#Make Predictions
y_pred = model.predict(X_test)

print("Predicted values:")
print(y_pred)

# %%
#Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("XGBoost Accuracy:", accuracy)

# %%
#Classification Report
print(classification_report(
y_test,
y_pred,
target_names=iris.target_names
))

# %%
#Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

# %%
#Feature Importance
importance = pd.Series(

model.feature_importances_,
index=X.columns
).sort_values(ascending=False)

print("Feature Importance:")
display(importance)

# %%
#Plot Feature Importance
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))

importance.sort_values().plot(
kind='barh',
color='steelblue'
)

plt.title("XGBoost Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()

# %%
#Predict a New Sample
new_sample = np.array([
[5.1, 3.5, 1.4, 0.2]
])

prediction = model.predict(new_sample)

print("Predicted class:",
iris.target_names[prediction[0]])

# %%



