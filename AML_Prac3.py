# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# %%
# Load the dataset
from sklearn.datasets import load_iris

if __name__ == "__main__":
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target

print(X.head())
print(y[:5])

# %%
#Split the data
X_train, X_test, y_train, y_test = train_test_split(
X, y,
test_size=0.2,
random_state=42,
stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# %%
#Create the model
rf = RandomForestClassifier(random_state=42)

#Grid Search
#Grid Search tries every combination of the specified hyperparameters.
#This dictionary contains the parameters that GridSearchCV will try.
param_grid = {
'n_estimators': [50, 100, 200], #Number of decision trees in the Random Forest 50trees,100 trees,200 trees
'max_depth': [None, 5, 10, 20],
'min_samples_split': [2, 5, 10], #Minimum samples required to split a node.
'min_samples_leaf': [1, 2, 4] #Minimum samples required in a leaf node.
}

grid_search = GridSearchCV(
estimator=rf,
param_grid=param_grid,
cv=5,
scoring='accuracy',
n_jobs=1 #This speeds up training
)

grid_search.fit(X_train, y_train)

#Display the best parameters
print("Best Parameters:")
print(grid_search.best_params_)
print("\nBest Cross-Validation Score:")
print(grid_search.best_score_)

#Evaluate Grid Search model
best_grid_model = grid_search.best_estimator_
y_pred = best_grid_model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#Randomized Search
param_distributions = {
    'n_estimators': [50, 100, 200, 300, 500],
    'max_depth': [None, 5, 10, 15, 20, 30],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 6],
    'max_features': ['sqrt', 'log2', None]
    }

random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_distributions,
    n_iter=20,
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=1
    )

random_search.fit(X_train, y_train)

#Display the best parameters
print("Best Parameters:")
print(random_search.best_params_)
print("\nBest Cross-Validation Score:")
print(random_search.best_score_)

#Evaluate Randomized Search model
best_random_model = random_search.best_estimator_
y_pred_random = best_random_model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred_random))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_random))

#Compare Grid Search and Random Search
print("Grid Search")
print("-" * 40)
print("Best Parameters:", grid_search.best_params_)
print("CV Accuracy:", grid_search.best_score_)
print("Test Accuracy:",accuracy_score(y_test,best_grid_model.predict(X_test)))

print("\nRandomized Search")
print("-" * 40)
print("Best Parameters:", random_search.best_params_)
print("CV Accuracy:", random_search.best_score_)
print("Test Accuracy:",accuracy_score(y_test,best_random_model.predict(X_test)))
