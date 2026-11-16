# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# %%
# Load a high-dimensional dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names
print("Shape of X:", X.shape)
print("Features:", feature_names)

# %%
# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Mean:", X_scaled.mean(axis=0))
print("Standard deviation:", X_scaled.std(axis=0)) # axis=0 means calculate the result column by column.

# %%
#Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print("Original shape:", X_scaled.shape)
print("Reduced shape:", X_pca.shape)

# %%
# Examine explained variance
print("Explained variance ratio:")
print(pca.explained_variance_ratio_)
print("\nTotal variance explained:",
pca.explained_variance_ratio_.sum())

# %%
# Visualize the PCA projection
plt.figure(figsize=(8, 6))
colors = ["red", "green", "blue"]
for i, target_name in enumerate(target_names): # enumerate() gives us both the index and name
    plt.scatter(
        X_pca[y == i, 0],
        X_pca[y == i, 1],
        label=target_name,
        color=colors[i],
        alpha=0.7
    )

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Projection of Iris Dataset")
plt.legend()
plt.grid(alpha=0.3)

plt.show()

# %%
# Visualize the explained variance
pca_full = PCA()
X_pca_full = pca_full.fit_transform(X_scaled)
explained_variance = pca_full.explained_variance_ratio_
plt.figure(figsize=(8, 5))

plt.plot(
range(1, len(explained_variance) + 1),
explained_variance,
marker="o"
)
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("Explained Variance by Principal Component")
plt.xticks(range(1, len(explained_variance) + 1))
plt.grid(alpha=0.3)
plt.show()

# Visualize the explained variance
pca_full = PCA()
X_pca_full = pca_full.fit_transform(X_scaled)
explained_variance = pca_full.explained_variance_ratio_
plt.figure(figsize=(8, 5))

plt.plot(
    range(1, len(explained_variance) + 1),
    explained_variance,
    marker="o"
    )
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("Explained Variance by Principal Component")
plt.xticks(range(1, len(explained_variance) + 1))
plt.grid(alpha=0.3)
plt.show()

# Examine PCA loadings
loadings = pd.DataFrame(
pca.components_.T,
columns=["PC1", "PC2"],
index=feature_names
)
print(loadings)

plt.figure(figsize=(8, 5))

sns.heatmap(
loadings,
annot=True,
cmap="coolwarm",
center=0
)
plt.title("PCA Feature Loadings")
plt.show()