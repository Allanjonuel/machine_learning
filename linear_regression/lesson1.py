from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns

iris = load_iris()
iris_data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
print(iris_data.head())
print(iris.target)
iris_data['species'] = iris.target
iris_data['species'] = iris_data['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("First five rows of the dataset:")
print(iris_data.head())

print("\nDataset dimensions:")
print(iris_data.shape)

print("\nColumn data types and null values:")
print(iris_data.info())

print("\nSummary statistics:")
print(iris_data.describe())

sns.set(style="whitegrid")

# 1. Feature Distributions
plt.figure(figsize=(10, 6))
for i, feature in enumerate(iris.feature_names):
    plt.subplot(2, 2, i + 1)
    sns.histplot(data=iris_data, x=feature, hue='species', kde=True)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()

# 2. Scatter Plot: Petal Length vs. Petal Width
plt.figure(figsize=(8, 6))
sns.scatterplot(data=iris_data, x='petal length (cm)', y='petal width (cm)', hue='species')
plt.title('Petal Length vs. Petal Width by Species')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.show()

# 3. Pair Plot
sns.pairplot(iris_data, hue='species', diag_kind='kde')
plt.show()
