import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

columns = [
    'Sepal_Length',
    'Sepal_Width',
    'Petal_Length',
    'Petal_Width',
    'Species'
]

iris = pd.read_csv(url, names=columns)

print("IRIS DATASET")
print("Source: https://archive.ics.uci.edu/ml/datasets/iris\n")

print("First 5 Rows of Dataset:\n")
print(iris.head())


print("\nShape of Dataset:")
print(iris.shape)

# Check data types
print("\nData Types:")
print(iris.dtypes)

# Check missing values
print("\nMissing Values:")
print(iris.isnull().sum())

# Statistical summary
print("\nStatistical Summary:")
print(iris.describe())

# ---------------------------------------------
# Data Formatting
# ---------------------------------------------

# Convert datatype explicitly
iris['Sepal_Length'] = iris['Sepal_Length'].astype(float)

print("\nUpdated Data Types:")
print(iris.dtypes)

# Convert Categorical Variable to Numeric

label_encoder = LabelEncoder()

iris['Species'] = label_encoder.fit_transform(iris['Species'])

print("\nAfter Label Encoding:")
print(iris.head())

# ---------------------------------------------
# Data Normalization
# ---------------------------------------------

scaler = MinMaxScaler()

numeric_columns = [
    'Sepal_Length',
    'Sepal_Width',
    'Petal_Length',
    'Petal_Width'
]

iris[numeric_columns] = scaler.fit_transform(iris[numeric_columns])

print("\nAfter Normalization:")
print(iris.head())

# ---------------------------------------------
# Final Dataset Information
# ---------------------------------------------

print("\nFinal Dataset Info:")
print(iris.info())