import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Dataset URL
csv_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

# Loading raw data without headers
iris_raw = pd.read_csv(csv_url, header=None)
print("Dataset without column names ")
print(iris_raw.head(), "\n")

# Loading data with defined column names
col_names = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width', 'Species']
iris = pd.read_csv(csv_url, names=col_names)

# Basic Inspection
print("Basic Inspection")
print(f"Shape: {iris.shape}")
print(f"Columns: {iris.columns.tolist()}")
print(f"Data Types: \n{iris.dtypes}\n")

# Slice Examples
print("Slice Examples")

# Selecting the 6th row (index 5)
print("6th row using iloc:\n", iris.iloc[5], "\n")

# Selecting specific columns for the first three rows
print("First 3 rows of specific columns:\n", iris.loc[:2, ['Sepal_Length', 'Petal_Length']], "\n")

# Label Encoding for the categorical 'Species' column
label_encoder = LabelEncoder()
iris['Species_Encoded'] = label_encoder.fit_transform(iris['Species'])

print("After Label Encoding (Added 'Species_Encoded')")
print(iris[['Species', 'Species_Encoded']].head(), "\n")

# Min-Max Scaling for numeric columns
scaler = MinMaxScaler()
numeric_cols = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']
iris[numeric_cols] = scaler.fit_transform(iris[numeric_cols])

# Final result inspection
print("Final Result (Scaled Features + Encoded Labels)")
print(iris.head())
