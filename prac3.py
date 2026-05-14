# Import Required Libraries
import pandas as pd
import numpy as np

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

columns = [
    'Sepal_Length',
    'Sepal_Width',
    'Petal_Length',
    'Petal_Width',
    'Species'
]

iris = pd.read_csv(url, names=columns)

print("First 5 Rows of Dataset:\n")
print(iris.head())

# ---------------------------------------------------
# 1. Summary Statistics Grouped by Categorical Variable
# ---------------------------------------------------

print("\nSummary Statistics Grouped by Species:\n")

grouped_data = iris.groupby('Species')

summary = grouped_data['Sepal_Length'].agg([
    'mean',
    'median',
    'min',
    'max',
    'std'
])

print(summary)

# ---------------------------------------------------
# Create List of Numeric Values for Each Category
# ---------------------------------------------------

print("\nList of Sepal Length Values for Each Species:\n")

species_lists = grouped_data['Sepal_Length'].apply(list)

print(species_lists)

# ---------------------------------------------------
# 2. Statistical Details for Each Species
# ---------------------------------------------------

species_names = iris['Species'].unique()

for species in species_names:

    print("\n====================================")
    print(f"Statistics for {species}")
    print("====================================")

    species_data = iris[iris['Species'] == species]

    print(species_data.describe())