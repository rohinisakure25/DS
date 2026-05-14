import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = sns.load_dataset('iris')
print("Dataset Info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())

# --- Task 2: Histograms (Feature Distributions) ---
# Visualizes the spread and skewness of individual numerical features
plt.figure(figsize=(10, 8))
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
for i, feature in enumerate(features):
    plt.subplot(2, 2, i+1)
    sns.histplot(df[feature], kde=True, color='skyblue')
    plt.title(f'Distribution of {feature}')

plt.tight_layout()
plt.show()

# --- Task 3: Boxplots (Identify Outliers & Variation) ---
# Compares features across species and visually identifies data points outside whiskers
plt.figure(figsize=(10, 8))
for i, feature in enumerate(features):
    plt.subplot(2, 2, i+1)
    sns.boxplot(x='species', y=feature, data=df)
    plt.title(f'{feature} by Species')

plt.tight_layout()
plt.show()

# --- Task 4: Compare Distributions & Identify Outliers (Statistical) ---
print("\n--- Inferences (IQR Method) ---")
for feature in features:
    # Calculate IQR to mathematically identify outliers
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]

    if not outliers.empty:
        print(f"Feature '{feature}' has {len(outliers)} total outliers.")
    else:
        print(f"Feature '{feature}' has no significant outliers.")
