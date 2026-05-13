import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the dataset using a URL
# This is a public version of the Boston Housing dataset
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)

# Convert column names to uppercase to match your original code logic 
df.columns = df.columns.str.upper()

# Step 2: Exploratory Data Analysis
print("First 5 rows:")
print(df.head()) 
print("\nDataset Info:")
df.info()

# Step 3: Data Preprocessing
# Define features (X) and target variable (y)
X = df.drop('MEDV', axis=1) 
y = df['MEDV']

# Split into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle missing values by filling them with the mean of the training set
X_train_imputed = X_train.fillna(X_train.mean()) 
X_test_imputed = X_test.fillna(X_train.mean()) 

# Step 4: Train the Model
model = LinearRegression()
model.fit(X_train_imputed, y_train)

# Display coefficients for each feature
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
}) 
print("\nModel Coefficients:")
print(coefficients.sort_values(by='Coefficient', ascending=False))

# Step 5: Prediction and Evaluation
y_pred = model.predict(X_test_imputed) 

mse = mean_squared_error(y_test, y_pred) 
r2 = r2_score(y_test, y_pred) 
rmse = np.sqrt(mse) 

print(f"\nMean Squared Error: {mse:.2f}") 
print(f"R-squared: {r2:.2f}") 
print(f"RMSE: {rmse:.2f}")
print(f"Intercept: {model.intercept_}") 

# Step 6: Visualization
plt.figure(figsize=(8,6)) 
plt.scatter(y_test, y_pred) 
plt.xlabel("Actual Prices") 
plt.ylabel("Predicted Prices") 
plt.title("Actual vs Predicted Prices") 
plt.show() 