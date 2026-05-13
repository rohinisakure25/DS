import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import seaborn as sns

# Step 1 & 2: Load Dataset
df = sns.load_dataset('iris')

# Step 3 & 4: Data Preprocessing
X = df.drop('species', axis=1) # Independent variables
y = df['species'] # Dependent variable

# Split into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Step 5: Train the Model
model = GaussianNB()
model.fit(X_train, y_train)

# Step 6: Prediction
y_pred = model.predict(X_test)

# Step 7 & 8: Evaluation
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy

# For multi-class, we use 'weighted' averaging
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

# Output Results
print("Confusion Matrix:\n", cm)
print("Accuracy:  ", round(accuracy, 4))
print("Error Rate:", round(error_rate, 4))
print("Precision: ", round(precision, 4))
print("Recall:    ", round(recall, 4))