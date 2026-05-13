import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# Step 1 & 2 - Load Dataset (using seaborn like Naive Bayes practical)
df = sns.load_dataset('titanic')

# Step 3 & 4 - Preprocessing
df = df.dropna(subset=['age', 'fare', 'sex'])
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])   # male=1 female=0

# Independent and dependent variables
X = df[['sex', 'age', 'fare']]
y = df['survived']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)

# Step 5 - Train
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Step 6 - Predict
y_pred = logreg.predict(X_test)

# Step 7 & 8 - Evaluate
cm         = confusion_matrix(y_test, y_pred)
accuracy   = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
precision  = precision_score(y_test, y_pred)
recall     = recall_score(y_test, y_pred)

TN, FP, FN, TP = cm.ravel()

print("Confusion Matrix:\n", cm)
print("TP:", TP, "| FP:", FP)
print("TN:", TN, "| FN:", FN)
print("Accuracy:  ", round(accuracy, 4))
print("Error Rate:", round(error_rate, 4))
print("Precision: ", round(precision, 4))
print("Recall:    ", round(recall, 4))