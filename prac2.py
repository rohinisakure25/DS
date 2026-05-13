import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# ── CREATE DATASET ────────────────────────────────────
data = {
    'Math_Score':      [72, 85, 60, 90, 78, 55, 88, 92, np.nan, 70],
    'Reading_Score':   [80, 88, 75, 95, 82, 78, 91, 85, 79,     np.nan],
    'Writing_Score':   [70, 80, 65, 88, 75, 60, 85, 90, 72,     68],
    'Placement_Score': [80, 92, 75, 98, 85, 70, 95, 99, 78,     82],
    'Club_Join_Date':  [2018, 2019, 2020, 2018, 2021, 2019, 2020, 2021, 2018, 2020],
    'Placement_Offer_Count': [1, 3, 1, 4, 2, 1, 3, 4, 2, 2]
}
df = pd.DataFrame(data)
print("Original Dataset:")
print(df)

# ── STEP 1: MISSING VALUES ────────────────────────────
print("\nMissing Values:")
print(df.isnull().sum())

# Fill missing values with mean
df['Math_Score'].fillna(df['Math_Score'].mean(), inplace=True)
df['Reading_Score'].fillna(df['Reading_Score'].mean(), inplace=True)

print("\nAfter Filling Missing Values:")
print(df.isnull().sum())

# ── STEP 2: OUTLIERS ──────────────────────────────────
# Add an outlier for demonstration
df.loc[0, 'Math_Score'] = 150   # artificial outlier

# Detect using boxplot
df[['Math_Score','Reading_Score','Writing_Score','Placement_Score']].boxplot()
plt.title("Boxplot - Outlier Detection")
plt.show()

# Detect using Z-score
from scipy import stats
z_scores = stats.zscore(df['Math_Score'])
print("\nZ-scores for Math_Score:")
print(z_scores)

# Remove outliers (z-score > 2)
df = df[abs(z_scores) <= 2]
print("\nAfter Removing Outliers:")
print(df['Math_Score'])

# ── STEP 3: DATA TRANSFORMATION ──────────────────────
# 1. Change scale - convert Club_Join_Date to Duration
current_year = 2024
df['Duration'] = current_year - df['Club_Join_Date']
print("\nClub Duration (years):")
print(df[['Club_Join_Date', 'Duration']])

# 2. Log transformation to reduce skewness
df['Math_Score_Log'] = np.log(df['Math_Score'])
print("\nAfter Log Transformation:")
print(df[['Math_Score', 'Math_Score_Log']])

# Check skewness before and after
print("\nSkewness before log:", round(df['Math_Score'].skew(), 2))
print("Skewness after log: ", round(df['Math_Score_Log'].skew(), 2))

# 3. Plot before and after
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
df['Math_Score'].hist(ax=axes[0])
axes[0].set_title('Before Log Transform')
df['Math_Score_Log'].hist(ax=axes[1])
axes[1].set_title('After Log Transform')
plt.show()