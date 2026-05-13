import pandas as pd
import seaborn as sns

# Load the dataset [cite: 1, 2]
df = sns.load_dataset('tips')

# Display the first 5 rows [cite: 3]
print(df.head(5))

# Group by 'day' and calculate summary statistics for 'total_bill' [cite: 5]
grouped_stats = df.groupby('day')['total_bill'].agg(['mean', 'median', 'min', 'max', 'std'])
print("Summary Statistics of Bill by Day: ")
print(grouped_stats)

# Create a numeric mapping for the 'day' column [cite: 22]
day_mapping = {day: i for i, day in enumerate(df['day'].unique())}
df['day_numeric'] = df['day'].map(day_mapping)

# Display the mapping results [cite: 24, 25]
print("\nFirst 5 rows with numeric day mapping:")
print(df[['day', 'day_numeric']].head())

# Filter data into Smokers and Non-Smokers [cite: 37, 39]
smokers = df[df['smoker'] == 'Yes']
non_smokers = df[df['smoker'] == 'No']

# Print descriptive statistics for tips among smokers [cite: 40, 41]
print("Statistics for Smokers:")
print(smokers['tip'].describe())

# Print descriptive statistics for tips among non-smokers [cite: 59, 60]
print("Statistics for Non-Smokers:")
print(non_smokers['tip'].describe())