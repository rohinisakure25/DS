import seaborn as sns
import matplotlib.pyplot as plt

# Load titanic dataset
dataset = sns.load_dataset('titanic')

# Basic info
print("Shape:", dataset.shape)
print(dataset.head())

# Histogram - fare distribution
sns.histplot(dataset['fare'], kde=False, bins=10)
plt.title("Fare Distribution of Titanic Passengers")
plt.xlabel("Fare")
plt.ylabel("Number of Passengers")
plt.show()