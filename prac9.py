import seaborn as sns
import matplotlib.pyplot as plt

titanic = sns.load_dataset('titanic')

print("Shape:", titanic.shape)
print(titanic[['sex', 'age', 'survived']].head())

# Box plot - age vs gender with survival info
sns.boxplot(x='sex', y='age', hue='survived', data=titanic)
plt.title("Age Distribution by Gender and Survival")
plt.show()