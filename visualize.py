import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the UCI Student Performance dataset
data = pd.read_csv("dataset/student-mat.csv", sep=";")

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)

# 1. Study Time vs Final Grade
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=data,
    x="studytime",
    y="G3"
)

plt.title("Study Time vs Final Grade")
plt.xlabel("Study Time")
plt.ylabel("Final Grade (G3)")
plt.tight_layout()
plt.show()

# 2. Absences vs Final Grade
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=data,
    x="absences",
    y="G3"
)

plt.title("Absences vs Final Grade")
plt.xlabel("Number of Absences")
plt.ylabel("Final Grade (G3)")
plt.tight_layout()
plt.show()

# 3. Previous Grade (G2) vs Final Grade (G3)
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=data,
    x="G2",
    y="G3"
)

plt.title("Previous Grade (G2) vs Final Grade (G3)")
plt.xlabel("Second Period Grade (G2)")
plt.ylabel("Final Grade (G3)")
plt.tight_layout()
plt.show()

print("\nVisualization completed successfully!")