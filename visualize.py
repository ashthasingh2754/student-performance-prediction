import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv("dataset/student_data.csv")

# Display basic information
print("Dataset loaded successfully!")
print(data.describe())

# 1. Study Hours vs Final Score
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=data,
    x="hours_studied",
    y="final_score"
)

plt.title("Study Hours vs Final Score")
plt.xlabel("Hours Studied")
plt.ylabel("Final Score")
plt.tight_layout()
plt.show()

# 2. Attendance vs Final Score
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=data,
    x="attendance",
    y="final_score"
)

plt.title("Attendance vs Final Score")
plt.xlabel("Attendance (%)")
plt.ylabel("Final Score")
plt.tight_layout()
plt.show()

# 3. Previous Score vs Final Score
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=data,
    x="previous_score",
    y="final_score"
)

plt.title("Previous Score vs Final Score")
plt.xlabel("Previous Score")
plt.ylabel("Final Score")
plt.tight_layout()
plt.show()

print("\nVisualization completed successfully!")