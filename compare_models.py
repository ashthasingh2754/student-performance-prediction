
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load dataset
data = pd.read_csv("dataset/student-mat.csv", sep=";")

features = [
    "age",
    "studytime",
    "failures",
    "absences",
    "G1",
    "G2"
]

X = data[features]
y = data["G3"]

# Use the same train/test split as train_model.py
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(
        max_depth=5, random_state=42
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, max_depth=5, random_state=42
    )
}

# Train and calculate MAE
mae_scores = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae_scores[name] = mean_absolute_error(y_test, predictions)

# Display scores
print("Model Comparison (MAE)")
for name, score in mae_scores.items():
    print(f"{name}: {score:.2f}")

# Plot comparison
plt.figure(figsize=(9, 5))
bars = plt.bar(mae_scores.keys(), mae_scores.values())

plt.title("Machine Learning Model Comparison")
plt.xlabel("Model")
plt.ylabel("Mean Absolute Error (Lower is Better)")
plt.bar_label(bars, fmt="%.2f", padding=3)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.show()

print("\nGraph saved as model_comparison.png")