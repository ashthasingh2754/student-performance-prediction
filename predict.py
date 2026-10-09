
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

# 1. Load the dataset
data = pd.read_csv("dataset/student-mat.csv", sep=";")

# 2. Select input features
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

# 3. Train the Random Forest model
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=5,
    random_state=42
)

model.fit(X, y)

print("=" * 45)
print("       STUDENT PERFORMANCE PREDICTOR")
print("          Model: Random Forest")
print("=" * 45)

# 4. Get student details
age = float(input("Enter age (15-22): "))
studytime = float(input("Enter study time (1-4): "))
failures = float(input("Enter previous failures (0 or more): "))
absences = float(input("Enter number of absences: "))
G1 = float(input("Enter first period grade G1 (0-20): "))
G2 = float(input("Enter second period grade G2 (0-20): "))

# 5. Create input DataFrame
student = pd.DataFrame(
    [[age, studytime, failures, absences, G1, G2]],
    columns=features
)

# 6. Predict final grade
prediction = model.predict(student)[0]

# Keep the displayed prediction within the grade scale
prediction = max(0, min(20, prediction))

print("\n" + "=" * 45)
print("PREDICTION RESULT")
print("=" * 45)
print(f"Predicted Final Grade (G3): {prediction:.2f} / 20")
print("=" * 45)
print("Note: This is an educational estimate, not a")
print("guarantee of a student's actual final grade.")