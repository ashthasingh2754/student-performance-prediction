import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("dataset/student-mat.csv", sep=";")

# Select features
X = data[
    [
        "age",
        "studytime",
        "failures",
        "absences",
        "G1",
        "G2"
    ]
]

# Target
y = data["G3"]

# Train model using the complete dataset
model = LinearRegression()
model.fit(X, y)

print("======================================")
print("   STUDENT PERFORMANCE PREDICTOR")
print("======================================")

# Get student information
age = float(input("Enter age: "))
studytime = float(input("Enter study time (1-4): "))
failures = float(input("Enter previous failures: "))
absences = float(input("Enter number of absences: "))
G1 = float(input("Enter first period grade (G1): "))
G2 = float(input("Enter second period grade (G2): "))

# Create input DataFrame
student = pd.DataFrame(
    [[age, studytime, failures, absences, G1, G2]],
    columns=[
        "age",
        "studytime",
        "failures",
        "absences",
        "G1",
        "G2"
    ]
)

# Make prediction
prediction = model.predict(student)

print("\n======================================")
print("Predicted Final Grade (G3):",
      round(prediction[0], 2))
print("======================================")