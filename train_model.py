import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load the new UCI dataset
data = pd.read_csv("dataset/student-mat.csv", sep=";")

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)

# 2. Select input features
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

# 3. Target variable
y = data["G3"]

# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# 5. Create the model
model = LinearRegression()

# 6. Train the model
model.fit(X_train, y_train)

print("\nModel training completed!")

# 7. Make predictions
predictions = model.predict(X_test)

# 8. Evaluate the model
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Performance")
print("-----------------")
print("Mean Absolute Error:", round(mae, 2))
print("Mean Squared Error:", round(mse, 2))
print("R2 Score:", round(r2, 2))

# 9. Test the model with a new student
new_student = pd.DataFrame(
    [[17, 2, 0, 5, 12, 13]],
    columns=[
        "age",
        "studytime",
        "failures",
        "absences",
        "G1",
        "G2"
    ]
)

predicted_score = model.predict(new_student)



print("\nNew Student Prediction")
print("----------------------")
print("Predicted Final Grade (G3):",
      round(predicted_score[0], 2))