import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load the dataset
data = pd.read_csv("dataset/student_data.csv")

print("Dataset loaded successfully!")
print(data.head())

# 2. Select input features and target
X = data[
    [
        "hours_studied",
        "attendance",
        "previous_score",
        "sleep_hours",
        "extracurricular"
    ]
]

y = data["final_score"]

# 3. Split the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# 4. Create the Linear Regression model
model = LinearRegression()

# 5. Train the model
model.fit(X_train, y_train)

print("\nModel training completed!")

# 6. Make predictions
predictions = model.predict(X_test)

# 7. Evaluate the model
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Performance")
print("-----------------")
print("Mean Absolute Error:", round(mae, 2))
print("Mean Squared Error:", round(mse, 2))
print("R2 Score:", round(r2, 2))

# 8. Test the model with a new student
new_student = [[6, 90, 78, 7, 1]]

predicted_score = model.predict(new_student)

print("\nNew Student Prediction")
print("----------------------")
print("Predicted Final Score:", round(predicted_score[0], 2))