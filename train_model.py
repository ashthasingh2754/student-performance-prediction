
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# 1. Load the dataset
data = pd.read_csv("dataset/student-mat.csv", sep=";")

print("Student Performance Dataset Loaded!")
print("Dataset shape:", data.shape)

# 2. Select features and target
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

# 3. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# 4. Define three machine learning models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(
        max_depth=5,
        random_state=42
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        random_state=42
    )
}

# 5. Train and evaluate each model
results = []

for name, model in models.items():
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": round(mae, 2),
        "MSE": round(mse, 2),
        "R2 Score": round(r2, 2)
    })

# 6. Display the comparison
results_df = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

# 7. Identify the model with the lowest MAE
best_model = results_df.loc[results_df["MAE"].idxmin()]

print("\n========== BEST MODEL ==========")
print("Model:", best_model["Model"])
print("MAE:", best_model["MAE"])
print("MSE:", best_model["MSE"])
print("R2 Score:", best_model["R2 Score"])

print("\nModel comparison completed successfully!")