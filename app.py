
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# ==========================================
# 2. LOAD DATA, TRAIN MODEL, EVALUATE MODEL
# ==========================================
@st.cache_resource
def load_model():
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

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Create and train Random Forest
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Evaluate using held-out test data
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    return model, features, metrics


model, features, metrics = load_model()

# ==========================================
# 3. APPLICATION HEADER
# ==========================================
st.title("🎓 Student Performance Predictor")

st.write(
    "Predict a student's final academic grade using "
    "a Random Forest machine learning model."
)

st.caption(
    "Dataset: UCI Student Performance | "
    "Prediction scale: 0–20"
)

st.divider()

# ==========================================
# 4. MODEL PERFORMANCE METRICS
# ==========================================
st.subheader("📊 Model Performance")

st.write(
    "The following metrics are calculated on a held-out "
    "test set that was not used to train the model."
)

col1, col2, col3 = st.columns(3)

col1.metric(
    label="MAE",
    value=f"{metrics['MAE']:.2f}"
)

col2.metric(
    label="RMSE",
    value=f"{metrics['RMSE']:.2f}"
)

col3.metric(
    label="R² Score",
    value=f"{metrics['R2']:.2f}"
)

with st.expander("What do these metrics mean?"):
    st.write(
        "**MAE (Mean Absolute Error):** Average absolute "
        "difference between predicted and actual grades. "
        "Lower is better."
    )

    st.write(
        "**RMSE (Root Mean Squared Error):** Measures "
        "prediction error while penalizing larger errors "
        "more heavily. Lower is better."
    )

    st.write(
        "**R² Score:** Measures how much variation in final "
        "grades the model explains on the test set. A value "
        "closer to 1 generally indicates a better fit."
    )

st.divider()

# ==========================================
# 5. STUDENT INPUT FORM
# ==========================================
st.subheader("📝 Enter Student Details")

st.info(
    "Enter the student's details below and click "
    "'Predict Final Grade'."
)

with st.form("student_form"):

    age = st.number_input(
        "Age",
        min_value=15,
        max_value=22,
        value=17,
        step=1
    )

    studytime = st.selectbox(
        "Weekly Study Time",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "Level 1 - Less than 2 hours",
            2: "Level 2 - 2 to 5 hours",
            3: "Level 3 - 5 to 10 hours",
            4: "Level 4 - More than 10 hours"
        }[x],
        index=1
    )

    failures = st.number_input(
        "Previous Class Failures",
        min_value=0,
        max_value=4,
        value=0,
        step=1
    )

    absences = st.number_input(
        "Number of Absences",
        min_value=0,
        max_value=100,
        value=5,
        step=1
    )

    G1 = st.slider(
        "First Period Grade (G1)",
        min_value=0,
        max_value=20,
        value=12
    )

    G2 = st.slider(
        "Second Period Grade (G2)",
        min_value=0,
        max_value=20,
        value=13
    )

    submitted = st.form_submit_button(
        "Predict Final Grade",
        use_container_width=True
    )

# ==========================================
# 6. PREDICTION RESULT
# ==========================================
if submitted:

    student = pd.DataFrame(
        [[age, studytime, failures, absences, G1, G2]],
        columns=features
    )

    prediction = float(model.predict(student)[0])

    # Keep displayed grade within the dataset's 0–20 scale
    prediction = max(0.0, min(20.0, prediction))

    st.divider()
    st.subheader("🎯 Prediction Result")

    st.metric(
        label="Estimated Final Grade (G3)",
        value=f"{prediction:.2f} / 20"
    )

    if prediction >= 10:
        st.success(
            "The predicted grade is 10 or above."
        )
    else:
        st.warning(
            "The predicted grade is below 10."
        )

    st.caption(
        "This is an educational estimate, not a guarantee "
        "of the student's actual final grade."
    )

# ==========================================
# 7. FOOTER
# ==========================================
st.divider()

st.caption(
    "Student Performance Prediction | "
    "Python • Pandas • Scikit-learn • Streamlit"
)
