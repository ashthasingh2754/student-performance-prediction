
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# Load dataset and train model
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

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        random_state=42
    )
    model.fit(X, y)

    return model, features


model, features = load_model()

# Application heading
st.title("🎓 Student Performance Predictor")
st.write(
    "Estimate a student's final grade using a "
    "Random Forest machine learning model."
)

st.info("Enter the student's details below. Grades use a 0–20 scale.")

# Input form
with st.form("student_form"):
    age = st.number_input(
        "Age",
        min_value=15,
        max_value=22,
        value=17
    )

    studytime = st.selectbox(
        "Weekly study time",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "Less than 2 hours",
            2: "2–5 hours",
            3: "5–10 hours",
            4: "More than 10 hours"
        }[x],
        index=1
    )

    failures = st.number_input(
        "Previous class failures",
        min_value=0,
        max_value=4,
        value=0
    )

    absences = st.number_input(
        "Number of absences",
        min_value=0,
        max_value=100,
        value=5
    )

    G1 = st.slider(
        "First period grade (G1)",
        min_value=0,
        max_value=20,
        value=12
    )

    G2 = st.slider(
        "Second period grade (G2)",
        min_value=0,
        max_value=20,
        value=13
    )

    submitted = st.form_submit_button(
        "Predict Final Grade",
        use_container_width=True
    )

# Prediction
if submitted:
    student = pd.DataFrame(
        [[age, studytime, failures, absences, G1, G2]],
        columns=features
    )

    prediction = float(model.predict(student)[0])
    prediction = max(0, min(20, prediction))

    st.subheader("Prediction Result")
    st.metric(
        label="Estimated Final Grade",
        value=f"{prediction:.2f} / 20"
    )

    if prediction >= 10:
        st.success("The estimated grade is passing (10 or above).")
    else:
        st.warning("The estimated grade is below 10.")

    st.caption(
        "This is an educational estimate, not a guarantee "
        "of actual academic performance."
    )

st.divider()
st.caption("Student Performance Prediction | Machine Learning Project")