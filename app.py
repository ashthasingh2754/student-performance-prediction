
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Performance Prediction")
st.write("Predict a student's final grade using machine learning.")


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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=5, random_state=42
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=5,
            random_state=42
        )
    }

    results = []
    trained_models = {}

    for name, algorithm in models.items():
        algorithm.fit(X_train, y_train)
        predictions = algorithm.predict(X_test)

        results.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, predictions),
            "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
            "R² Score": r2_score(y_test, predictions)
        })

        trained_models[name] = algorithm

    comparison = pd.DataFrame(results)

    return (
        trained_models["Random Forest"],
        features,
        comparison
    )


model, features, comparison = load_model()

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


# Model performance metrics
st.header("📊 Random Forest Model Performance")

rf_metrics = comparison[
    comparison["Model"] == "Random Forest"
].iloc[0]

col1, col2, col3 = st.columns(3)

col1.metric("MAE", f"{rf_metrics['MAE']:.2f}")
col2.metric("RMSE", f"{rf_metrics['RMSE']:.2f}")
col3.metric("R² Score", f"{rf_metrics['R² Score']:.3f}")

with st.expander("What do these metrics mean?"):
    st.write("""
    - **MAE:** Average absolute difference between actual and predicted grades.
      Lower is better.
    - **RMSE:** Penalizes larger prediction errors more heavily.
      Lower is better.
    - **R² Score:** Indicates how much variation in grades the model explains.
      A value closer to 1 generally indicates a better fit.
    """)


# Existing error chart
st.subheader("Random Forest Error Metrics")

fig, ax = plt.subplots()
ax.bar(
    ["MAE", "RMSE"],
    [rf_metrics["MAE"], rf_metrics["RMSE"]]
)
ax.set_ylabel("Error")
ax.set_title("Random Forest Prediction Errors")
st.pyplot(fig)
plt.close(fig)


# NEW: Model comparison section
st.divider()
st.header("🔍 Compare Machine Learning Models")

st.write(
    "All three models are evaluated on the same test dataset. "
    "Their metrics are calculated automatically."
)

display_comparison = comparison.copy()
display_comparison["MAE"] = display_comparison["MAE"].round(3)
display_comparison["RMSE"] = display_comparison["RMSE"].round(3)
display_comparison["R² Score"] = display_comparison["R² Score"].round(3)

st.dataframe(display_comparison, hide_index=True, use_container_width=True)

with st.expander("View model comparison charts"):
    chart_option = st.selectbox(
        "Choose a metric to compare",
        ["MAE", "RMSE", "R² Score"]
    )

    chart_data = comparison.set_index("Model")[chart_option]

    fig, ax = plt.subplots()
    chart_data.plot(kind="bar", ax=ax)
    ax.set_ylabel(chart_option)
    ax.set_title(f"Model Comparison: {chart_option}")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

st.caption(
    "For MAE and RMSE, lower values indicate smaller errors. "
    "For R², a value closer to 1 generally indicates a better fit."
)


# Student input form
st.divider()
st.header("📝 Predict Student Final Grade")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age", min_value=15, max_value=22, value=17
        )

        studytime = st.selectbox(
            "Weekly study time",
            options=[1, 2, 3, 4],
            format_func=lambda x: {
                1: "Less than 2 hours",
                2: "2–5 hours",
                3: "5–10 hours",
                4: "More than 10 hours"
            }[x]
        )

        failures = st.number_input(
            "Past class failures",
            min_value=0, max_value=4, value=0
        )

    with col2:
        absences = st.number_input(
            "Number of absences",
            min_value=0, max_value=100, value=0
        )

        G1 = st.number_input(
            "First period grade (0–20)",
            min_value=0, max_value=20, value=10
        )

        G2 = st.number_input(
            "Second period grade (0–20)",
            min_value=0, max_value=20, value=10
        )

    submitted = st.form_submit_button("Predict Final Grade")

if submitted:
    input_data = pd.DataFrame(
        [[age, studytime, failures, absences, G1, G2]],
        columns=features
    )

    prediction = float(model.predict(input_data)[0])
    prediction = float(np.clip(prediction, 0, 20))

    st.session_state.prediction_history.append({
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Age": age,
        "Study Time": studytime,
        "Failures": failures,
        "Absences": absences,
        "G1": G1,
        "G2": G2,
        "Predicted Grade": round(prediction, 2)
    })

    st.subheader("Prediction Result")
    st.metric("Predicted Final Grade (G3)", f"{prediction:.2f} / 20")

    if prediction >= 10:
        st.success("The predicted grade is at or above 10/20.")
    else:
        st.warning("The predicted grade is below 10/20.")

    st.caption(
        "This is a machine learning estimate, not a guaranteed actual grade."
    )


# Prediction history
st.divider()
st.header("📜 Prediction History")

if st.session_state.prediction_history:
    history_df = pd.DataFrame(st.session_state.prediction_history)
    st.dataframe(history_df, use_container_width=True)

    if st.button("Clear Prediction History"):
        st.session_state.prediction_history = []
        st.rerun()
else:
    st.info("Your prediction history will appear here after your first prediction.")


st.divider()
st.caption("Student Performance Prediction | Python, Scikit-learn & Streamlit")
