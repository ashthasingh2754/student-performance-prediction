# Student Performance Prediction

A machine learning project that predicts students' final academic grades using student performance data. It includes model comparison, performance evaluation, data visualization, and an interactive Streamlit web application.

## Project Overview

This project uses the UCI Student Performance dataset containing information about 395 students. Machine learning models are trained to predict the final grade (G3).

### Input Features
- Age
- Study time
- Previous failures
- Number of absences
- First period grade (G1)
- Second period grade (G2)

**Target Variable:** Final grade (G3)

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Git and GitHub

## Features

- Predict student final grades using machine learning.
- Enter student information through an interactive Streamlit web app.
- Compare Linear Regression, Decision Tree, and Random Forest models.
- Evaluate model performance using MAE, RMSE, and R².
- Visualize student data and model performance.

## Machine Learning Model Comparison

The following results are from the model comparison performed in this project. Verify that these values match your actual evaluation output before publishing them.

| Model | MAE | MSE | R² Score |
|---|---:|---:|---:|
| Linear Regression | 1.33 | 4.46 | 0.78 |
| Decision Tree | 1.18 | 4.25 | 0.79 |
| Random Forest | 1.06 | 2.92 | 0.86 |

**Best Model:** Random Forest, based on the results shown above.

## Performance Metrics

- **MAE (Mean Absolute Error):** Average absolute difference between actual and predicted scores.
- **RMSE (Root Mean Squared Error):** Square root of the mean squared prediction error. Lower values indicate smaller prediction errors.
- **R² Score:** Measures how much variation in the target scores is explained by the model. A score closer to 1 generally indicates a better fit.

## How to Run the Streamlit App

1. Clone the repository.
2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the application:

   ```bash
   python -m streamlit run app.py
   ```

4. Open the local URL displayed in the terminal.
5. Enter the student details and view the predicted final grade.

## Project Structure

```text
student-performance-prediction/
├── dataset/
│   ├── student_data.csv
│   └── student-mat.csv
├── app.py
├── train_model.py
├── visualize.py
├── predict.py
├── model_comparison.png
├── requirements.txt
├── README.md
└── .gitignore
```


## Future Improvements

- Improve prediction accuracy through hyperparameter tuning.
- Add more relevant student features.
- Deploy the Streamlit app online.
