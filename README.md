# Student Performance Prediction

A Python Machine Learning project that predicts a student's final score based on academic performance data.

## Features

- Loads student performance data from CSV
- Preprocesses the dataset
- Splits data into training and testing sets
- Trains a Linear Regression model
- Evaluates model performance
- Predicts the final score of a new student

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Machine Learning
- Linear Regression

## Project Structure

student-performance-prediction/
│
├── dataset/
│   └── student_data.csv
│
├── train_model.py
│
├── README.md
│
└── .gitignore

## Model Performance

Mean Absolute Error: 0.36

Mean Squared Error: 0.14

R2 Score: 1.0

## Example Prediction

The model predicts a final score of approximately:

82.07

for the sample student provided in the program.

## How to Run

1. Clone the repository.
2. Open the project in VS Code.
3. Create and activate a Python virtual environment.
4. Install the required libraries.
5. Run:

```bash
python train_model.py