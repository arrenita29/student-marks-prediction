# Student Marks Prediction Using Machine Learning

## Project Description
This mini AI/ML project predicts a student's marks based on the number of study hours. The system uses the **Linear Regression** algorithm to learn from sample data and estimate marks for new input values. It demonstrates the basic concepts of Machine Learning, including data collection, model training, and prediction.

**Project Level:** Beginner (Mini Project)

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Matplotlib (for the regression plot)

## Key Features
- Takes study hours as input
- Predicts expected marks (clamped to 0-100)
- Simple and beginner-friendly AI project
- Demonstrates supervised machine learning
- Saves a Study Hours vs Marks plot with the fitted regression line

## Project Structure
```
student-marks/
├── students.csv                   # sample dataset (Hours, Marks)
├── student_marks_prediction.py    # load -> train -> evaluate -> predict
└── regression_plot.png            # generated plot
```

## How to Run
```bash
pip install pandas scikit-learn matplotlib

python student_marks_prediction.py              # interactive prediction
python student_marks_prediction.py --hours 6.5  # one-shot prediction
```

## How It Works
1. **Data collection:** read `students.csv` with Pandas (25 sample rows).
2. **Training:** split 80/20 into train and test sets, then fit `LinearRegression` on study hours.
3. **Evaluation:** measure R², MAE and RMSE on the unseen test set.
4. **Prediction:** enter study hours and get the expected marks.

## Sample Results
| Metric | Value |
|--------|-------|
| Learned equation | Marks = 8.85 x Hours + 4.85 |
| R² score | 0.998 |
| MAE | 0.81 marks |
| RMSE | 0.98 marks |
| Prediction for 6.5 hours | ~62.4 marks |

> The dataset is a small synthetic sample, so the near-perfect score reflects clean data. Replace `students.csv` with a real dataset (same `Hours`, `Marks` columns) for realistic results.

## Learning Outcomes
- Understanding Machine Learning basics
- Working with datasets
- Training and testing ML models
- Making predictions using AI algorithms

## Possible Extensions
- Add more features (attendance, previous marks) for multiple regression
- Compare with Random Forest or Polynomial Regression
- Build a Streamlit UI for live predictions
