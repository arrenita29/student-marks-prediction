"""
Student Marks Prediction using Linear Regression
-------------------------------------------------
Predicts a student's marks from the number of study hours.

Run:
    python student_marks_prediction.py            # train, evaluate, interactive prediction
    python student_marks_prediction.py --hours 6.5  # one-shot prediction
"""

import argparse

import matplotlib
matplotlib.use("Agg")  # works without a display
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

DATA_FILE = "students.csv"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print("Dataset preview:")
    print(df.head(), "\n")
    print(f"Shape: {df.shape}  |  Missing values: {int(df.isna().sum().sum())}\n")
    return df


def train_model(df: pd.DataFrame):
    X = df[["Hours"]]  # 2D feature matrix
    y = df["Marks"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Model trained.")
    print(f"  Equation : Marks = {model.coef_[0]:.2f} x Hours + {model.intercept_:.2f}")
    print(f"  R2 score : {r2_score(y_test, y_pred):.3f}")
    print(f"  MAE      : {mean_absolute_error(y_test, y_pred):.2f} marks")
    print(f"  RMSE     : {mean_squared_error(y_test, y_pred) ** 0.5:.2f} marks\n")
    return model, X_train, X_test, y_train, y_test


def plot_results(model, df, out="regression_plot.png"):
    plt.figure(figsize=(7, 4.5))
    plt.scatter(df["Hours"], df["Marks"], label="Students", color="#2563eb")
    line = model.predict(df[["Hours"]].sort_values("Hours"))
    plt.plot(df["Hours"].sort_values(), line, color="#dc2626", label="Regression line")
    plt.xlabel("Study hours")
    plt.ylabel("Marks")
    plt.title("Study Hours vs Marks")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Plot saved to {out}\n")


def predict_marks(model, hours: float) -> float:
    hours_df = pd.DataFrame({"Hours": [hours]})  # keeps feature name, avoids sklearn warning
    marks = float(model.predict(hours_df)[0])
    return max(0.0, min(100.0, marks))  # marks can't go outside 0-100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, help="Study hours to predict marks for")
    args = parser.parse_args()

    df = load_data(DATA_FILE)
    model, *_ = train_model(df)
    plot_results(model, df)

    if args.hours is not None:
        print(f"Predicted marks for {args.hours} hours: {predict_marks(model, args.hours):.1f}")
        return

    print("Enter study hours to predict marks (type 'q' to quit).")
    while True:
        text = input("Study hours: ").strip()
        if text.lower() in {"q", "quit", "exit"}:
            break
        try:
            hours = float(text)
            if hours < 0 or hours > 24:
                print("  Please enter a value between 0 and 24.")
                continue
            print(f"  Predicted marks: {predict_marks(model, hours):.1f}")
        except ValueError:
            print("  Please enter a valid number.")


if __name__ == "__main__":
    main()
