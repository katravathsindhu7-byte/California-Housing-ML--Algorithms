# California-Housing-ML--Algorithms
# California Housing — Machine Learning Algorithms

## Project Overview

This project applies eight machine learning algorithms to the California Housing dataset, covering both classification and regression tasks.

## Algorithms

### Classification

* KNN Classifier
* SVM Classifier (SVC)
* Decision Tree Classifier
* Random Forest Classifier

### Regression

* Decision Tree Regressor
* Random Forest Regressor
* Gradient Boosting Regressor
* XGBoost Regressor

## Dataset

California Housing dataset from Scikit-learn.

Classification models use three house-value categories: Low, Medium and High. Regression models predict continuous median house values.

## Technologies

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost and Google Colab.

## Evaluation

Classification: Accuracy, Precision, Recall and F1 Score.

Regression: MAE, MSE, RMSE and R².

## How to Run

Open the notebook in Google Colab and run the code cell.

## Results

See the classification and regression comparison CSV files for the measured results.

## Author

K. Sindhu






# ==============================================================
# CALIFORNIA HOUSING - 8 MACHINE LEARNING ALGORITHMS
# 4 Classification Models + 4 Regression Models
# Platform: Google Colab
# ==============================================================

# STEP 1: Install XGBoost and import libraries
%pip -q install xgboost

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor
)
from xgboost import XGBRegressor

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)

# STEP 2: Load the dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame.copy()

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print(df.head())

df.to_csv("california_housing.csv", index=False)

# STEP 3: Prepare features and regression target
X = df.drop(columns=["MedHouseVal"])
y_reg = df["MedHouseVal"]

# Convert continuous values into three price categories.
# Values are in units of $100,000.
# Low: < $150,000
# Medium: $150,000 to < $300,000
# High: >= $300,000
y_class = pd.cut(
    y_reg,
    bins=[-np.inf, 1.5, 3.0, np.inf],
    labels=["Low", "Medium", "High"],
    right=False
)

print("\nPrice category counts:")
print(y_class.value_counts())

# STEP 4: Split data
X_train, X_test, yr_train, yr_test = train_test_split(
    X, y_reg, test_size=0.20, random_state=42
)

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X, y_class, test_size=0.20, random_state=42, stratify=y_class
)

print("\nRegression training samples:", len(X_train))
print("Classification training samples:", len(Xc_train))

# STEP 5: Define all eight models

classification_models = {
    "KNN_Classifier": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ]),

    "SVM_Classifier": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="rbf", C=1.0))
    ]),

    "Decision_Tree_Classifier": DecisionTreeClassifier(
        max_depth=12, random_state=42
    ),

    "Random_Forest_Classifier": RandomForestClassifier(
        n_estimators=100, max_depth=15,
        random_state=42, n_jobs=-1
    )
}

regression_models = {
    "Decision_Tree_Regressor": DecisionTreeRegressor(
        max_depth=15, random_state=42
    ),

    "Random_Forest_Regressor": RandomForestRegressor(
        n_estimators=100, max_depth=20,
        random_state=42, n_jobs=-1
    ),

    "Gradient_Boosting_Regressor": GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1,
        max_depth=3, random_state=42
    ),

    "XGBoost_Regressor": XGBRegressor(
        n_estimators=100, learning_rate=0.1,
        max_depth=6, objective="reg:squarederror",
        random_state=42, n_jobs=-1
    )
}

# STEP 6: Create folders for outputs
os.makedirs("model_outputs", exist_ok=True)

classification_results = []
regression_results = []
trained_models = {}

# STEP 7: Train and evaluate classification models
for name, model in classification_models.items():
    print(f"\nTraining {name}...")

    model.fit(Xc_train, yc_train)
    predictions = model.predict(Xc_test)
    trained_models[name] = model

    accuracy = accuracy_score(yc_test, predictions)
    precision = precision_score(
        yc_test, predictions, average="weighted", zero_division=0
    )
    recall = recall_score(
        yc_test, predictions, average="weighted", zero_division=0
    )
    f1 = f1_score(
        yc_test, predictions, average="weighted", zero_division=0
    )

    classification_results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1
    })

    pd.DataFrame({
        "Actual_Category": np.asarray(yc_test),
        "Predicted_Category": predictions
    }).to_csv(
        f"model_outputs/{name}_predictions.csv", index=False
    )

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

# STEP 8: Train and evaluate regression models
for name, model in regression_models.items():
    print(f"\nTraining {name}...")

    model.fit(X_train, yr_train)
    predictions = model.predict(X_test)
    trained_models[name] = model

    mae = mean_absolute_error(yr_test, predictions)
    mse = mean_squared_error(yr_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(yr_test, predictions)

    regression_results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2_Score": r2
    })

    pd.DataFrame({
        "Actual_Value": yr_test.to_numpy(),
        "Predicted_Value": predictions,
        "Actual_Value_Dollars": yr_test.to_numpy() * 100000,
        "Predicted_Value_Dollars": predictions * 100000
    }).to_csv(
        f"model_outputs/{name}_predictions.csv", index=False
    )

    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")

# STEP 9: Save combined metrics
classification_df = pd.DataFrame(classification_results)
regression_df = pd.DataFrame(regression_results)

classification_df.to_csv(
    "classification_model_comparison.csv", index=False
)
regression_df.to_csv(
    "regression_model_comparison.csv", index=False
)

print("\n========== CLASSIFICATION COMPARISON ==========")
print(classification_df.round(4).to_string(index=False))

print("\n========== REGRESSION COMPARISON ==========")
print(regression_df.round(4).to_string(index=False))

# STEP 10: Plot classification comparison
classification_df.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1_Score"]
].plot(kind="bar", figsize=(12, 6))

plt.title("Classification Model Performance")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.show()

# STEP 11: Plot regression comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.barplot(
    data=regression_df, x="Model", y="RMSE", ax=axes[0]
)
axes[0].set_title("Regression RMSE (lower is better)")
axes[0].tick_params(axis="x", rotation=35)

sns.barplot(
    data=regression_df, x="Model", y="R2_Score", ax=axes[1]
)
axes[1].set_title("Regression R² (higher is better)")
axes[1].tick_params(axis="x", rotation=35)

plt.tight_layout()
plt.show()

# STEP 12: Plot actual vs predicted for best RMSE model
best_regression_name = regression_df.loc[
    regression_df["RMSE"].idxmin(), "Model"
]
best_model = trained_models[best_regression_name]
best_predictions = best_model.predict(X_test)

plt.figure(figsize=(8, 6))
plt.scatter(yr_test, best_predictions, alpha=0.35)

low = min(yr_test.min(), best_predictions.min())
high = max(yr_test.max(), best_predictions.max())

plt.plot([low, high], [low, high], "r--")
plt.xlabel("Actual House Value ($100,000 units)")
plt.ylabel("Predicted House Value ($100,000 units)")
plt.title(f"Best Regression Model: {best_regression_name}")
plt.tight_layout()
plt.show()

print("\nBest regression model by test RMSE:", best_regression_name)

# STEP 13: Save summary text
with open("model_outputs/project_summary.txt", "w") as f:
    f.write("California Housing ML Model Results\n\n")
    f.write("Classification Results:\n")
    f.write(classification_df.to_string(index=False))
    f.write("\n\nRegression Results:\n")
    f.write(regression_df.to_string(index=False))
    f.write(f"\n\nBest regression model by RMSE: {best_regression_name}\n")

# STEP 14: Package all output files into a ZIP
import shutil

shutil.make_archive(
    "california_housing_model_outputs",
    "zip",
    "model_outputs"
)

print("\n========== PROJECT COMPLETED ==========")
print("Files created:")
print("- california_housing.csv")
print("- classification_model_comparison.csv")
print("- regression_model_comparison.csv")
print("- california_housing_model_outputs.zip")
print("- model_outputs/ (individual prediction CSVs)")
