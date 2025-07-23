# logistic_regression.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression

def run_logistic_regression(df_arrests):
    # Train-test split
    df_arrests_train, df_arrests_test = train_test_split(
        df_arrests,
        test_size=0.3,
        shuffle=True,
        stratify=df_arrests["y"],
        random_state=42
    )

    # Select features
    features = ["num_fel_arrests_last_year", "current_charge_felony"]

    # Parameter grid for logistic regression
    param_grid = {
        "C": [0.01, 0.1, 1]
    }

    # Initialize logistic regression model
    lr_model = LogisticRegression(solver="liblinear")

    # GridSearchCV with 5-fold cross-validation
    gs_cv = GridSearchCV(estimator=lr_model, param_grid=param_grid, cv=5)
    gs_cv.fit(df_arrests_train[features], df_arrests_train["y"])

    # Optimal C value and explanation
    optimal_C = gs_cv.best_params_["C"]
    if optimal_C == min(param_grid["C"]):
        reg_strength = "most"
    elif optimal_C == max(param_grid["C"]):
        reg_strength = "least"
    else:
        reg_strength = "middle"

    print(f"Optimal C value: {optimal_C}")
    print(f"This value corresponds to the {reg_strength} amount of regularization.")

    # Predict on test set
    df_arrests_test["pred_lr"] = gs_cv.predict(df_arrests_test[features])

    # Save for later steps if needed
    df_arrests_train.to_csv("data/arrests_train.csv", index=False)
    df_arrests_test.to_csv("data/arrests_test.csv", index=False)

    return df_arrests_train, df_arrests_test
