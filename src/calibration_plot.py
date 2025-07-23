import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, roc_auc_score
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns

def calibration_plot(y_true, y_prob, n_bins=10):
    """
    Create a calibration plot with a 45-degree dashed line.
    """
    prob_true, bin_means = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    sns.set(style="whitegrid")
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
    plt.plot(bin_means, prob_true, marker='o', label="Model")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Plot")
    plt.legend(loc="best")
    plt.show()

def run_calibration_and_metrics(df_test):
    # Check required columns
    required_cols = ['y', 'pred_lr', 'pred_dt']
    for col in required_cols:
        if col not in df_test.columns:
            raise ValueError(f"Missing column in test data: {col}")

    y_true = df_test['y']
    y_prob_lr = df_test['pred_lr']  # Logistic regression predicted probabilities
    y_prob_dt = df_test['pred_dt']  # Decision tree predicted probabilities

    # Plot calibration curves
    print("Logistic Regression Calibration Plot:")
    calibration_plot(y_true, y_prob_lr, n_bins=5)

    print("Decision Tree Calibration Plot:")
    calibration_plot(y_true, y_prob_dt, n_bins=5)

    print("Which model is more calibrated? Based on the calibration plots, describe your answer here.")

    # -------- Extra Credit --------
    # Compute PPV for top 50 highest risk arrestees predicted by logistic regression
    top50_lr = df_test.nlargest(50, 'pred_lr')
    ppv_lr = precision_score(top50_lr['y'], np.round(top50_lr['pred_lr']))

    # Compute PPV for top 50 highest risk arrestees predicted by decision tree
    top50_dt = df_test.nlargest(50, 'pred_dt')
    ppv_dt = precision_score(top50_dt['y'], np.round(top50_dt['pred_dt']))

    # Compute AUC for both models
    auc_lr = roc_auc_score(y_true, y_prob_lr)
    auc_dt = roc_auc_score(y_true, y_prob_dt)

    print(f"PPV for top 50 Logistic Regression predictions: {ppv_lr:.3f}")
    print(f"PPV for top 50 Decision Tree predictions: {ppv_dt:.3f}")
    print(f"AUC for Logistic Regression model: {auc_lr:.3f}")
    print(f"AUC for Decision Tree model: {auc_dt:.3f}")

    print("Do both metrics agree that one model is more accurate than the other? Explain your answer here.")
