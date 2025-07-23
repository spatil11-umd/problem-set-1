'''
You will run this problem set from main.py, so set things up accordingly
'''

import pandas as pd
import etl
import preprocessing
import logistic_regression
from decision_tree import run_decision_tree
import calibration_plot
#pip install matplotlib (run it)
#pip install seaborn

def main():
    # PART 1: Run ETL
    etl.run_etl()
   
    # PART 2: Pre-processing and get the arrests dataframe
    df_arrests = preprocessing.preprocess()

    # PART 3: Logistic Regression training and prediction
    df_arrests_train, df_arrests_test = logistic_regression.run_logistic_regression(df_arrests)

    # PART 4: Decision Tree training and prediction
    df_arrests_train, df_arrests_test = run_decision_tree(df_arrests_train, df_arrests_test)
    
    # PART 5: Calibration and metrics using test data
    calibration_plot.run_calibration_and_metrics(df_arrests_test)

if __name__ == "__main__":
    main()
