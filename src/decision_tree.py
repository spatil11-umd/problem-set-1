import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

def run_decision_tree(df_arrests_train, df_arrests_test):
    # Define features and target
    features = ['num_fel_arrests_last_year', 'current_charge_felony']
    target = 'y'

    # Check if expected columns exist
    for col in features + [target]:
        if col not in df_arrests_train.columns:
            print(f"Missing column in training data: {col}")
            return None, None

    # Set up parameter grid for max_depth
    param_grid = {"max_depth": [3, 5, 10]}

    # Initialize decision tree classifier
    dt = DecisionTreeClassifier(random_state=42)

    # Use GridSearchCV to find the best max_depth using 5-fold CV
    grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy')

    # Train the model
    grid_search.fit(df_arrests_train[features], df_arrests_train[target])

    # Show best parameters
    print("Best max_depth found:", grid_search.best_params_['max_depth'])
    print("Most regularization corresponds to shallower trees (smaller max_depth).")
    print("Least regularization corresponds to deeper trees (larger max_depth).")

    # Add predictions to test dataframe
    df_arrests_test['pred_dt'] = grid_search.predict(df_arrests_test[features])

    # Optionally save updated data
    df_arrests_train.to_csv('data/df_arrests_train.csv', index=False)
    df_arrests_test.to_csv('data/df_arrests_test.csv', index=False)

    return df_arrests_train, df_arrests_test
