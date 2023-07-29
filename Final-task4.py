# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 08:06:15 2023

@author: anura
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

# 1. Read the data
def load_data(path: str = "C:\College\GalaG.csv"):
    df = pd.read_csv(f"{path}")
    #df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
    return df


def create_target_and_predictors(
    data: pd.DataFrame = None, 
    target: str = "estimated_stock_pct"
):
    
    '''
    Function Description:
    Splits the data into the target variable and the other features to train and run the model.
    
    Function Requires:
    Pandas Dataframe
    
    Function returns:
    X- Set of features that is used to predict the outcome
    y-Variable to predict

    '''
    data=load_data()
    # Check to see if the target variable is present in the data
    if target not in data.columns:
        raise Exception(f"Target: {target} is not present in the data")
    
    X = data.drop(columns=[target])
    y = data[target]
    return X, y

# 2. Train algorithm
def train_algorithm_with_cross_validation(
    X: pd.DataFrame = None, 
    y: pd.Series = None,
    K=5
):
    """
    Function Description:
    Obtains the Dependent(Target variable) and independent variables(Predictors).
    Model used: Random forest.
    Performance is enhanced and measured by cross-validation and performance metrics.
    
    Function Requires:
    X: pandas dataframe with the cleaned and analysed predictors
    y: Target variable
    K-K fold: Number of times the iteration has to be performed

    Function Returns:
    None
    
    """
    X,y=create_target_and_predictors()
    # Create a list that will store the accuracies of each fold
    accuracy = []

    # Enter a loop to run K folds of cross-validation
    for fold in range(0, K):

        # Instantiate algorithm and scaler
        model = RandomForestRegressor()
        scaler = StandardScaler()

        # Create training and test samples
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=42)

        # Brings all the data to a single numerical range to avoid partial and incorrect data.
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # Train model
        trained_model = model.fit(X_train, y_train)

        # Generate predictions on test sample
        y_pred = trained_model.predict(X_test)

        # Compute accuracy, using mean absolute error
        mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
        accuracy.append(mae)
        print(f"Fold {fold + 1}: MAE = {mae:.3f}")

    # Finish by computing the average MAE across all folds
    print(f"Average MAE: {(sum(accuracy) / len(accuracy)):.2f}")


if __name__ =="__main__":
    path="C:\College\GalaG.csv"
    train_algorithm_with_cross_validation()



