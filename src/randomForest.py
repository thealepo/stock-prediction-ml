import pandas as pd
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.ensemble import RandomForestRegressor

def randomForest_train_predict(x_train , y_train , x_test):

    # Creating the Model
    model = RandomForestRegressor()

    # Defining my hyperparameters
    param_grid = {
        'n-estimators': [50 , 100],
        'max_depth': [None , 10],
        'max_features': ['log2' , 'sqrt'],
        'min_samples_split': [2 , 5 , 10],
        'min_samples_leaf': [1 , 2 , 4],
        'criterion': ['mse' , 'squared_error']
    }

    # use GridSearchCV to get the Best Model with the best possible hyperparameters
    grid_search = GridSearchCV(estimator= model , param_grid= param_grid , cv=3 , verbose=2)


