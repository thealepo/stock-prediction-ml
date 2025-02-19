import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

def svr_train_predict(x_train , y_train , x_test):
    # Scale the data
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()

    x_train_scale = scaler_X.fit_transform(x_train)
    x_test_scale = scaler_X.transform(x_test)
    y_train_scale = scaler_Y.fit_transform(y_train.values.reshape(-1,1))

    # Creating the SVR Model
    model = SVR()

    # Defining hyperparameters
    param_grid = {
        'kernel': ['linear' , 'rbf' , 'sigmoid'],
        'C': [1 , 10],
        'gamma': ['scale']
    }

    # Performing Grid Search & getting the Best Model according to the hyperparameters
    grid_search = GridSearchCV(estimate= model , param_grid= param_grid , cv= 3 , n_jobs= -1, verbose= 2)
    grid_search.fit(x_train_scale , y_train_scale.ravel())

    model = grid_search.best_estimator_

    # Predict the test data
    predictions_scale = model.predict(x_test_scale)
    predictions = scaler_Y.inverse_transform(predictions_scale.reshape(-1,1))

    return predictions


