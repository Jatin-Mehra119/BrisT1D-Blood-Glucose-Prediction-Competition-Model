import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
import mlflow
import mlflow.lightgbm
import os


# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")


def train_model(X, y, save_dir="models", params=None):
    """
    Train a LightGBM model with early stopping, log results with MLflow, 
    and save the model to a specified folder.

    Args:
        X (np.ndarray or pd.DataFrame): Features for training.
        y (np.ndarray or pd.Series): Target variable.
        save_dir (str): Directory to save the model.
        params (dict, optional): LightGBM parameters.

    Returns:
        lgb.Booster: Trained LightGBM model.
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Split the data
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create LightGBM datasets
    train_data = lgb.Dataset(train_X, label=train_y)
    test_data = lgb.Dataset(test_X, label=test_y)

    # Set default LightGBM parameters if not provided
    if params is None:
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9
        }

    
    if params is not None:
        run_name = f"LightGBM Training - {params['num_leaves']} leaves, {params['learning_rate']} lr"
    else:
        run_name = "LightGBM Training"

    # Start an MLflow experiment for training
    
    with mlflow.start_run(run_name=run_name):
        # Train the model
        gbm = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, test_data],
            callbacks=[lgb.early_stopping(stopping_rounds=10)]
        )

        # Make predictions and evaluate
        pred = gbm.predict(test_X)
        mae = mean_absolute_error(test_y, pred)
        rmse = np.sqrt(mean_squared_error(test_y, pred))

        # Log parameters and metrics to MLflow
        mlflow.log_params(params)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)

        # Log the model with input example and signature
        input_example = test_X[:5]  # Select a few rows from the test set
        signature = mlflow.models.infer_signature(test_X, pred)

        try:
            mlflow.lightgbm.log_model(
                gbm, 
                artifact_path="lightgbm_model", 
                input_example=input_example, 
                signature=signature
            )
        except Exception as e:
            print(f"Error logging model to MLflow: {e}")

        print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")

        # Save the model to the specified folder
        model_path = os.path.join(save_dir, "lightgbm_model.txt")
        gbm.save_model(model_path)

    return gbm


def tune_hyperparameters(X, y):
    """
    Perform hyperparameter tuning with GridSearchCV and log results with MLflow.

    Args:
        X (np.ndarray or pd.DataFrame): Features for training.
        y (np.ndarray or pd.Series): Target variable.

    Returns:
        dict: Best parameters found during tuning.
    """
    print("Starting hyperparameter tuning...")
    # Define parameter grid
    param_grid = {
        'num_leaves': [31, 50, 70],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [1000, 2000, 3000]
    }

    # Create LightGBM estimator
    estimator = lgb.LGBMRegressor(objective='regression', metric='rmse', boosting_type='gbdt')

    # Start an MLflow experiment for hyperparameter tuning
    with mlflow.start_run(run_name="Hyperparameter Tuning"):
        # Perform GridSearchCV
        gbm_cv = GridSearchCV(estimator, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
        gbm_cv.fit(X, y)

        # Get the best parameters and log them
        best_params = gbm_cv.best_params_
        mlflow.log_params(best_params)

        # Log the best RMSE from GridSearchCV
        best_rmse = np.sqrt(-gbm_cv.best_score_)
        mlflow.log_metric("Best RMSE", best_rmse)

        print("Best Parameters:", best_params)
        print(f"Best RMSE: {best_rmse:.4f}")

    return best_params
