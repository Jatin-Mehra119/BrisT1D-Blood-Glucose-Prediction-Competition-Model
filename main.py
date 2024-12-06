from src.preprocessing import create_featuresCOLS, load_data, preprocess_data
from src.train_model import train_model, tune_hyperparameters
import pandas as pd


# File paths
TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
SUBMISSION_PATH = "data/sample_submission.csv"


def main():
    """
    Main function for loading data, training models, and generating predictions.
    """
    # Load and preprocess data
    df_train, df_test, df_subm = load_data(TRAIN_PATH, TEST_PATH, SUBMISSION_PATH)

    feature_cols, target_col, group_col, date_col, bg_cols, insu_cols, carb_cols, hr_cols, step_cols, cals_cols = create_featuresCOLS()
    
    X, y, df_test_final = preprocess_data(
        df_train, 
        df_test, 
        feature_cols, 
        target_col, 
        group_col, 
        date_col, 
        bg_cols, 
        insu_cols, 
        carb_cols, 
        hr_cols, 
        step_cols, 
        cals_cols
    )

    # Train and evaluate the model without tuning
    print("Training model without hyperparameter tuning...")
    gbm = train_model(X, y)

    # Generate predictions for submission
    results = gbm.predict(df_test_final)
    df_subm['bg+1:00'] = results
    submission_path_no_tuning = 'Submission_LGB_No_Tuning.csv'
    df_subm.to_csv(submission_path_no_tuning, index=True)
    print(f"Submission file saved: {submission_path_no_tuning}")

    # Tune hyperparameters
    print("Starting hyperparameter tuning...")
    best_params = tune_hyperparameters(X, y)

    # Train the model with the best hyperparameters
    print("Training model with fine-tuned hyperparameters...")
    best_model = train_model(X, y, params=best_params)

    # Generate predictions for submission
    results = best_model.predict(df_test_final)
    df_subm['bg+1:00'] = results
    submission_path_tuned = 'Submission_LGB_Fine_Tuned.csv'
    df_subm.to_csv(submission_path_tuned, index=True)
    print(f"Submission file saved: {submission_path_tuned}")


if __name__ == "__main__":
    main()
