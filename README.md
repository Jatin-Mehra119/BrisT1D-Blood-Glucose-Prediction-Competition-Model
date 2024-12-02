
# Blood Glucose Prediction Using Continuous Monitoring Data

This repository contains the solution for the Kaggle competition **Blood Glucose Prediction**. The goal is to predict blood glucose levels one hour into the future using a dataset collected from young adults in the UK with type 1 diabetes. Participants used continuous glucose monitors (CGMs), insulin pumps, and smartwatches to collect data aggregated into five-minute intervals.

### Competetion:[NoteBook](https://github.com/Jatin-Mehra119/BrisT1D-Blood-Glucose-Prediction-Notebook/blob/main/COMP-NoteBook.ipynb)

## Dataset Description

The dataset consists of:
- **Training data**: Samples from the first three months of the study for 9 participants, including features and the target blood glucose value (`bg+1:00`).
- **Testing data**: Samples from the later study period for 15 participants, excluding the target value, with no overlap with the training data and in a randomized order.

### Key Features
- **Medical Data**: Includes blood glucose readings, insulin doses, carbohydrate intake, heart rate, step count, calorie burn, and activity labels for six-hour historical windows.
- **Challenges**:
  - Missing values and noise.
  - Different device models used by participants.
  - Test data contains unseen participants.

### Data Files
- `activities.txt`: List of activity labels.
- `train.csv`: Training data with features and target.
- `test.csv`: Test data with features only.
- `sample_submission.csv`: Sample submission file format.

## Solution Overview

### Key Components
1. **Preprocessing**:
   - Handles missing values using interpolation and imputation.
   - Adds cyclical features for time of day (sine and cosine transformations).
   - Standardizes the dataset for machine learning.

2. **Model Training**:
   - Trains a LightGBM regression model to predict blood glucose levels.
   - Supports hyperparameter tuning via grid search.
   - Logs training results and metrics using MLflow.

3. **Outputs**:
   - **Preprocessed Data**: For faster loading in subsequent runs.
     - `X.npy`: Preprocessed features for training.
     - `y.npy`: Preprocessed target labels.
     - `df_test_final.npy`: Preprocessed test data features.
   - **Model**: Saved LightGBM model in the `models/` folder.
   - **Submission Files**:
     - `Submission_LGB_No_Tuning.csv`: Predictions from the baseline model.
     - `Submission_LGB_Fine_Tuned.csv`: Predictions from the fine-tuned model.


## How to Run

1. **Set Up Environment**:
   - Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```
   - Start an MLflow tracking server:
     ```bash
     mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
     ```

2. **Run the Pipeline**:
   - Preprocess data, train models, and generate submissions:
     ```bash
     python main.py
     ```

3. **Outputs**:
   - Preprocessed data saved in the `preprocessed_data/` folder.
   - Trained LightGBM model saved in the `models/` folder.
   - Submission files saved in the project root directory.

## Results
- **Baseline Model**:
  - Mean Absolute Error (MAE): *Logged in MLflow*.
  - Root Mean Squared Error (RMSE): *Logged in MLflow*.
- **Fine-Tuned Model**:
  - Achieves improved performance with hyperparameter optimization.

## Future Work
- Explore advanced models like recurrent neural networks (RNNs) or transformers for time-series prediction.
- Implement feature selection and importance analysis for better interpretability.
- Experiment with additional imputation methods for missing data.
