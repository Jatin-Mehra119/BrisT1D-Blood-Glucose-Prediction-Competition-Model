import numpy as np
import pandas as pd
from itertools import product
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Function to create feature column names based on time intervals
def create_featuresCOLS():
    # Define ranges for hours and minutes
    hours = range(0, 6, 1)  # 0 to 5 hours
    minutes = range(0, 60, 5)  # 0 to 55 minutes in steps of 5

    # Define column names for target, group, and date
    target_col = "bg+1-00"
    group_col = "p_num"
    date_col = "time"

    # Generate column names for different data categories based on time intervals
    bg_cols = [f"bg-{i}-{j:02d}" for i, j in product(hours, minutes)][:12]  # Blood glucose columns
    insu_cols = [f"insulin-{i}-{j:02d}" for i, j in product(hours, minutes)][:12]  # Insulin columns
    carb_cols = [f"carbs-{i}-{j:02d}" for i, j in product(hours, minutes)][:12]  # Carbohydrate columns
    hr_cols = [f"hr-{i}-{j:02d}" for i, j in product(hours, minutes)][:12]  # Heart rate columns
    step_cols = [f"steps-{i}-{j:02d}" for i, j in product(hours, minutes)][:12]  # Step count columns
    cals_cols = [f"cals-{i}-{j:02d}" for i, j in product(hours, minutes)][:12]  # Calories burned columns

    # Combine all feature columns into a single list
    feature_cols = bg_cols + insu_cols + carb_cols + hr_cols + step_cols + cals_cols
    return feature_cols, target_col, group_col, date_col, bg_cols, insu_cols, carb_cols, hr_cols, step_cols, cals_cols

# Function to load data from CSV files
def load_data(train_path, test_path, subm_path):
    try:
        # Read training, testing, and submission data
        df_train = pd.read_csv(train_path, index_col='id', parse_dates=['time'])
        df_test = pd.read_csv(test_path, index_col='id', parse_dates=['time'])
        df_subm = pd.read_csv(subm_path, index_col='id')

        # Replace ':' with '-' in column names for consistency
        df_train.columns = df_train.columns.str.replace(':', '-')
        df_test.columns = df_test.columns.str.replace(':', '-')

        print("Data loaded successfully!")
        return df_train, df_test, df_subm
    except Exception as e:
        # Handle errors during data loading
        raise ValueError(f"Error loading data: {e}")

# Function to preprocess data
def preprocess_data(df_train, df_test, feature_cols, target_col, group_col, date_col, bg_cols, insu_cols, carb_cols, hr_cols, step_cols, cals_cols):
    seed = 43  # Seed for reproducibility
    Thr_NAN = 49  # Threshold for missing values

    # Interpolate and fill missing values for each column set
    for colset in [bg_cols, insu_cols, carb_cols, hr_cols, step_cols, cals_cols]:
        df_train[colset] = df_train[colset].interpolate(axis=1).fillna(method="bfill", axis=1).fillna(method="ffill", axis=1)
        df_test[colset] = df_test[colset].interpolate(axis=1).fillna(method="bfill", axis=1).fillna(method="ffill", axis=1)

    # Filter rows in training data with missing values below the threshold
    mask = df_train[feature_cols].isna().sum(axis=1) <= Thr_NAN
    df_train = df_train[mask]

    # Impute remaining missing values
    imputer = SimpleImputer()
    df_train[feature_cols] = imputer.fit_transform(df_train[feature_cols])
    df_test[feature_cols] = imputer.transform(df_test[feature_cols])

    # Add cyclical features for hour representation (sin and cos transformations)
    df_train["sin_hour"] = np.sin(np.pi * df_train[date_col].dt.hour / 12)
    df_train["cos_hour"] = np.cos(np.pi * df_train[date_col].dt.hour / 12)
    df_test["sin_hour"] = np.sin(np.pi * df_test[date_col].dt.hour / 12)
    df_test["cos_hour"] = np.cos(np.pi * df_test[date_col].dt.hour / 12)

    # Extend feature columns to include new cyclical features
    feature_cols.extend(["sin_hour", "cos_hour"])

    # Standardize features using StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(df_train[feature_cols])  # Standardize training features
    df_test_final = scaler.transform(df_test[feature_cols])  # Standardize testing features

    # Extract target variable and save preprocessed data
    y = np.array(df_train[target_col])
    np.save("preprocessed_data/X.npy", X)  # Save standardized training features
    np.save("preprocessed_data/y.npy", y)  # Save target variable
    np.save("preprocessed_data/df_test_final.npy", df_test_final)  # Save standardized test data

    print("Data preprocessed and saved successfully!")
    return X, y, df_test_final
