import numpy as np
import pandas as pd
from itertools import product
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def create_featuresCOLS():
    hours = range(0, 6, 1)
    minutes = range(0, 60, 5)

    target_col = "bg+1-00"
    group_col = "p_num"
    date_col = "time"

    bg_cols = [f"bg-{i}-{j:02d}" for i, j in product(hours, minutes)][:12]
    insu_cols = [f"insulin-{i}-{j:02d}" for i, j in product(hours, minutes)][:12]
    carb_cols = [f"carbs-{i}-{j:02d}" for i, j in product(hours, minutes)][:12]
    hr_cols = [f"hr-{i}-{j:02d}" for i, j in product(hours, minutes)][:12]
    step_cols = [f"steps-{i}-{j:02d}" for i, j in product(hours, minutes)][:12]
    cals_cols = [f"cals-{i}-{j:02d}" for i, j in product(hours, minutes)][:12]

    feature_cols = bg_cols + insu_cols + carb_cols + hr_cols + step_cols + cals_cols
    return feature_cols, target_col, group_col, date_col, bg_cols, insu_cols, carb_cols, hr_cols, step_cols, cals_cols

def load_data(train_path, test_path, subm_path):
    try:
        df_train = pd.read_csv(train_path, index_col='id', parse_dates=['time'])
        df_test = pd.read_csv(test_path, index_col='id', parse_dates=['time'])
        df_subm = pd.read_csv(subm_path, index_col='id')

        df_train.columns = df_train.columns.str.replace(':', '-')
        df_test.columns = df_test.columns.str.replace(':', '-')

        print("Data loaded successfully!")
        return df_train, df_test, df_subm
    except Exception as e:
        raise ValueError(f"Error loading data: {e}")

def preprocess_data(df_train, df_test, feature_cols, target_col, group_col, date_col, bg_cols, insu_cols, carb_cols, hr_cols, step_cols, cals_cols):
    seed = 43
    Thr_NAN = 49

    for colset in [bg_cols, insu_cols, carb_cols, hr_cols, step_cols, cals_cols]:
        df_train[colset] = df_train[colset].interpolate(axis=1).fillna(method="bfill", axis=1).fillna(method="ffill", axis=1)
        df_test[colset] = df_test[colset].interpolate(axis=1).fillna(method="bfill", axis=1).fillna(method="ffill", axis=1)

    mask = df_train[feature_cols].isna().sum(axis=1) <= Thr_NAN
    df_train = df_train[mask]

    imputer = SimpleImputer()
    df_train[feature_cols] = imputer.fit_transform(df_train[feature_cols])
    df_test[feature_cols] = imputer.transform(df_test[feature_cols])

    df_train["sin_hour"] = np.sin(np.pi * df_train[date_col].dt.hour / 12)
    df_train["cos_hour"] = np.cos(np.pi * df_train[date_col].dt.hour / 12)
    df_test["sin_hour"] = np.sin(np.pi * df_test[date_col].dt.hour / 12)
    df_test["cos_hour"] = np.cos(np.pi * df_test[date_col].dt.hour / 12)

    feature_cols.extend(["sin_hour", "cos_hour"])

    scaler = StandardScaler()
    X = scaler.fit_transform(df_train[feature_cols])
    df_test_final = scaler.transform(df_test[feature_cols])

    y = np.array(df_train[target_col])
    np.save("preprocessed_data/X.npy", X)
    np.save("preprocessed_data/y.npy", y)
    np.save("preprocessed_data/df_test_final.npy", df_test_final)

    print("Data preprocessed and saved successfully!")
    return X, y, df_test_final
