def clean_data(df, missing_value_placeholders=None, max_missing_per_row=None):
    import numpy as np
    import pandas as pd

    # Copy of the DataFrame
    df_cleaned = df.copy()

    # Replace missing value placeholders with NaN
    if missing_value_placeholders is None:
        missing_value_placeholders = [
            "n/a", "na", "--", "-", " ", "", "?", "NA", "N/A",
            "92", "-999", "999", "-1", "None", "NULL", "null", "nan",
            np.inf, -np.inf
        ]
    df_cleaned.replace(missing_value_placeholders, np.nan, inplace=True)

    # Remove rows with more than 'max_missing_per_row' missing values
    if max_missing_per_row is not None:
        missing_counts = df_cleaned.isnull().sum(axis=1)
        df_cleaned = df_cleaned[missing_counts <= max_missing_per_row]
        df_cleaned.reset_index(drop=True, inplace=True)

    # Treat the four specific columns as numerical, and the rest as categorical
    numerical_cols = ['AgeInMonth', 'AgeinYears', 'GCSTotal', 'PatNum']
    categorical_cols = df_cleaned.columns.difference(numerical_cols)

    # Convert categorical columns to categorical data types
    for col in categorical_cols:
        df_cleaned[col] = df_cleaned[col].astype('category')

    # Impute missing values in numerical columns with the mean
    df_cleaned[numerical_cols] = df_cleaned[numerical_cols].fillna(df_cleaned[numerical_cols].mean())

    # Impute missing values in categorical columns with the mode
    for col in categorical_cols:
        mode = df_cleaned[col].mode()
        if not mode.empty:
            df_cleaned[col] = df_cleaned[col].fillna(mode[0])
        else:
            df_cleaned[col] = df_cleaned[col].fillna('unknown')

    # Return the cleaned DataFrame
    return df_cleaned
