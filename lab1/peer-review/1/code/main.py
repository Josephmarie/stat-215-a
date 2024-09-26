import pandas as pd
import numpy as np

def clean_data(df, outlier='none', fill_na_with='mode', alpha=1.5):
    """
    Cleans a df by handling missing values and trimming outliers.
    """
    # replace NaN with
    if fill_na_with == 'mean':
        df = df.fillna(df.mean())
    elif fill_na_with == 'mode':
        for c in df.columns:
            mode_val = df[c].mode()[0]
            df[c] = df[c].fillna(mode_val)
    elif fill_na_with == 'drop':
        df = df.dropna()
    
    # trim outliers
    if outlier == 'iqr':
        q1 = df.quantile(0.25)
        q3 = df.quantile(0.75)
        iqr = q3 - q1
        df = df[~((df < (q1 - alpha * iqr)) | (df > (q3 + alpha * iqr))).any(axis=1)]
    
    elif outlier == 'z':
        df = df[(np.abs(stats.zscore(df)) < alpha).all(axis=1)]
    
    return df




if __name__ == "__main__":
    data = pd.read_csv("../data/TBI PUD 10-08-2013.csv")
    clean(data)