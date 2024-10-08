def drop_rows_zero(data, categorical_cols, threshold):
    """
    Drops rows from ling_data where the number of zeros in the categorical columns 
    exceeds or equals the specified threshold.
    
    Parameters:
    ling_data (pd.DataFrame): The input dataframe.
    categorical_cols (list): List of categorical column names to check.
    threshold (int): The minimum number of zero values in categorical columns to drop the row.
    
    Returns:
    pd.DataFrame: A new dataframe with the filtered rows.
    """
    # Create a boolean mask where each row has threshold or more zeros in the specified categorical columns
    mask = (data[categorical_cols] == 0).sum(axis=1) >= threshold
    
    # Return a dataframe with the rows where the condition is False (i.e., not meeting the threshold)
    return data[~mask]
