# Imports
import pandas as pd


##### Cleaning data functions #####

def clean_data(df, columns_to_drop, do_imputation = False):
    # Remove columns with many missing values
    final_df = df.drop(columns = columns_to_drop)

    # If imputation is enabled, apply value-based imputation on category columns
    if do_imputation == True:
        final_df = value_impute(final_df, 'primary_category', 'secondary_category')
        final_df = value_impute(final_df, 'secondary_category', 'tertiary_category')

    # Drop null values and duplicates
    final_df = final_df.dropna(axis=0)
    final_df = final_df.drop_duplicates()

    return final_df


# Function to impute the values based on similar columns: if there's a missing value, the function imputes the most common value for each category, or returns the same value
def value_impute(df, col, target_col):
    df = df.copy()

    mode_map = (
        df
        .dropna(subset=[target_col])
        .groupby(col)[target_col]
        .agg(lambda x: x.mode().iloc[0])
    )
    def impute_column(row):
        if pd.isnull(row[target_col]):
            return mode_map.get(row[col], row[col])
        else:
            return row[target_col]
    df[target_col] = df.apply(impute_column, axis=1)
    return df


