import pandas as pd
import numpy as np

def preprocess_data(filepath):
    """
    Cleans and prepares the water consumption data for the CRL model.

    Args:
        filepath (str): The path to the raw CSV data file.

    Returns:
        pandas.DataFrame: A cleaned and preprocessed DataFrame.
    """
    df = pd.read_csv(filepath)

    # --- Data Cleaning ---
    # Correcting data types
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

    # Handling missing or placeholder values in 'Population'
    # Simple strategy: Fill with the mean of the facility's population
    df['Population'] = pd.to_numeric(df['Population'], errors='coerce')
    df['Population'] = df.groupby('Facility')['Population'].transform(lambda x: x.fillna(x.mean()))
    df['Population'] = df['Population'].fillna(df['Population'].mean()).astype(int)

    # Clean numeric columns with potential non-numeric entries
    numeric_cols = ['Daily_Usage_L', 'Drinking_L', 'Sanitation_L', 'Labs_Cleaning_L', 'Cooking_L', 'Rainfall_mm', 'Temp_Min_C', 'Temp_Max_C', 'Humidity_Percent', 'Leak_Reports', 'Repair_Hrs']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0) # Fill NaNs with 0 after coercion

    # --- Feature Engineering ---
    # Create a 'Day_of_Week' feature
    df['Day_of_Week'] = df['Date'].dt.dayofweek # Monday=0, Sunday=6

    # Convert categorical features into numerical format using one-hot encoding
    categorical_cols = ['Pipe_Condition', 'Special_Event', 'Vacation_Exam', 'Facility']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # For simplicity in the RL model, we'll create a single 'water_stress_level'
    # This is a simplified metric for demonstration
    # More complex logic could be used here
    df['water_stress_level'] = (df['Daily_Usage_L'] / df['Population']).fillna(0)
    df['water_stress_level'] = pd.qcut(df['water_stress_level'], 4, labels=False, duplicates='drop')


    # Drop columns that are not needed for the model
    df = df.drop(columns=['Date', 'Peak_Hours'])

    print("Data preprocessing complete.")
    print("Preprocessed data sample:")
    print(df.head())

    return df

if __name__ == '__main__':
    # This allows you to run this script directly to see the preprocessed output
    preprocessed_df = preprocess_data('vit_chennai_water_dataset_noisy.csv')
