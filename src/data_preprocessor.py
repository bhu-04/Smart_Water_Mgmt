import pandas as pd
import numpy as np

def preprocess_data(filepath):
    """
    Cleans, engineers features for, and prepares the water consumption data for the CRL model.
    """
    df = pd.read_csv(filepath)

    # --- Data Cleaning ---
    # Correcting data types
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

    # Handling missing or placeholder values in 'Population'
    df['Population'] = pd.to_numeric(df['Population'], errors='coerce')
    df['Population'] = df.groupby('Facility')['Population'].transform(lambda x: x.fillna(x.mean()))
    df['Population'] = df['Population'].fillna(df['Population'].mean()).astype(int)

    # Clean numeric columns with potential non-numeric entries
    numeric_cols = ['Daily_Usage_L', 'Drinking_L', 'Sanitation_L', 'Labs_Cleaning_L', 'Cooking_L', 'Rainfall_mm', 'Temp_Min_C', 'Temp_Max_C', 'Humidity_Percent', 'Leak_Reports', 'Repair_Hrs']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0) # Fill NaNs with 0 after coercion

    # --- Feature Engineering & Categorical Mapping ---
    # Create a 'Day_of_Week' feature
    df['Day_of_Week'] = df['Date'].dt.dayofweek # Monday=0, Sunday=6

    # Map Pipe_Condition to a numerical scale (ordinal)
    # Fill missing values with the most common condition, 'Good'
    df['Pipe_Condition'] = df['Pipe_Condition'].fillna('Good')
    pipe_condition_map = {'Good': 0, 'Fair': 1, 'Poor': 2}
    df['Pipe_Condition_Mapped'] = df['Pipe_Condition'].map(pipe_condition_map).fillna(0).astype(int)

    # Map Special_Event to a numerical scale (nominal)
    # Fill missing values with 'None'
    df['Special_Event'] = df['Special_Event'].fillna('None')
    event_map = {'None': 0, 'Conference': 1, 'Fest': 2, 'Sports': 3}
    df['Special_Event_Mapped'] = df['Special_Event'].map(event_map).fillna(0).astype(int)


    # For simplicity in the RL model, we'll create a single 'water_stress_level'
    # Using np.inf to handle potential division by zero if population is 0
    df['water_stress_level'] = (df['Daily_Usage_L'] / df['Population']).fillna(0)
    df['water_stress_level'].replace([np.inf, -np.inf], 0, inplace=True)
    df['water_stress_level'] = pd.qcut(df['water_stress_level'], 4, labels=False, duplicates='drop')

    print("Data preprocessing and feature engineering complete.")

    # --- Feature Selection for RL State ---
    # This is the crucial step to reduce the state space dimension
    selected_features = [
        'Daily_Usage_L',
        'Rainfall_mm',
        'Temp_Max_C',
        'Leak_Reports',
        'Day_of_Week',
        'Pipe_Condition_Mapped',
        'Special_Event_Mapped',
        'water_stress_level' # The target/reward signal
    ]
    
    df_selected = df[selected_features].copy()

    print("Selected features for RL model state space.")
    print("Preprocessed data sample:")
    print(df_selected.head())

    return df_selected

