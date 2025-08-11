import fastf1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

fastf1.Cache.enable_cache('cache')

def fetch_f1_data(year, round_number):
    """Fetch data using official F1 API via FastF1"""
    quali = fastf1.get_session(year, round_number, 'Q')
    quali.load()
    results = quali.results[['DriverNumber', 'FullName', 'TeamName', 'Q1', 'Q2', 'Q3']]
    results = results.rename(columns={'FullName': 'Driver'})
    for col in ['Q1', 'Q2', 'Q3']:
        results[col + '_sec'] = results[col].apply(
            lambda x: x.total_seconds() if pd.notnull(x) else None
            )
    print("\nQualifying Results Structure:")
    print(results.head())
    return results

def convert_time_to_seconds(time_str):
    if pd.isna(time_str):
        return None

    if ':' in time_str:
        minutes, seconds = time_str.split(':')
        return float(minutes) * 60 + float(seconds)
    else:
        return float(time_str)

def clean_data(df):
    print("\nBefore cleaning:")
    print(df[['Driver', 'Q1', 'Q2', 'Q3']].head())
    
    df['Q1_sec'] = df['Q1'].apply(convert_time_to_seconds)
    df['Q2_sec'] = df['Q2'].apply(convert_time_to_seconds)
    df['Q3_sec'] = df['Q3'].apply(convert_time_to_seconds)
    
    print("\nAfter cleaning:")
    print(df[['Driver', 'Q1_sec', 'Q2_sec', 'Q3_sec']].head())
    
    return df.dropna()

def train_and_evaluate(df):
    X = df[['Q1_sec', 'Q2_sec']]
    y = df['Q3_sec']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)

    predictions = model.predict(X)
    
    results_df = df[['Driver', 'TeamName', 'Q1_sec', 'Q2_sec', 'Q3_sec']].copy()
    results_df['Predicted_Q3'] = predictions
    results_df['Difference'] = results_df['Predicted_Q3'] - results_df['Q3_sec']
    
    results_df = results_df.sort_values('Predicted_Q3')
    
    print("\nPredicted Q3 Rankings:")
    print("=" * 70)
    print(f"{'Position':<10}{'Driver':<15}{'Team':<20}{'Predicted Time':<15}{'Actual Time':<15}")
    print("-" * 70)
    
    for idx, row in results_df.iterrows():
        pred_time = f"{row['Predicted_Q3']:.3f}"
        actual_time = f"{row['Q3_sec']:.3f}" if not pd.isna(row['Q3_sec']) else "N/A"
        print(f"{results_df.index.get_loc(idx)+1:<10}{row['Driver']:<15}{row['TeamName']:<20}{pred_time:<15}{actual_time:<15}")

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nModel Performance Metrics:")
    print(f'Mean Absolute Error: {mae:.2f} seconds')
    print(f'R^2 Score: {r2:.2f}')

def fetch_recent_data():
    """Fetch data from recent races using FastF1"""
    all_data = []
    
    current_year = 2025
    for round_num in range(1, 15):  # First 14 races of 2025
        print(f"Fetching data for {current_year} round {round_num}...")
        df = fetch_f1_data(current_year, round_num)
        if df is not None:
            df['Year'] = current_year
            df['Round'] = round_num
            all_data.append(df)
    
    print("Fetching 2024 Dutch GP data...")  
    dutch_2024 = fetch_f1_data(2024, 15) 
    if dutch_2024 is not None:
        dutch_2024['Year'] = 2024
        dutch_2024['Round'] = 15
        all_data.append(dutch_2024)
    
    return all_data

def apply_performance_factors(predictions_df):
    """Apply 2025-specific performance factors"""
    base_time = 70.0  # in seconds
    
    team_factors = {
        'Red Bull Racing': 0.999,   
        'Ferrari': 0.992,         
        'McLaren': 0.990,         
        'Mercedes': 0.995,        
        'Aston Martin': 1.001,     
        'RB': 1.002,              
        'Williams': 1.003,         
        'Haas F1 Team': 1.004,     
        'Kick Sauber': 1.000,     
        'Alpine': 1.005,          
    }
    
    driver_factors = {
        'Max Verstappen': 0.992,    
        'Charles Leclerc': 0.994,   
        'Carlos Sainz': 0.999,     
        'Lando Norris': 0.991,     
        'Oscar Piastri': 0.990,    
        'Kimi Antonelli': 0.999,   
        'Lewis Hamilton': 0.999,   
        'George Russell': 0.995,   
        'Fernando Alonso': 1.000,    
        'Lance Stroll': 1.001,     
        'Alex Albon': 1.001,        
        'Franco Colapinto': 1.001, 
        'Yuki Tsunoda': 1.002,     
        'Liam Lawson': 1.002,  
        'Isack Hadjar': 1.003,       
        'Gabriel Bortoleto': 1.000,  
        'Nico Hulkenberg': 1.002,   
        'Oliver Bearman': 1.004,  
        'Pierre Gasly': 1.003,     
        'Esteban Ocon': 1.004,   
    }
    
    for idx, row in predictions_df.iterrows():
        team_factor = team_factors.get(row['Team'], 1.000)
        driver_factor = driver_factors.get(row['Driver'], 1.000)
        
        base_prediction = base_time * team_factor * driver_factor
        
        random_variation = np.random.uniform(-0.15, 0.15)
        predictions_df.loc[idx, 'Predicted_Q3'] = base_prediction + random_variation
    
    return predictions_df

def predict_dutch_gp(model, latest_data):
    """Predict Q3 times for Dutch GP 2025"""

    driver_teams = {
        'Max Verstappen': 'Red Bull Racing',
        'Yuki Tsunoda': 'Red Bull Racing',
        'Charles Leclerc': 'Ferrari',
        'Lewis Hamilton': 'Ferrari',
        'Kimi Antonelli': 'Mercedes',
        'George Russell': 'Mercedes',
        'Lando Norris': 'McLaren',
        'Oscar Piastri': 'McLaren',
        'Fernando Alonso': 'Aston Martin',
        'Lance Stroll': 'Aston Martin',
        'Liam Lawson': 'RB',
        'Isack Hadjar ': 'RB',
        'Alexander Albon': 'Williams',
        'Carlos Sainz': 'Williams',
        'Gabriel Bortaleto': 'Kick Sauber',
        'Nico Hulkenberg': 'Kick Sauber',
        'Estaban Ocon': 'Haas F1 Team',
        'Oliver Bearman': 'Haas F1 Team',
        'Pierre Gasly': 'Alpine',
        'Franco Colapinto': 'Alpine'
    }
    
    results_df = pd.DataFrame(list(driver_teams.items()), columns=['Driver', 'Team'])
    
    results_df = apply_performance_factors(results_df)
    
    results_df = results_df.sort_values('Predicted_Q3')
    
    print("\nDutch GP 2025 Qualifying Predictions:")
    print("=" * 100)
    print(f"{'Position':<10}{'Driver':<20}{'Team':<25}{'Predicted Q3':<15}")
    print("-" * 100)
    
    for idx, row in results_df.iterrows():
        print(f"{results_df.index.get_loc(idx)+1:<10}"
              f"{row['Driver']:<20}"
              f"{row['Team']:<25}"
              f"{row['Predicted_Q3']:.3f}s")


if __name__ == "__main__":
    print("Initializing enhanced F1 prediction model...")

    all_data = fetch_recent_data()
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        valid_data = combined_df.dropna(subset=['Q1_sec', 'Q2_sec', 'Q3_sec'], how='all')

        imputer = SimpleImputer(strategy='median')

        X = valid_data[['Q1_sec', 'Q2_sec']]
        y = valid_data['Q3_sec']

        X_clean = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        y_clean = pd.Series(imputer.fit_transform(y.values.reshape(-1, 1)).ravel())

        model = GradientBoostingRegressor()
        model.fit(X_clean, y_clean)

        predict_dutch_gp(model, valid_data)

        y_pred = model.predict(X_clean)
        mae = mean_absolute_error(y_clean, y_pred)
        r2 = r2_score(y_clean, y_pred)

        print("\nModel Performance Metrics:")
        print(f'Mean Absolute Error: {mae:.2f} seconds')
        print(f'R^2 Score: {r2:.2f}')
    else:
        print("Failed to fetch F1 data")

