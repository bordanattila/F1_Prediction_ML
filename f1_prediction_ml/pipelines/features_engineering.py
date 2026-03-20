"""
This is a script that loads cvs files for feature engineering. Each session type is processed separately.
"""

import sys
import os
from pathlib import Path
import pandas as pd

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from f1_prediction_ml.features.features_quali import QualifyingFeatures
from f1_prediction_ml.features.features_race import RaceFeatures
from f1_prediction_ml.features.features_free_practice import FreePracticeFeatures
from f1_prediction_ml.ml_utils import CYAN, MAGENTA, YELLOW, RESET, create_list_of_sessions_file, get_list_of_sessions, remove_unnecessary_columns
from f1_prediction_ml.features.features_utils import concatenate_session_dfs

BASE_ID_COLS = [
    'year', 'grand_prix', 'event_id', 'row_id', 'driver_id', 'abbreviation', 'team_name', 'team_id'
]

SESSION_METRICS = [
    'lap_count', 'lap_mean', 'lap_std', 'lap_best', 'lap_median',
    'air_temp_mean', 'air_temp_std',
    'track_temp_mean', 'track_temp_std',
    'humidity_mean', 'humidity_std',
    'wind_speed_mean', 'wind_speed_std',
    'rain_any', 'rain_samples_ratio',
    'yellow_count', 'red_count',
    'vsc_deployed_count', 'sc_deployed_count',
    'disruption_ratio'
]

def create_features_per_session(list_of_sessions):
    data_frames = {}
    
    for session in list_of_sessions:
        file_path = f'{project_root}/data/processed/normalized_csv_files/{session}_normalized.csv'
        df = pd.read_csv(file_path)

        session_type = df['session_type'].iloc[0]

        if session_type in ['Q', 'SQ']:
            feature_ready_df = QualifyingFeatures().create_quali_features(df, SESSION_METRICS, BASE_ID_COLS)
        elif session_type in ['R', 'S', 'SS']:
            feature_ready_df = RaceFeatures().create_race_features(df, SESSION_METRICS, BASE_ID_COLS)
        elif session_type in ['FP1', 'FP2', 'FP3']:
            feature_ready_df = FreePracticeFeatures().create_free_practice_features(df, SESSION_METRICS, BASE_ID_COLS)
        else:
            print(f'{MAGENTA}Unknown session type: {session_type}{RESET}')
            continue

        data_frames[session] = feature_ready_df

    if data_frames:
        session_master_files = concatenate_session_dfs(data_frames)
        for session in session_master_files.keys():
            create_list_of_sessions_file(project_root / 'data' / 'list_of_available_sessions', 'list_of_session_master_files.csv', session, source='session_master')
    else:        
        print(f'{MAGENTA}WARNING: No data frames were created for any session. Please check the input data and session types.{RESET}')  


def concatenate_model_training_data(list_of_sessions):
    data_path = project_root / 'data' / 'processed' / 'session_master_files'
    dataframes = []
    for session in list_of_sessions:
        file_path = f'{data_path}/{session.lower()}'
        df = pd.read_csv(file_path)
        dataframes.append(df)
    
    master_log_df = pd.concat(dataframes, ignore_index=True)

    os.makedirs(project_root / 'data' / 'processed', exist_ok=True)
    master_log_df.to_csv(project_root / 'data' / 'processed' / 'master_log_df.csv', index=False)

    print(f'{CYAN}INFO: Concatenated DataFrame saved to {project_root / 'data' / 'processed' / 'master_log_df.csv'}{RESET}')
    print(f'{CYAN}INFO: Concatenated DataFrame info:{RESET}')
    print(master_log_df.info())
    print(f'{CYAN}INFO: Concatenated DataFrame head:{RESET}')
    print(master_log_df.head())
    print(f'{CYAN}INFO: Concatenated DataFrame shape:{RESET}')
    print(master_log_df.shape)

    return master_log_df