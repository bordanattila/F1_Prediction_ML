"""
This is a script that loads cvs files for feature engineering. Each session type is processed separately.
"""

from functools import reduce
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
from f1_prediction_ml.colors import CYAN, MAGENTA, YELLOW, RESET
from f1_prediction_ml.ml_utils import create_list_of_sessions_file, get_list_of_sessions, remove_unnecessary_columns
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

# Merge keys for combining session data into model training data. These should be present in all session master files 
# and uniquely identify each row.
KEYS = ['event_id', 'driver_id']

# Columns that should only come from the base race table. These are metadata columns and target variable columns that we want to ensure 
# only come from the race table to avoid any issues with missing values or inconsistencies across session files.
BASE_META_COLS = [
    'year',
    'grand_prix',
    'event_id',
    'row_id',
    'driver_id',
    'abbreviation',
    'team_name',
    'team_id',
]

def create_features_per_session(list_of_sessions):
    """
    Load each normalized session CSV and generate prefixed feature columns based on session type.

    Results are grouped by session type and saved as session master CSVs.

    Args:
        list_of_sessions: List of session identifiers whose normalized CSVs will be loaded.
    """
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
    """
    Merge session master CSVs (race, qualifying, free practice, etc.) into a single
    model-training DataFrame keyed on (event_id, driver_id).

    The race table is used as the base; all other sessions are left-joined onto it.

    Args:
        list_of_sessions: List of session master filenames (e.g. 'race_master_df.csv').
    Returns:
        pd.DataFrame: The merged model training DataFrame.
    """
    data_path = project_root / 'data' / 'processed' / 'session_master_files'
    session_tables = {}
    for session in list_of_sessions:
        file_path = f'{data_path}/{session.lower()}'
        df = pd.read_csv(file_path)

        # Define merge keys 
        KEYS = ['event_id', 'driver_id'] 
        missing = [c for c in KEYS if c not in df.columns]
        if missing:
            raise ValueError(f'{session} missing merge keys: {missing}')

        # Enforce 1 row per key per session file
        if df.duplicated(KEYS).any():
            duplicates = df[df.duplicated(KEYS, keep=False)].sort_values(KEYS)
            raise ValueError(f'{session} has duplicate rows per {KEYS}. Example:\n{duplicates.head(10)}')
        
        session_lower = session.lower()

        if session_lower == "race_master_df.csv":
            # Keep metadata + race columns only from race table
            race_cols = [c for c in df.columns if c in BASE_META_COLS or c.startswith("race_")]
            session_tables["race"] = df[race_cols].copy()

        elif session_lower == "fp1_master_df.csv":
            fp1_cols = KEYS + [c for c in df.columns if c.startswith("fp1_")]
            session_tables["fp1"] = df[fp1_cols].copy()

        elif session_lower == "fp2_master_df.csv":
            fp2_cols = KEYS + [c for c in df.columns if c.startswith("fp2_")]
            session_tables["fp2"] = df[fp2_cols].copy()

        elif session_lower == "fp3_master_df.csv":
            fp3_cols = KEYS + [c for c in df.columns if c.startswith("fp3_")]
            session_tables["fp3"] = df[fp3_cols].copy()

        elif session_lower == "qualifying_master_df.csv":
            q_cols = KEYS + [c for c in df.columns if c.startswith("quali_")]
            session_tables["q"] = df[q_cols].copy()

        elif session_lower == "sprint_qualifying_master_df.csv":
            sq_cols = KEYS + [c for c in df.columns if c.startswith("sq_")]
            session_tables["sq"] = df[sq_cols].copy()

        elif session_lower == "sprint_race_master_df.csv":
            s_cols = KEYS + [c for c in df.columns if c.startswith("sprint_")]
            session_tables["s"] = df[s_cols].copy()
        
        elif session_lower == "sprint_shootout_master_df.csv":
            s_cols = KEYS + [c for c in df.columns if c.startswith("sprint_")]
            session_tables["ss"] = df[s_cols].copy()

        else:
            print(f'{YELLOW}Skipping unsupported session: {session}{RESET}')

    if 'race' not in session_tables:
        raise ValueError('Race table ("race") is required as the base table.')

    # Start from race table so final rows represent actual race entrants
    model_training_data = session_tables['race'].copy()

    # Merge pre-race sessions onto race entrants
    merge_order = ['q', 'sq', 's', 'ss', 'fp1', 'fp2', 'fp3',]

    for name in merge_order:
        if name in session_tables:
            model_training_data = model_training_data.merge(
                session_tables[name],
                on=KEYS,
                how='left'
            )

    os.makedirs(project_root / 'data' / 'processed', exist_ok=True)
    model_training_data.to_csv(project_root / 'data' / 'processed' / 'model_training_data.csv', index=False)

    print(f'{CYAN}INFO: Concatenated DataFrame saved to {project_root / 'data' / 'processed' / 'model_training_data.csv'}{RESET}')
    print(f'{CYAN}INFO: Concatenated DataFrame info:{RESET}')
    print(model_training_data.info())
    print(f'{CYAN}INFO: Concatenated DataFrame head:{RESET}')
    print(model_training_data.head())
    print(f'{CYAN}INFO: Concatenated DataFrame shape:{RESET}')
    print(model_training_data.shape)

    return model_training_data