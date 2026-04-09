import sys
import os
from pathlib import Path
import pandas as pd

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from f1_prediction_ml.normalize.normalize_free_practice import FreePracticeNormalizer
from f1_prediction_ml.normalize.normalize_quali import QualifyingNormalizer
from f1_prediction_ml.normalize.normalize_race import RaceNormalizer
from f1_prediction_ml.colors import CYAN, RESET
from f1_prediction_ml.ml_utils import remove_unnecessary_columns, create_list_of_sessions_file

columns_to_remove = ['vsc_duration', 'vsc_ending_duration', 'sc_duration', 'not_green_duration', 'total_duration']

def normalize_data(list_of_sessions):
    """
    Normalize each organized session CSV by delegating to the appropriate session-type normalizer.

    Args:
        list_of_sessions: List of session identifiers whose organized CSVs will be normalized.
    """
    for session in list_of_sessions:
        file_path = f'{project_root}/data/interim/organized_csv_files/{session}_organized.csv'
        df = pd.read_csv(file_path)
        os.makedirs(project_root / 'data' / 'processed' / 'normalized_csv_files', exist_ok=True)

        session_type = df['session_type'].iloc[0]
        target_data_dir = project_root / 'data' / 'list_of_available_sessions'

        if session_type in ['Q', 'SQ']:
            df = remove_unnecessary_columns(df, columns_to_remove)
            normalized_df = QualifyingNormalizer().normalize_quali_data(df)
            normalized_df.to_csv(f'{project_root}/data/processed/normalized_csv_files/{session}_normalized.csv', index=False)
            create_list_of_sessions_file(target_data_dir, 'list_of_normalized_files.csv', session, source='normalized')
        elif session_type in ['R', 'S', 'SS']:     
            df = remove_unnecessary_columns(df, columns_to_remove)
            normalized_df = RaceNormalizer().normalize_race_data(df)
            normalized_df.to_csv(f'{project_root}/data/processed/normalized_csv_files/{session}_normalized.csv', index=False)
            create_list_of_sessions_file(target_data_dir, 'list_of_normalized_files.csv', session, source='normalized')
        elif session_type in ['FP1', 'FP2', 'FP3']:
            df = remove_unnecessary_columns(df, columns_to_remove)
            normalized_df = FreePracticeNormalizer().normalize_free_practice_data(df)
            normalized_df.to_csv(f'{project_root}/data/processed/normalized_csv_files/{session}_normalized.csv', index=False)
            create_list_of_sessions_file(target_data_dir, 'list_of_normalized_files.csv', session, source='normalized')
        else:
            print(f'Unknown session type: {session_type}')

def concatenate_master_log_csv_files(list_of_sessions):
    """
    Concatenate all normalized session CSVs into a single master log DataFrame.

    Args:
        list_of_sessions: List of session identifiers to concatenate.
    Returns:
        pd.DataFrame: The concatenated master log DataFrame.
    """
    data_path = project_root / 'data' / 'processed' / 'normalized_csv_files'
    dataframes = []
    for session in list_of_sessions:
        file_path = f'{data_path}/{session}_normalized.csv'
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
