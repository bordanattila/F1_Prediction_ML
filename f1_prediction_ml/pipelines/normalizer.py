import sys
import os
from pathlib import Path
import pandas as pd

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from f1_prediction_ml.normalize.normalize_free_prectice import FreePracticeNormalizer
from f1_prediction_ml.normalize.normalize_quali import QualifyingNormalizer
from f1_prediction_ml.normalize.normalize_race import RaceNormalizer
from f1_prediction_ml.ml_utils import CYAN, RESET, remove_unnecessary_columns, create_list_of_sessions_file

columns_to_remove = ['vsc_duration', 'vsc_ending_duration', 'sc_duration', 'not_green_duration', 'total_duration']

def normalize_data(list_of_sessions):
    for session in list_of_sessions:
        file_path = f'{project_root}/data/interim/organized_csv_files/{session}_organized.csv'
        df = pd.read_csv(file_path)
        os.makedirs(project_root / 'data' / 'processed' / 'normalized_csv_files', exist_ok=True)
        session_type = df['session_type'].iloc[0]
        if session_type in ['Qualifying', 'Sprint_Qualifying']:
            df = remove_unnecessary_columns(df, columns_to_remove)
            normalized_df = QualifyingNormalizer().normalize_quali_data(df[df['session_type'] == session_type])
            normalized_df.to_csv(f'{project_root}/data/processed/normalized_csv_files/{session}_normalized.csv', index=False)
        elif session_type in ['Race', 'Sprint', 'Sprint_Shootout']:     
            df = remove_unnecessary_columns(df, columns_to_remove)
            normalized_df = RaceNormalizer().normalize_race_data(df[df['session_type'] == session_type])
            normalized_df.to_csv(f'{project_root}/data/processed/normalized_csv_files/{session}_normalized.csv', index=False)
        elif session_type in ['Practice_1', 'Practice_2', 'Practice_3']:
            df = remove_unnecessary_columns(df, columns_to_remove)
            normalized_df = FreePracticeNormalizer().normalize_free_practice_data(df[df['session_type'] == session_type])
            normalized_df.to_csv(f'{project_root}/data/processed/normalized_csv_files/{session}_normalized.csv', index=False)
        else:
            print(f'Unknown session type: {session_type}')

        # Save list of processed session for further processing
        filename = session
        target_data_dir = project_root / 'data' / 'list_of_available_sessions'
        create_list_of_sessions_file(target_data_dir, 'list_of_normalized_files.csv', filename)
