"""
This is a script that loads cvs files normalizes them. Each session type is normalized separately
"""

import sys
import os
from pathlib import Path
import pandas as pd

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from f1_prediction_ml.normalize.normalize_quali import QualifyingNormalizer
from f1_prediction_ml.normalize.normalize_race import RaceNormalizer
from f1_prediction_ml.normalize.normalize_free_practice import FreePracticeNormalizer


def normalize_data(list_of_sessions):
    for session in list_of_sessions:
        file_path = f'{project_root}/data/interim/organized_csv_files/{session}'
        df = pd.read_csv(file_path)
        os.makedirs(project_root / 'data' / 'processed' / 'normalized_csv_files', exist_ok=True)

        session_type = df['session_type'].iloc[0]
        if session_type in ['Qualifying', 'Sprint_Qualifying']:
            normalizer = QualifyingNormalizer(processed_data_dir=f'{project_root}/data/interim/organized_csv_files', normalize_data_dir=f'{project_root}/data/normalized')
            normalized_df = normalizer.normalize_quali_data(df[df['session_type'] == session_type])
            normalized_df.to_csv(f'{project_root}/data/processed/normalized_csv_files/{session}_normalized.csv', index=False)
        elif session_type in ['Race', 'Sprint', 'Sprint_Shootout']:
            normalizer = RaceNormalizer(processed_data_dir=f'{project_root}/data/interim/organized_csv_files', normalize_data_dir=f'{project_root}/data/normalized')
            normalized_df = normalizer.normalize_race_data(df[df['session_type'] == session_type])
            normalized_df.to_csv(f'{project_root}/data/processed/normalized_csv_files/{session}_normalized.csv', index=False)
        elif session_type in ['Practice_1', 'Practice_2', 'Practice_3']:
            normalizer = FreePracticeNormalizer(processed_data_dir=f'{project_root}/data/interim/organized_csv_files', normalize_data_dir=f'{project_root}/data/normalized')
            normalized_df = normalizer.normalize_free_practice_data(df[df['session_type'] == session_type])
            normalized_df.to_csv(f'{project_root}/data/processed/normalized_csv_files/{session}_normalized.csv', index=False)
        else:
            print(f'Unknown session type: {session_type}')  
