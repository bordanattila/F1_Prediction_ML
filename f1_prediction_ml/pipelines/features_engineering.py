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


def create_features(list_of_sessions):
    for session in list_of_sessions:
        file_path = f'{project_root}/data/processed/normalized_csv_files/{session}_normalized.csv'
        df = pd.read_csv(file_path)
        os.makedirs(project_root / 'data' / 'processed' / 'feature_ready_csv_files', exist_ok=True)

        session_type = df['session_type'].iloc[0]
        if session_type in ['Qualifying', 'Sprint_Qualifying']:
            normalized_df = QualifyingFeatures().create_quali_features(df[df['session_type'] == session_type])
            normalized_df.to_csv(f'{project_root}/data/processed/feature_ready_csv_files/{session}_feature_ready.csv', index=False)
        elif session_type in ['Race', 'Sprint', 'Sprint_Shootout']:
            normalized_df = RaceFeatures().create_race_features(df[df['session_type'] == session_type])
            normalized_df.to_csv(f'{project_root}/data/processed/feature_ready_csv_files/{session}_feature_ready.csv', index=False)
        elif session_type in ['Practice_1', 'Practice_2', 'Practice_3']:
            normalized_df = FreePracticeFeatures().create_free_practice_features(df[df['session_type'] == session_type])
            normalized_df.to_csv(f'{project_root}/data/processed/feature_ready_csv_files/{session}_feature_ready.csv', index=False)
        else:
            print(f'Unknown session type: {session_type}')  
