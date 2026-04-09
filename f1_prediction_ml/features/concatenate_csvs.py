"""
Concatenates individual CSV files into a single DataFrame and adding new features 'year', 'event_id' and 'row_id' using the list_of_files.csv.
"""

import sys
import os
import pandas as pd
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from f1_prediction_ml.ml_utils import get_list_of_sessions

from f1_prediction_ml.colors import CYAN, RESET
 
data_path = project_root / 'data' / 'processed' / 'normalized_csv_files'

list_of_sessions = get_list_of_sessions('list_of_normalized_files.csv')

def concatenate_master_log_csv_files(list_of_sessions):
    """
    Concatenate all normalized session CSVs into a single master log DataFrame and save it to disk.

    Args:
        list_of_sessions: List of session identifiers (e.g. '2022_Bahrain_FP1') whose
            normalized CSVs will be loaded and concatenated.
    Returns:
        pd.DataFrame: The concatenated master log DataFrame.
    """
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

