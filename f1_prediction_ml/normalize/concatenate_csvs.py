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

from f1_prediction_ml.normalize.normalization_utils import get_list_of_sessions

# Color codes
CYAN = "\033[36m"
RESET = "\033[0m"
 
data_path = project_root / 'data' / 'interim' / 'organized_csv_files'

list_of_sessions = get_list_of_sessions()

def concatenate_csv_files(list_of_sessions):
    dataframes = []
    for session in list_of_sessions:
        file_path = f'{data_path}/{session}'
        df = pd.read_csv(file_path)

        # Get year from date column and add to dataframe
        df['year'] = df['date'].str.split('-').str[0]
        # Create event_id by concatenating year and grandprix
        grand_prix = df['grandprix'].iloc[0].replace(' ', '_')
        df['event_id'] = df['year'] + '_' + grand_prix
        # Create row_id by concatenating event_id, sessiontype and drivernumber
        df['row_id'] = df['event_id'] + '_' + df['sessiontype'] + '_' + df['drivernumber'].astype(str)

        dataframes.append(df)
    
    concatenated_df = pd.concat(dataframes, ignore_index=True)

    os.makedirs(project_root / 'data' / 'processed', exist_ok=True)
    concatenated_df.to_csv(project_root / 'data' / 'processed' / 'concatenated_df.csv', index=False)

    print(f'{CYAN}INFO: Concatenated DataFrame saved to {project_root / 'data' / 'processed' / 'concatenated_df.csv'}{RESET}')
    print(f'{CYAN}INFO: Concatenated DataFrame info:{RESET}')
    print(concatenated_df.info())
    print(f'{CYAN}INFO: Concatenated DataFrame head:{RESET}')
    print(concatenated_df.head())
    print(f'{CYAN}INFO: Concatenated DataFrame shape:{RESET}')
    print(concatenated_df.shape)

    return concatenated_df


concatenate_csv_files(list_of_sessions)

