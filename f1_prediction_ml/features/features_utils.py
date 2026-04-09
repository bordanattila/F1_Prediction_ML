import os
from pathlib import Path
import pandas as pd

from f1_prediction_ml.colors import CYAN, YELLOW, RESET

SESSION_TYPE_TO_FILENAME = {
    'FP1': 'fp1_master_df.csv',
    'FP2': 'fp2_master_df.csv',
    'FP3': 'fp3_master_df.csv',
    'Q': 'qualifying_master_df.csv',
    'SQ': 'sprint_qualifying_master_df.csv',
    'R': 'race_master_df.csv',
    'S': 'sprint_race_master_df.csv',
    'SS': 'sprint_shootout_master_df.csv',
}


def create_row_id(df):
    """
    Creates a unique row_id for each row in the DataFrame by concatenating event_id, session_type and driver_number.
    Args:
        df (pd.DataFrame): The DataFrame containing the session data.
    Returns:
        pd.DataFrame: A DataFrame containing the session data with a new row_id column.
    """

    # Get year from date column and add to dataframe
    df['year'] = df['date'].str.split('-').str[0]
    # Create event_id by concatenating year and grand_prix
    grand_prix = df['grand_prix'].iloc[0].replace(' ', '_')
    df['event_id'] = df['year'] + '_' + grand_prix
    # Create session_id by concatenating event_id and session_type
    df['session_id'] = df['event_id'] + '_' + df['session_type']
    # Create row_id by concatenating session_id and driver_number
    df['row_id'] = df['session_id'] + '_' + df['driver_number'].astype(str)

    return df


def concatenate_session_dfs(session_data_frames: dict):
    """
    Concatenates the processed DataFrames for each session type into separate master CSV files.
    Groups DataFrames by session type (FP1, FP2, FP3, Q, SQ, R, S, SS) and saves each to its own file.
    
    Args:
        session_data_frames (dict): A dictionary containing the processed DataFrames.
            Keys are formatted as '{year}_{grand_prix}_{session_type}' 
            (e.g., '2018_Australia_FP1', '2018_Bahrain_Q', '2018_Monaco_R').
    Returns:
        dict: A dictionary where keys are filenames (e.g., 'fp1_master_df.csv') 
              and values are the master DataFrames.
    """
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / 'data' / 'processed' / 'session_master_files'
    os.makedirs(output_dir, exist_ok=True)

    grouped_dfs = {session_type: [] for session_type in SESSION_TYPE_TO_FILENAME}
    for key, df in session_data_frames.items():
        session_type = key.split('_')[-1]
        if session_type in grouped_dfs:
            grouped_dfs[session_type].append(df)
        else:
            print(f'{YELLOW}WARNING: Unknown session type "{session_type}" in key "{key}"{RESET}')

    result_dfs = {}
    for session_type, dfs in grouped_dfs.items():
        if not dfs:
            continue

        print(f'{CYAN}Concatenating {len(dfs)} {session_type} dataframes...{RESET}')
        master_df = pd.concat(dfs, ignore_index=True)
        
        if 'row_id' in master_df.columns:
            master_df = master_df.drop_duplicates(subset=['row_id'], keep='last')

        filename = SESSION_TYPE_TO_FILENAME[session_type]
        output_path = output_dir / filename
        master_df.to_csv(output_path, index=False)
        result_dfs[filename] = master_df
        print(f'{CYAN}Saved {session_type} master ({len(master_df)} rows) to {output_path}{RESET}')

    return result_dfs