from pathlib import Path
import pandas as pd
import os

# Color codes
CYAN = "\033[36m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
RESET = "\033[0m"
# Add project root to Python path
project_root = Path(__file__).resolve().parents[1]

def get_list_of_sessions(source_file) -> list:
    """ 
    This script imports the list of file names of processed sessions
    from the data_organizer module, which is responsible for organizing and saving the processed session data.
    The list of file names is stored in a CSV file named 'list_of_files.csv' located in the 'organized_csv_files' directory.
    Args:        None
    Returns:     list_of_sessions (list): A list of file names of processed sessions.
    """
    file_names = []
    list_of_files_path = project_root / 'data' / 'list_of_available_sessions' / source_file

    with open(list_of_files_path, 'r') as file:
        list_of_files = file.readlines()[1:]
        for file_name in list_of_files:
            file_names.append(file_name.strip())

    return file_names


# Convert time columns to seconds
def convert_time_columns_to_seconds(df, time_columns):
    """
    Converts time columns in the DataFrame to seconds.
    Args:
        df (pd.DataFrame): The DataFrame containing the time columns to be converted.
        time_columns (list): A list of column names that contain time values to be converted.
    Returns:
        pd.DataFrame: The DataFrame with the specified time columns converted to seconds.       
    """
    for col in time_columns:
        if col in df.columns:
            new_col_name = f'{col}_seconds'
            df[new_col_name] = pd.to_timedelta(df[col]).dt.total_seconds()
    return df

# Remove columns that are not relevant for the model
def remove_unnecessary_columns(df, columns_to_remove):
    """
    Removes columns that are not relevant for the model.
    Args:
        df (pd.DataFrame): The DataFrame to be modified.
        columns_to_remove (list): A list of column names to be removed.
    Returns:
        pd.DataFrame: The DataFrame with the specified columns removed.
    """
    return df.drop(columns=columns_to_remove, errors='ignore')

# Create list of sessions for which we have data
def create_list_of_sessions_file(target_data_dir, target_file_name, input_filename, source):
    """
    Creates a list of sessions for which we have data.
    Args:
        target_data_dir (str): The directory where the list of sessions file is located.
        target_file_name (str): The name of the list of sessions file.
        input_filename (str): The name of the session file to be added to the list.
    Returns:
        None
    """
    list_of_files = os.path.join(target_data_dir, target_file_name)
    
    if os.path.exists(list_of_files):
        existing_files_df = pd.read_csv(list_of_files)
        # Only add if not already in the list
        if input_filename in existing_files_df['filename'].values:
            return
        new_file_entry = pd.DataFrame({'filename': [input_filename]})
        updated_files_df = pd.concat([existing_files_df, new_file_entry], ignore_index=True)
        updated_files_df.to_csv(list_of_files, index=False)
    else:
        new_file_entry = pd.DataFrame({'filename': [input_filename]})
        new_file_entry.to_csv(list_of_files, index=False)
    print(f'{CYAN}INFO: Added {input_filename} to list of {source} files{RESET}')

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