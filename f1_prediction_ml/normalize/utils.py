from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]

def get_list_of_sessions() -> list:
    """ 
    This script imports the list of file names of processed sessions
    from the data_organizer module, which is responsible for organizing and saving the processed session data.
    The list of file names is stored in a CSV file named 'list_of_files.csv' located in the 'organized_csv_files' directory.
    Args:        None
    Returns:     list_of_sessions (list): A list of file names of processed sessions.
    """
    file_names = []
    list_of_files_path = project_root / 'data' / 'interim' / 'organized_csv_files' / 'list_of_files.csv'

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
            df[col] = pd.to_timedelta(df[col]).dt.total_seconds()
    return df