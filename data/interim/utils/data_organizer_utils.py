# Color codes
CYAN = "\033[36m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
RESET = "\033[0m"

# Normalize common column names
def standardize_column_names(df):
    """
    Standardizes column names by stripping whitespace, replacing spaces with underscores, and converting to lowercase.
    Example: 'DriverNumber' -> 'driver_number', 'LapTime' -> 'lap_time'
    """
    # If column name is camelCase (i.e.: 'DriverNumber') insert '_' before uppercase letter
    df.columns = df.columns.str.replace('([a-z])([A-Z])', r'\1_\2', regex=True)
    # Handle digit followed by uppercase (e.g., 'Sector1Time' -> 'Sector1_Time')
    df.columns = df.columns.str.replace('([0-9])([A-Z])', r'\1_\2', regex=True)
    # Strip whitespace, replace spaces with underscores, and convert to lowercase
    df.columns = [col.strip().replace(' ', '_').lower() for col in df.columns]
    return df

# Standardize practice session names (Practice 1 -> Practice_1, Practice 2 -> Practice_2, Practice 3 -> Practice_3)
def standardize_practice_session_names(df):
    """
    Standardizes practice session names by replacing 'Practice 1', 'Practice 2', 'Practice 3' with 'Practice_1', 'Practice_2', 'Practice_3'.
    """
    df['session_type'] = df['session_type'].replace({'Practice 1': 'Practice_1', 'Practice 2': 'Practice_2', 'Practice 3': 'Practice_3'})
    return df