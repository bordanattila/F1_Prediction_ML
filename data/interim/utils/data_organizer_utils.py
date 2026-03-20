import pandas as pd

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

# Normalize common session names
def standardize_session_names(df):
    """
    Normalizes session names by replacing 'Practice 1', 'Practice 2', 'Practice 3', 'Qualifying', 'Race', 'Sprint', 'Sprint Qualifying',
    'Sprint Shootout' with 'FP1', 'FP2', 'FP3', 'Q', 'R', 'S', 'SQ', 'SS' respectively.
    Args:
        df (pd.DataFrame): The DataFrame containing the session_type column to be normalized.
    Returns:
        pd.DataFrame: The DataFrame with the session_type column normalized."""
    df["session_type"] = (
        df["session_type"]
        .astype(str)
        .str.strip()
        .replace({
            "Practice 1": "FP1",
            "Practice 2": "FP2",
            "Practice 3": "FP3",
            "Qualifying": "Q",
            "Race": "R",
            "Sprint": "S",
            "Sprint Qualifying": "SQ",
            "Sprint Shootout": "SS",
        })
    )

    return df


# Replace missing driver_id with last_name
def fill_missing_driver_id(df):
    """
    Fills missing driver_id values with the corresponding last_name values.
    Args:
        df (pd.DataFrame): The DataFrame containing the driver_id and last_name columns.
    Returns:
        pd.DataFrame: The DataFrame with missing driver_id values filled."""
    df['driver_id'] = df.apply(lambda row: row['last_name'].lower() if pd.isna(row['driver_id']) else row['driver_id'], axis=1)
    return df