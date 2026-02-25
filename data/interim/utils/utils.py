# Color codes
CYAN = "\033[36m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
RESET = "\033[0m"

# Normalize common column names
def standardize_cols(df):
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

# Drop unnecessary columns
def drop_columns(df):
    """
    Drops unnecessary columns from the DataFrame.
    """
    cols = ['unnamed:_0', 'country_code', 'headshot_url', 'first_name', 'last_name', 'broadcast_name', 'full_name', 'time']
    return df.drop(columns=cols, errors='ignore')

# # Create a new column indicating finished position if 'status' is 'Finished' or '+X Laps'
# def create_finished_position_column(df):
#     """
#     Creates a new column 'finished_position' indicating the finishing position of the driver if 'status' is 'Finished' or '+X Laps'.
#     If 'status' is 'Finished', 'finished_position' is set to the value in the 'position' column.
#     If 'status' contains '+X Laps', 'finished_position' is set to the value in the 'position' column.
#     For all other statuses, 'finished_position' is set to NaN.
#     """
#     def calculate_finished_position(row):
#         if row['status'] == 'Finished':
#             return row['position']
#         elif isinstance(row['status'], str) and row['status'].startswith('+') and row['status'].endswith('Laps'):
#             try:
#                 laps_down = int(row['status'][1:-5].strip())
#                 return row['position'] 
#             except ValueError:
#                 return None
#         else:
#             return None

#     df['finished_position'] = df.apply(calculate_finished_position, axis=1)
#     return df