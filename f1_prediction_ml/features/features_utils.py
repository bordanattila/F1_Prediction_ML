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
    # Create row_id by concatenating event_id, session_type and driver_number
    df['row_id'] = df['event_id'] + '_' + df['session_type'] + '_' + df['driver_number'].astype(str)

    return df