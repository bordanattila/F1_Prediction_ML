from f1_prediction_ml.ml_utils import convert_time_columns_to_seconds, remove_unnecessary_columns, create_row_id

class QualifyingNormalizer:
    def normalize_quali_data(self, df):
        """
        Normalizes qualifying data by converting time columns to seconds and removing columns that are not relevant for free practice performance.
        Creates new features for qualifying performance reached_q1, reached_q2, reached_q3, best_quali_seconds, quali_delta_to_pole, and quali_percent_of_pole.

        Args:
            df (pd.DataFrame): The input DataFrame containing qualifying data.
        Returns:
            pd.DataFrame: The normalized DataFrame with time columns converted to seconds and missing values handled.
        """

        active_df = df.copy()
        # Convert qualifying time columns to seconds
        quali_cols = ['q1', 'q2', 'q3', 'vsc_duration', 'vsc_ending_duration', 'sc_duration', 'not_green_duration', 'total_duration']
        convert_time_columns_to_seconds(active_df, quali_cols)

        # Remove columns that are not relevant for qualifying performance
        columns_to_remove = ['grid_position', 'classified_position', 'status', 'laps', 'points', 'unnamed:_0', 'q1', 'q2', 'q3']
        active_df = remove_unnecessary_columns(active_df, columns_to_remove)

        # Rename position column to quali_finish_position
        active_df.rename(columns={'position': 'quali_finish_position'}, inplace=True)

        # Create new columns for qualifying performance
        active_df['reached_q1'] = active_df['q1_seconds'].notna().astype(int)
        active_df['reached_q2'] = active_df['q2_seconds'].notna().astype(int)
        active_df['reached_q3'] = active_df['q3_seconds'].notna().astype(int)

        # Determine best qualifying time in seconds for the driver
        active_df['best_quali_seconds'] = active_df['q3_seconds'].combine_first(active_df['q2_seconds']).combine_first(active_df['q1_seconds'])
        # Get pole position time for the event
        pole = active_df['best_quali_seconds'].min()

        # Create new columns for qualifying performance relative to pole position
        active_df['quali_delta_to_pole'] = active_df['best_quali_seconds'] - pole
        active_df['quali_percent_of_pole'] = active_df['best_quali_seconds'] / pole * 100

        create_row_id(active_df)

        return active_df