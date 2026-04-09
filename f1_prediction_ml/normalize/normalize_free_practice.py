from f1_prediction_ml.ml_utils import remove_unnecessary_columns, create_row_id

class FreePracticeNormalizer:
    """Cleans free practice data and adds session-relative performance metrics (delta to best, percent of best)."""

    def normalize_free_practice_data(self, df):
        """
        Normalizes the free practice data by removing columns that are not relevant and create new columns for free practice performance.

        Args:
            df (pd.DataFrame): The DataFrame to be normalized.
        Returns:
            pd.DataFrame: The normalized DataFrame.
        """
        active_df = df.copy()
        # Remove columns that are not relevant for free practice performance
        columns_to_remove = ['grid_position', 'classified_position','position', 'status', 'laps', 'points', 'unnamed:_0', 'q1', 'q2', 'q3',
                             'vsc_duration', 'vsc_ending_duration', 'sc_duration', 'not_green_duration', 'total_duration']
        active_df = remove_unnecessary_columns(active_df, columns_to_remove)

        # Determine best free practice time in seconds for the driver
        active_df['best_free_practice_sec'] = active_df['lap_best'].min()

        # Create new columns for free practice performance relative to best in the session
        active_df['free_practice_delta_to_best_lap_sec'] = active_df['lap_best'] - active_df['best_free_practice_sec']
        active_df['free_practice_percent_of_best_lap_sec'] = active_df['lap_best'] / active_df['best_free_practice_sec'] * 100

        create_row_id(active_df)

        return active_df