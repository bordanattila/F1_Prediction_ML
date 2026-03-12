from f1_prediction_ml.ml_utils import convert_time_columns_to_seconds, remove_unnecessary_columns

class QualifyingNormalizer:
    def normalize_quali_data(self, df):
        """
        Normalizes qualifying data by converting time columns to seconds and removing columns that are not relevant for free practice performance.
        
        Args:
            df (pd.DataFrame): The input DataFrame containing qualifying data.
        Returns:
            pd.DataFrame: The normalized DataFrame with time columns converted to seconds and missing values handled.
        """

        active_df = df.copy()
        # Convert qualifying time columns to seconds
        quali_cols = ['q1', 'q2', 'q3']
        convert_time_columns_to_seconds(active_df, quali_cols)

        # Remove columns that are not relevant for qualifying performance
        columns_to_remove = ['grid_position', 'classified_position', 'status', 'laps', 'points', 'unnamed:_0', 'q1', 'q2', 'q3']
        active_df = remove_unnecessary_columns(active_df, columns_to_remove)

        # Rename position column to quali_finish_position
        active_df.rename(columns={'position': 'quali_finish_position'}, inplace=True)

        return active_df