from f1_prediction_ml.ml_utils import remove_unnecessary_columns

class FreePracticeNormalizer:
    def normalize_free_practice_data(self, df):
        """
        Normalizes the free practice data by removing columns that are not relevant for free practice performance.

        Args:
            df (pd.DataFrame): The DataFrame to be normalized.
        Returns:
            pd.DataFrame: The normalized DataFrame.
        """
        active_df = df.copy()
        # Remove columns that are not relevant for free practice performance
        columns_to_remove = ['grid_position', 'classified_position','position', 'status', 'laps', 'points', 'unnamed:_0', 'q1', 'q2', 'q3']
        active_df = remove_unnecessary_columns(active_df, columns_to_remove)

        return active_df