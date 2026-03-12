from f1_prediction_ml.ml_utils import remove_unnecessary_columns

class RaceNormalizer:
    def normalize_race_data(self, df):
        """ 
        Normalizes the race data by removing columns that are not relevant for race performance.
        Args:
            df (pd.DataFrame): The DataFrame to be normalized.
        Returns:
            pd.DataFrame: The normalized DataFrame.
        """
        active_df = df.copy()
        # Remove columns that are not relevant for race performance
        columns_to_remove = ['q1', 'q2', 'q3', 'unnamed:_0']
        active_df = remove_unnecessary_columns(active_df, columns_to_remove)

        return active_df