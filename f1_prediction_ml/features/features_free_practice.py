from f1_prediction_ml.features.features_utils import create_row_id

class FreePracticeFeatures:
    def create_free_practice_features(self, df):
        """
        Creates new features for free practice performance, 'free_practice_best_lap_sec', 'free_practice_delta_to_session_best', 
        and 'free_practice_percent_of_session_best'.

        Args:
            df (pd.DataFrame): The DataFrame to be normalized.
        Returns:
            pd.DataFrame: The normalized DataFrame.
        """
        active_df = df.copy()

        # Determine best free practice time in seconds for the driver
        active_df['best_free_practice_sec'] = active_df['lap_best'].min()

        # Create new columns for free practice performance relative to best in the session
        active_df['free_practice_delta_to_best_lap_sec'] = active_df['lap_best'] - active_df['best_free_practice_sec']
        active_df['free_practice_percent_of_best_lap_sec'] = active_df['lap_best'] / active_df['best_free_practice_sec'] * 100

        create_row_id(active_df)
        
        return active_df