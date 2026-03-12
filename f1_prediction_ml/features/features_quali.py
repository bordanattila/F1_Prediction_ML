from f1_prediction_ml.features.features_utils import create_row_id

class QualifyingFeatures:
    def create_quali_features(self, df):
        """
        Creates new features for qualifying performance reached_q1, reached_q2, reached_q3, best_quali_seconds, quali_delta_to_pole, and quali_percent_of_pole.

        Args:
            df (pd.DataFrame): The DataFrame to be normalized.
        Returns:
            pd.DataFrame: The normalized DataFrame.
        """

        active_df = df.copy()

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
    