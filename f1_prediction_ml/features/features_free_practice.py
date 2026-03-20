from f1_prediction_ml.features.features_utils import create_row_id

FP_SEESION_METRICS = ['best_free_practice_sec', 'free_practice_delta_to_best_lap_sec', 'free_practice_percent_of_best_lap_sec']

class FreePracticeFeatures:
    def create_free_practice_features(self, df, session_metrics, base_id_cols):
        """
        Creates new features for free practice sessions by prefixing the session metrics with the session type (e.g., 'fp1_lap_mean', 'fp2_lap_mean', etc.) and keeping only the relevant columns.

        Args:
            df (pd.DataFrame): The DataFrame containing the normalized free practice data.
            session_metrics (list): A list of session metrics to be prefixed and included in the final DataFrame.
            base_id_cols (list): A list of base identifier columns to be included in the final DataFrame.
        Returns:
            pd.DataFrame: A DataFrame containing the new features for free practice sessions.
        """
        session_metrics = session_metrics + FP_SEESION_METRICS

        active_df = df.copy()
        
        create_row_id(active_df)
        
        ses_type = active_df['session_type'].iloc[0]   
        prefix_map = {'FP1': 'fp1_', 'FP2': 'fp2_', 'FP3': 'fp3_'}
        prefix = prefix_map.get(ses_type)

        if prefix is None:
            raise ValueError(f'Unexpected session_type: {ses_type}')

        # # Create the prefixed columns on active_df
        active_df[[f'{prefix}{c}' for c in session_metrics]] = active_df[session_metrics]

        # Create a NEW df that only has identifiers + the new prefixed columns
        new_cols = base_id_cols + [f'{prefix}{c}' for c in session_metrics]

        prefixed_df = active_df[new_cols].copy()

        
        
        return prefixed_df