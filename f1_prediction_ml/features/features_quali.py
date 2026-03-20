from f1_prediction_ml.features.features_utils import create_row_id

QUALI_SEESION_METRICS = ['reached_q1', 'reached_q2', 'reached_q3', 'best_quali_seconds', 'quali_delta_to_pole', 'quali_percent_of_pole']

class QualifyingFeatures:
    def create_quali_features(self, df, session_metrics, base_id_cols):
        """
        Creates new features for qualifying performance reached_q1, reached_q2, reached_q3, best_quali_seconds, quali_delta_to_pole, and quali_percent_of_pole.

        Args:
            df (pd.DataFrame): The DataFrame to be normalized.
        Returns:
            pd.DataFrame: The normalized DataFrame.
        """

        session_metrics = session_metrics + QUALI_SEESION_METRICS

        active_df = df.copy()
        
        create_row_id(active_df)

        ses_type = active_df['session_type'].iloc[0]   
        prefix_map = {'Q': 'quali_', 'SQ': 'sprint_quali_'}
        prefix = prefix_map.get(ses_type)

        if prefix is None:
            raise ValueError(f'Unexpected session_type: {ses_type}')

        # # Create the prefixed columns on active_df
        active_df[[f'{prefix}{c}' for c in session_metrics]] = active_df[session_metrics]

        # Create a NEW df that only has identifiers + the new prefixed columns
        new_cols = base_id_cols + [f'{prefix}{c}' for c in session_metrics]

        prefixed_df = active_df[new_cols].copy()

        
        
        return prefixed_df
    