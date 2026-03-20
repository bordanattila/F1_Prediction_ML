import re
from f1_prediction_ml.features.features_utils import create_row_id

RACE_SESSION_METRICS = ['race_finish_position', 'is_dnf', 'laps_down', 'started_from_pit_lane', 'is_winner', 'finish_position_relative_to_grid_start_position']
class RaceFeatures:
    def create_race_features(self, df, session_metrics, base_id_cols):
        """
        Creates new features for race performance, 'race_finish_position', 'is_dnf', 'laps_down', 'started_from_pit_lane', is_winner.

        Args:
            df (pd.DataFrame): The DataFrame to be normalized.
        Returns:
            pd.DataFrame: The normalized DataFrame.
        """
        active_df = df.copy()
        
        # Rename position column to race_finish_position
        active_df.rename(columns={'position': 'race_finish_position'}, inplace=True)
        
        # Create new column for DNF status, 0=false, 1=true
        active_df['is_dnf'] = (active_df['classified_position'].astype(str) == "R").astype(int)
        
        # Create new column for laps down, if status is '+X Laps', extract X, otherwise set to 0
        # Matches: '+1 Lap' or '+2 Laps' (optionally with extra spaces)
        LAPS_DOWN_RE = re.compile(r'^\+\s*(\d+)\s*Lap(?:s)?\s*$')
        active_df['laps_down'] = active_df['status'].apply(lambda x: int(LAPS_DOWN_RE.match(x).group(1)) if isinstance(x, str) and LAPS_DOWN_RE.match(x) else 0)
        
        # Create new column for finishing position relative to starting grid position, only for drivers who finished the race
        active_df['finish_position_relative_to_grid_start_position'] = active_df.apply(
            lambda row: row['race_finish_position'] - row['grid_position'] if row['is_dnf'] == 0 else None, axis=1
            )
    
        # Create new column for starting from pit lane
        active_df['started_from_pit_lane'] = active_df['grid_position'].apply(lambda x: 1 if x == 'pit' else 0)

        # Create new column for winner, 1 if the driver won the race, 0 otherwise
        active_df['is_winner'] = (active_df['race_finish_position'] == 1).astype(int)   

        create_row_id(active_df)

        session_metrics = session_metrics + RACE_SESSION_METRICS

        ses_type = active_df['session_type'].iloc[0]   
        prefix_map = {'R': 'race_', 'S': 'sprint_', 'SS': 'sprint_shoot_'}
        prefix = prefix_map.get(ses_type)

        if prefix is None:
            raise ValueError(f'Unexpected session_type: {ses_type}')

        # # Create the prefixed columns on active_df
        active_df[[f'{prefix}{c}' for c in session_metrics]] = active_df[session_metrics]

        # Create a NEW df that only has identifiers + the new prefixed columns
        new_cols = base_id_cols + [f'{prefix}{c}' for c in session_metrics]

        prefixed_df = active_df[new_cols].copy()

        
        
        return prefixed_df 