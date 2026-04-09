from f1_prediction_ml.ml_utils import remove_unnecessary_columns, create_row_id
import re

class RaceNormalizer:
    """Cleans race data and derives outcome columns (DNF, laps down, pit-lane start, winner flag)."""

    def normalize_race_data(self, df):
        """ 
        Normalizes the race data by removing columns that are not relevant for race performance.
        Creates new features for race performance, 'race_finish_position', 'is_dnf', 'laps_down', 'started_from_pit_lane', is_winner.

        Args:
            df (pd.DataFrame): The DataFrame to be normalized.
        Returns:
            pd.DataFrame: The normalized DataFrame.
        """
        active_df = df.copy()
        # Remove columns that are not relevant for race performance
        columns_to_remove = ['q1', 'q2', 'q3', 'vsc_duration', 'vsc_ending_duration', 'sc_duration', 'not_green_duration', 'total_duration', 'unnamed:_0']
        active_df = remove_unnecessary_columns(active_df, columns_to_remove)
        
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

        return active_df