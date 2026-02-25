import os
import sys
from pathlib import Path
import re

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from f1_prediction_ml.normalize.utils import get_list_of_sessions
from f1_prediction_ml.normalize.utils import convert_time_columns_to_seconds


class Race_Normalizer:
    def __init__(self, processed_data_dir: str, normalize_data_dir: str):
        self.processed_data_dir = processed_data_dir
        self.normalize_data_dir = normalize_data_dir
        os.makedirs(self.normalize_data_dir, exist_ok=True)

    def normalize_race_data(self, df):
        """
        Normalizes the race data by creating new columns for race performance, 'race_finish_position', 'is_dnf', 'laps_down', 'started_from_pit_lane'.

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
        active_df['started_from_pitlane'] = active_df['grid'].apply(lambda x: 1 if x == 'pit' else 0)   