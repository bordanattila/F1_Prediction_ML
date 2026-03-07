import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


class FreePracticeNormalizer:
    def __init__(self, processed_data_dir: str, normalize_data_dir: str):
        self.processed_data_dir = processed_data_dir
        self.normalize_data_dir = normalize_data_dir
        os.makedirs(self.normalize_data_dir, exist_ok=True)

    def normalize_free_practice_data(self, df):
        """
        Normalizes the free practice data by creating new columns for free practice performance, 'free_practice_best_lap_sec', 'free_practice_delta_to_session_best', 
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

        return active_df