import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from f1_prediction_ml.normalize.normalization_utils import convert_time_columns_to_seconds

class QualifyingNormalizer:
    def __init__(self, processed_data_dir: str, normalize_data_dir: str):
        self.processed_data_dir = processed_data_dir
        self.normalize_data_dir = normalize_data_dir
        os.makedirs(self.normalize_data_dir, exist_ok=True)

    def normalize_quali_data(self, df):
        """
        Normalizes the data by creating new columns for qualifying performance reached_q1, reached_q2, reached_q3, best_quali_sec, quali_delta_to_pole, and quali_percent_of_pole.
        Creates a new column for starting from pit lane.

        Args:
            df (pd.DataFrame): The DataFrame to be normalized.
        Returns:
            pd.DataFrame: The normalized DataFrame.
        """

        active_df = df.copy()
        # Convert qualifying time columns to seconds
        quali_cols = ['q1', 'q2', 'q3']
        convert_time_columns_to_seconds(active_df, quali_cols)

        # Create new columns for qualifying performance
        active_df['reached_q1'] = active_df['q1'].notna().astype(int)
        active_df['reached_q2'] = active_df['q2'].notna().astype(int)
        active_df['reached_q3'] = active_df['q3'].notna().astype(int)

        # Determine best qualifying time in seconds for the driver
        active_df['best_quali_sec'] = active_df['q3_sec'].combine_first(active_df['q2_sec']).combine_first(active_df['q1_sec'])
        # Get pole position time for the event
        pole = active_df['best_quali_sec'].min()

        # Create new columns for qualifying performance relative to pole position
        active_df['quali_delta_to_pole'] = active_df['best_quali_sec'] - pole
        active_df['quali_percent_of_pole'] = active_df['best_quali_sec'] / pole * 100

        # Rename position column to quali_finish_position
        active_df.rename(columns={'position': 'quali_finish_position'}, inplace=True)

        return active_df
    