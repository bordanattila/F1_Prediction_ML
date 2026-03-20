import pandas as pd
import os
from pathlib import Path
from data.interim.utils.data_organizer_utils import standardize_column_names, standardize_session_names, fill_missing_driver_id, CYAN, RESET
from data.interim.aggregators.weather_aggregate import aggregate_weather_data 
from data.interim.aggregators.track_status_aggregate import aggregate_track_status_data
from data.interim.aggregators.laps_aggregate import aggregate_laps_data
from f1_prediction_ml.ml_utils import remove_unnecessary_columns, create_list_of_sessions_file


class DataOrganizer:
    def __init__(self, raw_data_dir: str, organized_data_dir: str):
        self.raw_data_dir = raw_data_dir
        self.organized_data_dir = organized_data_dir
        os.makedirs(self.organized_data_dir, exist_ok=True)

    def organize_session_data(self, year: int, grand_prix: str, session_type: str):
        """
        Organizes raw session data into a structured format.

        Args:
            year (int): The year of the season.
            grand_prix (str): The name of the grand prix.
            session_type (str): The type of session ('FP1', 'FP2', 'FP3', 'Q', 'S', 'SS', 'SQ', 'R').
        Return:
            pd.DataFrame: A DataFrame containing the organized session data.
        """
        grand_prix = grand_prix.replace(' ', '_')

        # Define file paths
        laps_file = os.path.join(self.raw_data_dir, f'{year}_{grand_prix}_{session_type}_laps.csv')
        weather_file = os.path.join(self.raw_data_dir, f'{year}_{grand_prix}_{session_type}_weather.csv')
        results_file = os.path.join(self.raw_data_dir, f'{year}_{grand_prix}_{session_type}_results.csv')
        track_status_file = os.path.join(self.raw_data_dir, f'{year}_{grand_prix}_{session_type}_track_status.csv')
        session_info_file = os.path.join(self.raw_data_dir, f'{year}_{grand_prix}_{session_type}_session_info.csv')           

        # Load raw data
        laps_df = pd.read_csv(laps_file)
        weather_df = pd.read_csv(weather_file)
        results_df = pd.read_csv(results_file)
        track_status_df = pd.read_csv(track_status_file)
        session_info_df = pd.read_csv(session_info_file)

        # Standardize column names
        laps_df = standardize_column_names(laps_df)
        weather_df = standardize_column_names(weather_df)
        results_df = standardize_column_names(results_df)
        track_status_df = standardize_column_names(track_status_df)
        session_info_df = standardize_column_names(session_info_df)  

        # Fill missing driver_id with last_name in results_df
        results_df = fill_missing_driver_id(results_df)

        # Standardize session names
        session_info_df = standardize_session_names(session_info_df)      

        # Merge data from cv files under a shared key into one DataFrame
        print(f'{CYAN}INFO: Confirm session_key to dataframes{RESET}')
        print(session_info_df.head())
        print(f'{CYAN}Session Key:', session_info_df.copy()['session_key'].iloc[0], f'{RESET}')
        
        print(f'{CYAN}INFO: Merging dataframes by key{RESET}')
        for merged_df in [laps_df, weather_df, results_df, track_status_df]:
            merged_df['session_key'] = session_info_df['session_key'].iloc[0]

        print(f'{CYAN}INFO: Results DataFrame{RESET}')
        print(f'{CYAN}************ Results DataFrame Head ***********{RESET}')
        
        # Drop unnecessary columns from results_df before merging
        cols = ['unnamed:_0', 'country_code', 'headshot_url', 'first_name', 'last_name', 'broadcast_name', 'full_name', 'time']
        results_df = remove_unnecessary_columns(results_df, cols)

        weather_aggregated = aggregate_weather_data(weather_df)
        track_status_aggregated = aggregate_track_status_data(track_status_df)
        laps_aggregated = aggregate_laps_data(laps_df)

        # Merge all organized data into a single DataFrame
        print(f'{CYAN}INFO: Merging all organized data into a single DataFrame{RESET}')
        merged_df = laps_aggregated.merge(weather_aggregated, on='session_key', how='left') \
            .merge(results_df, on=['session_key', 'driver_number'], how='left') \
            .merge(track_status_aggregated, on='session_key', how='left') \
            .merge(session_info_df, on='session_key', how='left')
        print(f'{CYAN}INFO: Merged DataFrame info:{RESET}')
        print(merged_df.info())
        print(f'{CYAN}INFO: Merged DataFrame head:{RESET}')
        print(merged_df.head())


        # Save new DataFrame to a csv file in organized_data_dir for processing
        output_file = os.path.join(self.organized_data_dir, f'{year}_{grand_prix}_{session_type}_organized.csv')
        merged_df.to_csv(output_file, index=False)
        print(f'{CYAN}INFO: Organized data saved to {output_file}{RESET}')

        # Save list of processed session for further processing
        filename = f'{year}_{grand_prix}_{session_type}'
        project_root = Path(__file__).resolve().parents[2]
        os.makedirs(project_root / 'data' / 'list_of_available_sessions', exist_ok=True)
        target_data_dir = project_root / 'data' / 'list_of_available_sessions'
        create_list_of_sessions_file(target_data_dir, 'list_of_organized_files.csv', filename, source='organized')

        return merged_df
