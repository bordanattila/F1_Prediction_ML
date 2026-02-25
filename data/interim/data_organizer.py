import pandas as pd
import os
from data.interim.utils.utils import standardize_cols, drop_columns, create_finished_position_column, CYAN, RESET
from data.interim.aggregators.weather_aggregate import aggregate_weather_data 
from data.interim.aggregators.track_status_aggregate import aggregate_track_status_data
from data.interim.aggregators.laps_aggregate import aggregate_laps_data


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
        laps_df = standardize_cols(laps_df)
        weather_df = standardize_cols(weather_df)
        results_df = standardize_cols(results_df)
        track_status_df = standardize_cols(track_status_df)
        session_info_df = standardize_cols(session_info_df)        

        # Merge data from cv files under a shared key into one DataFrame
        print(f'{CYAN}INFO: Confirm session_key to dataframes{RESET}')
        print(session_info_df.head())
        print(f'{CYAN}Session Key:', session_info_df.copy()['session_key'].iloc[0], f'{RESET}')
        
        print(f'{CYAN}INFO: Merging dataframes by key{RESET}')
        for merged_df in [laps_df, weather_df, results_df, track_status_df]:
            merged_df['session_key'] = session_info_df['session_key'].iloc[0]

        print(f'{CYAN}INFO: Results DataFrame{RESET}')
        print(f'{CYAN}************ Results DataFrame Head ***********{RESET}')
        print(results_df.head())
        
        results_df = drop_columns(results_df)

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
        list_of_files = os.path.join(self.organized_data_dir, 'list_of_files.csv')
        filename = f'{year}_{grand_prix}_{session_type}_organized.csv'
        if os.path.exists(list_of_files):
            existing_files_df = pd.read_csv(list_of_files)
            new_file_entry = pd.DataFrame({'filename': [filename]})
            updated_files_df = pd.concat([existing_files_df, new_file_entry], ignore_index=True)
            updated_files_df.to_csv(list_of_files, index=False)
        else:
            new_file_entry = pd.DataFrame({'filename': [filename]})
            new_file_entry.to_csv(list_of_files, index=False)
        print(f'{CYAN}INFO: Added {filename} to list of processed files{RESET}')
        

        return merged_df
