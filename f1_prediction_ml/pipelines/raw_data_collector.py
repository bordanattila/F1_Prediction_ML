"""
This is a script that iterates through the list of races and utilises the RawDataCollector class to pull data from FastF1.
"""
import sys
from pathlib import Path
import time
import fastf1 as f1

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from data.raw.raw_data_collector import RawDataCollector

# Color codes
CYAN = "\033[36m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
RESET = "\033[0m"

# Map session type codes to descriptive names
FREE_PRACTICE_SESSION_TYPE_MAP = {
    'FP1': 'Practice_1',
    'FP2': 'Practice_2',
    'FP3': 'Practice_3',
}

# session_year = 2018
# sessions = ['Australia', 'Bahrain', 'China', 'Azerbaijan', 'Spain', 'Monaco', 'Canada', 'France', 'Austria', 'Great Britain', 'Germany',
#          'Hungary', 'Belgium', 'Italy', 'Singapore', 'Russia', 'Japan', 'United States', 'Mexico', 'Brazil', 'Abu Dhabi']
# session_type = ['FP1', 'FP2', 'FP3', 'SQ', 'Q', 'S', 'SS', 'R']

data_collector = RawDataCollector(cache_dir=str(project_root))

def raw_data_collection_pipeline(session_year, sessions, session_type):
    # Iterate through the list of races and session types
    for session_name in sessions:
        session_name = session_name.replace(' ', '_')
        print(f'{CYAN}********** Processing session: {session_name} **********{RESET}')
        for ses_type in session_type:
            try:
                print(f'{CYAN}********** Fetching data for session type: {ses_type} **********{RESET}')
                session_data = data_collector.fetch_session_data(session_year, session_name, ses_type)
                
                if session_data is None:
                    raise ValueError("No data found for this session type.")
                
            except f1.req.RateLimitExceededError as e:
                print(f"{MAGENTA}RATE LIMIT hit for {session_year} {session_name} {ses_type}: {e}{RESET}")
                time.sleep(30)  # backoff
                continue
                
            except ValueError as ve:
                print(f"{MAGENTA}WARNING: {ve} for {session_year} {session_name} {ses_type} session.{RESET}")
                continue

            except Exception as e:
                print(f"{YELLOW}WARNING: Could not fetch data for {session_year} {session_name} {ses_type} session. Error: {e}{RESET}")
                continue
            
            # Save the fetched data to CSV files
            output_dir = project_root / 'data/raw/raw_csv_files'
            session_data['laps'].to_csv(output_dir / f'{session_year}_{session_name}_{ses_type}_laps.csv')
            session_data['weather_data'].to_csv(output_dir / f'{session_year}_{session_name}_{ses_type}_weather.csv')
            session_data['results'].to_csv(output_dir / f'{session_year}_{session_name}_{ses_type}_results.csv')
            session_data['track_status'].to_csv(output_dir / f'{session_year}_{session_name}_{ses_type}_track_status.csv')
            # Map practice sessions to descriptive names (FP1 -> Practice_1, etc.)
            if ses_type in FREE_PRACTICE_SESSION_TYPE_MAP:
                session_data['session_info']['SessionType'] = FREE_PRACTICE_SESSION_TYPE_MAP[ses_type]
            session_data['session_info'].to_csv(output_dir / f'{session_year}_{session_name}_{ses_type}_session_info.csv')
            print(f"{CYAN}INFO: Saved data for {session_year} {session_name} {ses_type} session to CSV files.{RESET}")
        