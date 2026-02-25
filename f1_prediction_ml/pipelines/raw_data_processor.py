"""
This is a script file that loads the csv files from raw/csv_files and utilizes the DataOrganizer class to organize them.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from data.interim.data_organizer import DataOrganizer

# Color codes
CYAN = "\033[36m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
RESET = "\033[0m"

# session_year = 2018
# sessions = ['Australia', 'Bahrain', 'China', 'Azerbaijan', 'Spain', 'Monaco', 'Canada', 'France', 'Austria', 'Great Britain', 'Germany',
#          'Hungary', 'Belgium', 'Italy', 'Singapore', 'Russia', 'Japan', 'United States', 'Mexico', 'Brazil', 'Abu Dhabi']
# session_type = ['FP1', 'FP2', 'FP3', 'SQ', 'Q', 'S', 'SS', 'R']

data_organizer = DataOrganizer(
    raw_data_dir=str(project_root / 'data/raw/raw_csv_files'),
    organized_data_dir=str(project_root / 'data/interim/organized_csv_files')
)

def raw_data_processing_pipeline(session_year, sessions, session_type):
    # Iterate through the list of races and session types
    for session_name in sessions:
        session_name = session_name.replace(' ', '_')
        print(f'{CYAN}********** Processing session: {session_name} **********{RESET}')
        for ses_type in session_type:
            try:
                print(f'{CYAN}********** Organizing data for session type: {ses_type} **********{RESET}')
                organized_data = data_organizer.organize_session_data(session_year, session_name, ses_type)
                
                if organized_data is None:
                    raise ValueError("No data found for this session type.")
                
            except Exception as e:
                print(f"{YELLOW}WARNING: Could not organize data for {session_year} {session_name} {ses_type} session. Error: {e}{RESET}")
                continue
        
