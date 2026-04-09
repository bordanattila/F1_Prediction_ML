"""
This is a script file that loads the csv files from raw/csv_files and utilizes the DataOrganizer class to organize them.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from data.interim.data_organizer import DataOrganizer

from f1_prediction_ml.colors import CYAN, YELLOW, MAGENTA, RESET

data_organizer = DataOrganizer(
    raw_data_dir=str(project_root / 'data/raw/raw_csv_files'),
    organized_data_dir=str(project_root / 'data/interim/organized_csv_files')
)

def raw_data_processing_pipeline(session_year, sessions, session_type):
    """
    Organize raw CSV files into structured session DataFrames using DataOrganizer.

    Args:
        session_year: Season year (e.g. 2022).
        sessions: List of Grand Prix names (e.g. ['Bahrain', 'Monaco']).
        session_type: List of session codes to process (e.g. ['FP1', 'Q', 'R']).
    """
    print(f'{MAGENTA}********** Session: {sessions} **********{RESET}')
    print(f'{MAGENTA}********** Session Type: {session_type} **********{RESET}')
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
        
