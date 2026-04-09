import pandas as pd
import os
import fastf1 as f1

from f1_prediction_ml.colors import CYAN, RESET


class RawDataCollector:
    """Fetches raw session data (laps, weather, results, track status) from the FastF1 API."""

    def __init__(self, cache_dir: str):
        """
        Args:
            cache_dir: Directory used for FastF1 HTTP cache.
        """
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        # FastF1 caching setup
        f1.Cache.enable_cache(self.cache_dir)

    def fetch_session_data(self, year: int, grand_prix: str, session_type: str) -> pd.DataFrame:
        """
        Fetches session data for a given year, grand prix, and session type using FastF1.

        Args:
            year (int): The year of the season.
            grand_prix (str): The name of the grand prix.
            session_type (str): The type of session ('FP1', 'FP2', 'FP3', 'Q', 'S', 'SS', 'SQ', 'R').  

        Returns:
            pd.DataFrame: DataFrame containing the session laps data.    
        """

        # Fetch session data      
        session = f1.get_session(year, grand_prix, session_type)
        session.load(laps=True, weather=True, telemetry=False, messages=False,)
        print(f"{CYAN}INFO: Fetched data for {year} {grand_prix} {session_type} session.{RESET}")

        print(f"{CYAN}********** Session Info **********{RESET}")
        print(session.session_info)

        session_info_short = {
            'GrandPrix': session.session_info['Meeting']['Name'],
            'Date': session.session_info['StartDate'],
            'RequestedSessionCode': session_type,                  # session requested from FastF1 
            'ResolvedSessionName': session.session_info['Name'],   # what FastF1 actually returned
            'ResolvedSessionType': session.session_info['Type'],   # Practice / Qualifying / Race
            'SessionType': session.session_info['Name'],
            'SessionKey': session.session_info['Key'],             # UNIQUE session key
            'MeetingKey': session.session_info['Meeting']['Key'],  # weekend key
            'CountryCode': session.session_info['Meeting']['Country']['Code'],
            'CountryName': session.session_info['Meeting']['Country']['Name'],
        }

        # session_info_short = {
        #     'GrandPrix': session.session_info['Meeting']['Name'],
        #     'Date': session.session_info['StartDate'],
        #     'SessionType': session.session_info['Name'],
        #     'SessionKey': session.session_info['Meeting']['Key'],
        #     'CountryCode': session.session_info['Meeting']['Country']['Code'],
        #     'CountryName': session.session_info['Meeting']['Country']['Name'],
        # }
        return {
            'laps': session.laps.copy() if session.laps is not None else None,
            'weather_data': session.weather_data.copy() if session.weather_data is not None else None,
            'results': session.results.copy() if session.results is not None else None,
            'track_status': session.track_status.copy() if session.track_status is not None else None,
            'session_info': pd.DataFrame([session_info_short]).copy() if session_info_short is not None else None
        }




