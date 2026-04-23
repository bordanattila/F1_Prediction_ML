"""
Dedicated inference pipeline: fetches and processes a single race weekend in memory,
producing a feature DataFrame suitable for the WinnerPredictor — without touching
the shared training files (session lists, master CSVs).
"""

import sys
from pathlib import Path
import pandas as pd
import fastf1

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from data.raw.raw_data_collector import RawDataCollector
from data.interim.utils.data_organizer_utils import (
    standardize_column_names,
    standardize_session_names,
    fill_missing_driver_id,
)
from data.interim.aggregators.weather_aggregate import aggregate_weather_data
from data.interim.aggregators.track_status_aggregate import aggregate_track_status_data
from data.interim.aggregators.laps_aggregate import aggregate_laps_data
from f1_prediction_ml.ml_utils import remove_unnecessary_columns
from f1_prediction_ml.normalize.normalize_free_practice import FreePracticeNormalizer
from f1_prediction_ml.normalize.normalize_quali import QualifyingNormalizer
from f1_prediction_ml.features.features_free_practice import FreePracticeFeatures
from f1_prediction_ml.features.features_quali import QualifyingFeatures
from f1_prediction_ml.colors import CYAN, MAGENTA, RESET

BASE_ID_COLS = [
    'year', 'grand_prix', 'event_id', 'row_id',
    'driver_id', 'abbreviation', 'team_name', 'team_id',
]

SESSION_METRICS = [
    'lap_count', 'lap_mean', 'lap_std', 'lap_best', 'lap_median',
    'air_temp_mean', 'air_temp_std',
    'track_temp_mean', 'track_temp_std',
    'humidity_mean', 'humidity_std',
    'wind_speed_mean', 'wind_speed_std',
    'rain_any', 'rain_samples_ratio',
    'yellow_count', 'red_count',
    'vsc_deployed_count', 'sc_deployed_count',
    'disruption_ratio',
]

KEYS = ['event_id', 'driver_id']

RESULTS_DROP_COLS = [
    'unnamed:_0', 'country_code', 'headshot_url',
    'first_name', 'last_name', 'broadcast_name', 'full_name', 'time',
]

NORMALIZE_DROP_COLS = [
    'vsc_duration', 'vsc_ending_duration', 'sc_duration',
    'not_green_duration', 'total_duration',
]

DEFAULT_SESSION_TYPES = ['FP1', 'FP2', 'FP3', 'Q']
SPRINT_SESSION_TYPES = ['FP1', 'SQ', 'S', 'SS','Q']

SESSION_TYPE_TO_PREFIX = {
    'FP1': 'fp1_', 'FP2': 'fp2_', 'FP3': 'fp3_',
    'Q': 'quali_', 'SQ': 'sprint_quali_', 'S': 'sprint_', 'SS': 'sprint_shootout_',
}


def _organize_in_memory(raw_data: dict) -> pd.DataFrame:
    """
    Replicate the DataOrganizer logic in memory, operating on the DataFrames
    returned by RawDataCollector.fetch_session_data().

    Returns:
        A single organized DataFrame (same schema as the on-disk organized CSVs).
    """
    laps_df = standardize_column_names(raw_data['laps'].copy())
    weather_df = standardize_column_names(raw_data['weather_data'].copy())
    results_df = standardize_column_names(raw_data['results'].copy())
    track_status_df = standardize_column_names(raw_data['track_status'].copy())
    session_info_df = standardize_column_names(raw_data['session_info'].copy())

    fill_missing_driver_id(results_df)
    standardize_session_names(session_info_df)

    session_key = session_info_df['session_key'].iloc[0]
    for df in [laps_df, weather_df, results_df, track_status_df]:
        df['session_key'] = session_key

    results_df = remove_unnecessary_columns(results_df, RESULTS_DROP_COLS)

    weather_agg = aggregate_weather_data(weather_df)
    track_status_agg = aggregate_track_status_data(track_status_df)
    laps_agg = aggregate_laps_data(laps_df)

    merged = (
        laps_agg
        .merge(weather_agg, on='session_key', how='left')
        .merge(results_df, on=['session_key', 'driver_number'], how='left')
        .merge(track_status_agg, on='session_key', how='left')
        .merge(session_info_df, on='session_key', how='left')
    )
    return merged


def _normalize(organized_df: pd.DataFrame, session_type: str) -> pd.DataFrame:
    """Apply the correct normalizer based on session type."""
    df = remove_unnecessary_columns(organized_df.copy(), NORMALIZE_DROP_COLS)

    if session_type in ('Q', 'SQ'):
        return QualifyingNormalizer().normalize_quali_data(df)
    if session_type in ('FP1', 'FP2', 'FP3', 'S', 'SS'):
        return FreePracticeNormalizer().normalize_free_practice_data(df)
    raise ValueError(f"Unsupported session type for inference: {session_type}")


def _extract_features(normalized_df: pd.DataFrame, session_type: str) -> pd.DataFrame:
    """Apply the correct feature builder based on session type."""
    if session_type in ('Q', 'SQ'):
        return QualifyingFeatures().create_quali_features(
            normalized_df, SESSION_METRICS, BASE_ID_COLS
        )
    if session_type in ('FP1', 'FP2', 'FP3', 'S', 'SS'):
        return FreePracticeFeatures().create_free_practice_features(
            normalized_df, SESSION_METRICS, BASE_ID_COLS
        )
    raise ValueError(f"Unsupported session type for inference: {session_type}")


def build_inference_features(
    year: int,
    grand_prix: str,
    session_types: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build a feature DataFrame for a single upcoming race weekend, entirely in memory.

    Fetches data via FastF1, organizes, normalizes, extracts features, and merges
    all pre-race sessions into one row per driver — without writing to shared
    training files or cumulative session lists.

    Args:
        year: Season year (e.g. 2024).
        grand_prix: Grand Prix name (e.g. 'Hungary').
        session_types: Session codes to include. Defaults to ['FP1', 'FP2', 'FP3', 'Q'].
    Returns:
        pd.DataFrame with one row per driver, containing identity columns and all
        available prefixed feature columns.
    """
    collector = RawDataCollector(cache_dir=str(project_root))

    if session_types is None:
        try:
            event = fastf1.get_event(year, grand_prix)
            event_format = event['EventFormat']
            print(f"{CYAN}INFO: Detected EventFormat = '{event_format}'{RESET}")
        except Exception as e:
            print(f"{MAGENTA}WARNING: Could not detect event format, defaulting to conventional: {e}{RESET}")
            event_format = 'conventional'

        if event_format in ('sprint', 'sprint_shootout', 'sprint_qualifying'):
            print(f"{CYAN}INFO: Sprint weekend detected ({event_format}) — using {SPRINT_SESSION_TYPES}{RESET}")
            session_types = list(SPRINT_SESSION_TYPES)
        else:
            session_types = list(DEFAULT_SESSION_TYPES)
    session_features: dict[str, pd.DataFrame] = {}

    for ses_type in session_types:
        try:
            raw_data = collector.fetch_session_data(year, grand_prix, ses_type)
            if raw_data is None:
                raise ValueError("No data returned")
        except Exception as e:
            print(f"{MAGENTA}WARNING: Could not fetch {year} {grand_prix} {ses_type}: {e}{RESET}")
            continue

        organized = _organize_in_memory(raw_data)
        normalized = _normalize(organized, ses_type)
        features = _extract_features(normalized, ses_type)
        session_features[ses_type] = features
        print(f"{CYAN}INFO: Processed {ses_type} — {len(features)} drivers{RESET}")

    if not session_features:
        raise ValueError(f"No sessions could be processed for {year} {grand_prix}")

    # Pick the base table: qualifying has every race entrant and their identity columns
    base_key = next(
        (k for k in ('Q', 'SQ') if k in session_features),
        next(iter(session_features)),
    )
    result = session_features.pop(base_key).copy()

    prefix = SESSION_TYPE_TO_PREFIX.get(base_key, '')
    for ses_type, feat_df in session_features.items():
        p = SESSION_TYPE_TO_PREFIX.get(ses_type, '')
        merge_cols = KEYS + [c for c in feat_df.columns if c.startswith(p)]
        result = result.merge(feat_df[merge_cols], on=KEYS, how='left')

    print(f"{CYAN}INFO: Inference features ready — {len(result)} drivers, {len(result.columns)} columns{RESET}")
    return result
