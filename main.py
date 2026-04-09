import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(project_root))   

from f1_prediction_ml.colors import CYAN, GREEN, RESET
from f1_prediction_ml.pipelines.collect_raw_data_pipeline import raw_data_collection_pipeline
from f1_prediction_ml.pipelines.raw_data_processor import raw_data_processing_pipeline
from f1_prediction_ml.pipelines.normalizer import normalize_data, concatenate_master_log_csv_files
from f1_prediction_ml.pipelines.features_engineering import create_features_per_session, concatenate_model_training_data
from f1_prediction_ml.ml_utils import get_list_of_sessions
from f1_prediction_ml.evaluation.model_evaluation import model_evaluation, inspect_feature_importance


session_year = 2022
sessions = ['Emilia-Romagna', 'Bahrain', 'Miami', 'Azerbaijan', 'Spain', 'Monaco', 'Canada', 'Australia', 'Austria', 'Great Britain', 'Netherlands', 'Germany',
         'Hungary', 'Belgium', 'Italy','Singapore', 'Japan', 'United States', 'Mexico', 'Brazil', 'Abu Dhabi', 'Saudi Arabia', 'France']
# sessions = ['Australia', 'Bahrain', 'China', 'Azerbaijan', 'Spain', 'Monaco', 'Canada', 'France', 'Austria', 'Great Britain', 'Germany',
        #  'Hungary', 'Belgium', 'Italy', 'Singapore', 'Russia', 'Japan', 'United States', 'Mexico', 'Brazil', 'Abu Dhabi']
session_type = ['FP1', 'FP2', 'FP3', 'SQ', 'Q', 'S', 'SS', 'R']


raw_data_collection_pipeline(session_year, sessions, session_type)
raw_data_processing_pipeline(session_year, sessions, session_type)

list_of_sessions = get_list_of_sessions('list_of_organized_files.csv')

normalize_data(list_of_sessions)

list_of_sessions = get_list_of_sessions('list_of_normalized_files.csv')

create_features_per_session(list_of_sessions)

list_of_sessions = get_list_of_sessions('list_of_feature_ready_files.csv')

concatenate_master_log_csv_files(list_of_sessions)

list_of_sessions = get_list_of_sessions('list_of_session_master_files.csv')

concatenate_model_training_data(list_of_sessions)

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

df = pd.read_csv('./data/processed/model_training_data.csv')


_, rfc = model_evaluation(df, RandomForestClassifier(n_estimators=100, random_state=42))
_, gbc = model_evaluation(df, GradientBoostingClassifier(n_estimators=100, random_state=42))

inspect_feature_importance(rfc)
inspect_feature_importance(gbc)

import pandas as pd

def pole_sitter_baseline(df):
    """
    Compute a naive baseline: predict the pole-sitter wins every race.

    Args:
        df: Model training DataFrame with qualifying and race result columns.
    Returns:
        float: Fraction of races where the pole-sitter actually won.
    """
    data = df.copy()

    required_cols = ['event_id', 'quali_quali_finish_position', 'race_is_winner']
    data = data.dropna(subset=required_cols)

    pole_sitters = (
        data.sort_values(['event_id', 'quali_quali_finish_position'], ascending=[True, True])
            .groupby('event_id', as_index=False)
            .first()
    )

    winner_pick_accuracy = pole_sitters['race_is_winner'].mean()

    print(f'{CYAN}Pole-sitter baseline winner pick accuracy:{RESET} {GREEN}{winner_pick_accuracy:.4f}{RESET}')
    print(f'{CYAN}Number of events:{RESET} {GREEN}{len(pole_sitters)}{RESET}')
    print(f'{CYAN}\nSample predictions:{RESET}')
    display_cols = ['event_id', 'quali_quali_finish_position', 'race_is_winner']
    if 'driver_id' in pole_sitters.columns:
        display_cols.insert(1, 'driver_id')
    if 'abbreviation' in pole_sitters.columns:
        display_cols.insert(2, 'abbreviation')

    print(pole_sitters[display_cols].head(10))

    return winner_pick_accuracy

pole_acc = pole_sitter_baseline(df)
print(pole_acc)