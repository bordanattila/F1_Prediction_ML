import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(project_root))   


from f1_prediction_ml.pipelines.collect_raw_data_pipeline import raw_data_collection_pipeline
from f1_prediction_ml.pipelines.raw_data_processor import raw_data_processing_pipeline
from f1_prediction_ml.pipelines.normalizer import normalize_data, concatenate_master_log_csv_files
from f1_prediction_ml.pipelines.features_engineering import create_features_per_session, concatenate_model_training_data
from f1_prediction_ml.pipelines.evaluate_model import run_evaluation
from f1_prediction_ml.evaluation.pole_sitter_baseline import pole_sitter_baseline
from f1_prediction_ml.ml_utils import get_list_of_sessions
from f1_prediction_ml.modeling.train_post_quali_model import train_and_save_model

session_year = 2026
sessions = ['China']
# sessions = ['Emilia-Romagna', 'Bahrain', 'Miami', 'Azerbaijan', 'Spain', 'Monaco', 'Canada', 'Australia', 'Austria', 'Great Britain', 'Netherlands', 'Germany',
#          'Hungary', 'Belgium', 'Italy','Singapore', 'Japan', 'United States', 'Mexico', 'Brazil', 'Abu Dhabi', 'Saudi Arabia', 'France']
# sessions = ['Australia', 'Bahrain', 'China', 'Azerbaijan', 'Spain', 'Monaco', 'Canada', 'France', 'Austria', 'Great Britain', 'Germany',
        #  'Hungary', 'Belgium', 'Italy', 'Singapore', 'Russia', 'Japan', 'United States', 'Mexico', 'Brazil', 'Abu Dhabi']
session_type = ['FP1', 'FP2', 'FP3', 'SQ', 'Q', 'S', 'SS', 'R']


raw_data_collection_pipeline(session_year, sessions, session_type)
# raw_data_processing_pipeline(session_year, sessions, session_type)

# list_of_sessions = get_list_of_sessions('list_of_organized_files.csv')

# normalize_data(list_of_sessions)

# list_of_sessions = get_list_of_sessions('list_of_normalized_files.csv')

# create_features_per_session(list_of_sessions)

# list_of_sessions = get_list_of_sessions('list_of_feature_ready_files.csv')

# concatenate_master_log_csv_files(list_of_sessions)

# list_of_sessions = get_list_of_sessions('list_of_session_master_files.csv')

# concatenate_model_training_data(list_of_sessions)

# run_evaluation()

# pole_sitter_baseline()

# train_and_save_model(df)

# import fastf1 as f1
# import pandas as pd

# schedule = f1.get_event_schedule(2023)
# print(schedule)
# schedule_df = pd.DataFrame(schedule)
# print(schedule_df.head())
# print(schedule_df.info())