import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(project_root))   


from f1_prediction_ml.pipelines.raw_data_collector import raw_data_collection_pipeline
from f1_prediction_ml.pipelines.raw_data_processor import raw_data_processing_pipeline

session_year = 2018
sessions = ['Australia', 'Bahrain', 'China', 'Azerbaijan', 'Spain', 'Monaco', 'Canada', 'France', 'Austria', 'Great Britain', 'Germany',
         'Hungary', 'Belgium', 'Italy', 'Singapore', 'Russia', 'Japan', 'United States', 'Mexico', 'Brazil', 'Abu Dhabi']
session_type = ['FP1', 'FP2', 'FP3', 'SQ', 'Q', 'S', 'SS', 'R']

# raw_data_collection_pipeline(session_year, sessions, session_type)
raw_data_processing_pipeline(session_year, sessions, session_type)