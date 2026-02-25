"""
This is a script that loads cvs files normalizes them. Each session type is normalized separately
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from f1_prediction_ml.normalize.utils import get_list_of_sessions
from f1_prediction_ml.normalize.normalize_quali import Quali_Normalizer
# from f1_prediction_ml.normalize.normalize_race import Race_Normalizer
# from f1_prediction_ml.normalize.concatenate_csvs import concatenate_csv_files   

list_of_sessions = get_list_of_sessions()