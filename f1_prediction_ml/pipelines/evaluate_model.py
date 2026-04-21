import pandas as pd
from pathlib import Path
import sys
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from f1_prediction_ml.evaluation.model_evaluation import model_evaluation, inspect_feature_importance

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))  
os.makedirs( project_root / 'data' / 'model_evaluation_summaries', exist_ok=True)


def run_evaluation():
    """
    Run model evaluation for Random Forest and Gradient Boosting classifiers, and inspect feature importance.

    Args:
        None
    Returns:
        None (prints results to console and saves evaluation summaries to disk)
    """
    file_path = project_root / 'data' / 'processed' / 'model_training_data.csv'

    df = pd.read_csv(file_path)

    _, rfc = model_evaluation(df, RandomForestClassifier(n_estimators=100, random_state=42))
    _, gbc = model_evaluation(df, GradientBoostingClassifier(n_estimators=100, random_state=42))

    inspect_feature_importance(rfc)
    inspect_feature_importance(gbc)