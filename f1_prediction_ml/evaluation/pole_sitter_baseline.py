import pandas as pd
from pathlib import Path
import sys
import os

from f1_prediction_ml.colors import CYAN, GREEN, RESET

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))  
os.makedirs( project_root / 'data' / 'model_evaluation_summaries', exist_ok=True)


def pole_sitter_baseline():
    """
    Compute a naive baseline: predict the pole-sitter wins every race.

    Args:
        None (uses the model training data CSV as input)
    Returns:
        float: Fraction of races where the pole-sitter actually won.
    """
    file_path = project_root / 'data' / 'processed' / 'model_training_data.csv'

    df = pd.read_csv(file_path)

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

