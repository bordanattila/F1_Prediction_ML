from pathlib import Path
import pandas as pd
import joblib

from f1_prediction_ml.modeling.build_next_race_features import (
    build_next_race_features,
    align_features_for_inference,
)

MODEL_PATH = Path('models/random_forest_winner.pkl')


class WinnerPredictor:
    'Loads a trained model artifact and predicts race winners from pre-built feature DataFrames.'

    def __init__(self, model_artifact_path: str = str(MODEL_PATH)):
        """
        Args:
            model_artifact_path: Path to a joblib artifact dict containing
                'model' (fitted pipeline) and 'feature_cols' (list of training column names).
        """
        artifact = joblib.load(model_artifact_path)
        self.model = artifact['model']
        self.feature_cols = artifact['feature_cols']

    def predict_next_race_winner(self, next_race_features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict win probability for every driver in the upcoming race.

        Args:
            next_race_features_df: One row per driver with the same engineered features used during training.
        Returns:
            pd.DataFrame with columns [event_id, driver_id, driver_name, win_proba], sorted by probability.
        """
        df = next_race_features_df.copy()

        id_cols = ['event_id', 'driver_id', 'abbreviation']
        missing = [col for col in id_cols if col not in df.columns]
        if missing:
            raise ValueError(f'Missing id columns: {missing}')

        X = align_features_for_inference(df, self.feature_cols)

        win_proba = self.model.predict_proba(X)[:, 1]

        results = df[id_cols].copy()
        results['win_proba'] = win_proba

        results = results.sort_values(
            by=['event_id', 'win_proba'],
            ascending=[True, False]
        ).reset_index(drop=True)

        return results

    def get_predicted_winner(self, prediction_df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns the top predicted driver per event.

        Args:
            prediction_df: DataFrame with columns [event_id, driver_id, driver_name, win_proba].

        Returns:
            pd.DataFrame with the predicted winner for each event.
        """
        winner_df = (
            prediction_df.sort_values(['event_id', 'win_proba'], ascending=[True, False])
            .groupby('event_id', as_index=False)
            .head(1)
            .reset_index(drop=True)
        )
        return winner_df


if __name__ == '__main__':
    next_race_df = pd.read_csv('data/processed/next_race_features.csv')
    next_race_features_df = build_next_race_features(next_race_df)

    predictor = WinnerPredictor()

    prediction_df = predictor.predict_next_race_winner(next_race_features_df)
    winner_df = predictor.get_predicted_winner(prediction_df)

    print('\nPredicted probabilities for upcoming race:')
    print(prediction_df[['abbreviation', 'win_proba']])

    print('\nPredicted winner:')
    print(winner_df[['event_id', 'abbreviation', 'win_proba']])