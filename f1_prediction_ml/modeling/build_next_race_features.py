import pandas as pd

def build_next_race_features(next_race_df: pd.DataFrame) -> pd.DataFrame:
    """
    next_race_df must already contain one row per driver for the upcoming event,
    with the same engineered features used during training.
    """
    required_id_cols = ['event_id', 'driver_id', 'abbreviation']

    missing = [col for col in required_id_cols if col not in next_race_df.columns]
    if missing:
        raise ValueError(f'Missing required columns in next race dataframe: {missing}')

    if next_race_df.duplicated(['event_id', 'driver_id']).any():
        raise ValueError('next_race_df must contain only one row per event_id + driver_id')

    return next_race_df.copy()


def align_features_for_inference(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Ensures inference dataframe has exactly the training feature columns
    in the same order.
    """
    X = df.copy()

    # Add any missing training columns as 0
    missing_cols = [col for col in feature_cols if col not in X.columns]
    for col in missing_cols:
        X[col] = 0

    # Drop any extra columns not used by the model
    extra_cols = [col for col in X.columns if col not in feature_cols]
    if extra_cols:
        X = X.drop(columns=extra_cols)

    # Reorder to match training exactly
    X = X[feature_cols]

    return X




