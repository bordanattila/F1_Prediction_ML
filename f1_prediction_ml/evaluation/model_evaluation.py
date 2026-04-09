import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    log_loss
)

from f1_prediction_ml.colors import CYAN, MAGENTA, RESET

def model_evaluation(
    df: pd.DataFrame,
    model,
    target: str = 'race_is_winner',
    group_col: str = 'event_id',
    n_splits: int = 5,
):
    """
    Evaluate a classification model for F1 winner prediction.

    This function:
    - uses GroupKFold so whole races stay together
    - predicts probability of winning for each driver
    - measures whether the top predicted driver for each race was the real winner

    Parameters
    ----------
    df : pd.DataFrame
        Final race-level modeling dataframe.
    model : sklearn-compatible classifier
        Example: RandomForestClassifier(), XGBClassifier(), LogisticRegression()
    target : str
        Target column name. Default: 'race_is_winner'
    group_col : str
        Column used to keep full races together in CV. Default: 'event_id'
    n_splits : int
        Number of CV folds.

    Returns
    -------
    results : dict
        Dictionary containing fold metrics and summary statistics.
    """

    data = df.copy()

    # 1. Drop rows with missing target
    data = data.dropna(subset=[target])

    # 2. Define columns to EXCLUDE
    exclude_cols = {
        target,
        group_col,
        'row_id',
    }

    race_leak_cols = {
        'race_lap_count',
        'race_lap_mean',
        'race_lap_std',
        'race_lap_best',
        'race_lap_median',
        'race_air_temp_mean',
        'race_air_temp_std',
        'race_track_temp_mean',
        'race_track_temp_std',
        'race_humidity_mean',
        'race_humidity_std',
        'race_wind_speed_mean',
        'race_wind_speed_std',
        'race_rain_any',
        'race_rain_samples_ratio',
        'race_yellow_count',
        'race_red_count',
        'race_vsc_deployed_count',
        'race_sc_deployed_count',
        'race_disruption_ratio',
        'race_race_finish_position',
        'race_is_dnf',
        'race_laps_down',
        'race_started_from_pit_lane',
        'race_is_winner',
        'race_finish_position_relative_to_grid_start_position',
    }

    exclude_cols = exclude_cols.union(set(c for c in data.columns if c in race_leak_cols))

    X = data.drop(columns=[c for c in exclude_cols if c in data.columns])
    y = data[target].astype(int)
    groups = data[group_col]

    # 3. Identify feature types
    numeric_features = X.select_dtypes(include=['number', 'bool']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # bools are fine as numeric, but cast if needed
    for col in numeric_features:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)

    # 4. Preprocessing
    numeric_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='median'))
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
        ]
    )

    classifier = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('model', model),
        ]
    )

    # 5. Cross-validation
    gkf = GroupKFold(n_splits=n_splits)

    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), start=1):
        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_train = y.iloc[train_idx].copy()
        y_test = y.iloc[test_idx].copy()

        test_event_ids = groups.iloc[test_idx].copy()

        classifier.fit(X_train, y_train)

        # Probability for positive class = winner
        y_proba = classifier.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)


        # 6. Standard classification metrics

        fold_accuracy = accuracy_score(y_test, y_pred)
        fold_f1 = f1_score(y_test, y_pred, zero_division=0)

        # ROC AUC can fail if a fold has only one class in y_test
        try:
            fold_roc_auc = roc_auc_score(y_test, y_proba)
        except ValueError:
            fold_roc_auc = np.nan

        try:
            fold_log_loss = log_loss(y_test, y_proba, labels=[0, 1])
        except ValueError:
            fold_log_loss = np.nan


        # 7. Race-level winner-pick metric

        # For each race, select the driver with the highest predicted probability
        test_eval = pd.DataFrame({
            'event_id': test_event_ids.values,
            'y_true': y_test.values,
            'y_proba': y_proba
        })

        predicted_winners = (
            test_eval.sort_values(['event_id', 'y_proba'], ascending=[True, False])
                     .groupby('event_id', as_index=False)
                     .first()
        )

        # If y_true == 1 on the selected row, the model picked the race winner correctly
        winner_pick_accuracy = predicted_winners['y_true'].mean()

        fold_results.append({
            'fold': fold_idx,
            'accuracy': fold_accuracy,
            'f1': fold_f1,
            'roc_auc': fold_roc_auc,
            'log_loss': fold_log_loss,
            'winner_pick_accuracy': winner_pick_accuracy,
            'n_test_rows': len(X_test),
            'n_test_events': predicted_winners['event_id'].nunique(),
        })

    # 8. Summary
    results_df = pd.DataFrame(fold_results)

    summary = {
        'model_name': model.__class__.__name__,
        'fold_results': results_df,
        'mean_accuracy': results_df['accuracy'].mean(),
        'std_accuracy': results_df['accuracy'].std(),
        'mean_f1': results_df['f1'].mean(),
        'std_f1': results_df['f1'].std(),
        'mean_roc_auc': results_df['roc_auc'].mean(),
        'std_roc_auc': results_df['roc_auc'].std(),
        'mean_log_loss': results_df['log_loss'].mean(),
        'std_log_loss': results_df['log_loss'].std(),
        'mean_winner_pick_accuracy': results_df['winner_pick_accuracy'].mean(),
        'std_winner_pick_accuracy': results_df['winner_pick_accuracy'].std(),
    }

    # 9. Pretty print
    print(f'\nModel: {summary['model_name']}')
    print(f'CV Accuracy: {summary['mean_accuracy']:.4f} ± {summary['std_accuracy']:.4f}')
    print(f'CV F1: {summary['mean_f1']:.4f} ± {summary['std_f1']:.4f}')
    print(f'CV ROC AUC: {summary['mean_roc_auc']:.4f} ± {summary['std_roc_auc']:.4f}')
    print(f'CV Log Loss: {summary['mean_log_loss']:.4f} ± {summary['std_log_loss']:.4f}')
    print(
        f'CV Winner Pick Accuracy: '
        f'{summary['mean_winner_pick_accuracy']:.4f} ± {summary['std_winner_pick_accuracy']:.4f}'
    )
    
    return summary, classifier


def inspect_feature_importance(pipeline):
    """Inspect feature importance for tree-based models."""

    model = pipeline.named_steps["model"]
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()

    importances = pd.Series(
        model.feature_importances_,
        index=feature_names
    ).sort_values(ascending=False)
    print(f'{MAGENTA} {model.__class__.__name__} {RESET} {CYAN} INFO: Top 25 features by importance:{RESET}')
    print(importances.head(25))