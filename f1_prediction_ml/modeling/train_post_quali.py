import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, get_scorer_names

from sklearn.inspection import permutation_importance

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

df = pd.read_csv(project_root / 'data' / 'processed' / 'model_training_data.csv')

target = 'race_is_winner'

numeric_features = ['year','fp1_lap_count', 'fp1_lap_mean', 'fp1_lap_std', 'fp1_lap_best', 'fp1_lap_median', 'fp1_air_temp_mean', 'fp1_air_temp_std', 
                    'fp1_track_temp_mean', 'fp1_track_temp_std', 'fp1_humidity_mean', 'fp1_humidity_std', 'fp1_wind_speed_mean', 'fp1_wind_speed_std',
                    'fp1_rain_any', 'fp1_rain_samples_ratio', 'fp1_yellow_count', 'fp1_red_count', 'fp1_vsc_deployed_count', 'fp1_sc_deployed_count',
                    'fp1_disruption_ratio', 'fp1_best_free_practice_sec', 'fp1_free_practice_delta_to_best_lap_sec', 'fp1_free_practice_percent_of_best_lap_sec', 
                    'fp2_lap_count', 'fp2_lap_mean', 'fp2_lap_std', 'fp2_lap_best', 'fp2_lap_median', 'fp2_air_temp_mean', 'fp2_air_temp_std', 'fp2_track_temp_mean', 
                    'fp2_track_temp_std', 'fp2_humidity_mean', 'fp2_humidity_std', 'fp2_wind_speed_mean', 'fp2_wind_speed_std', 'fp2_rain_any', 'fp2_rain_samples_ratio', 
                    'fp2_yellow_count', 'fp2_red_count', 'fp2_vsc_deployed_count', 'fp2_sc_deployed_count', 'fp2_disruption_ratio', 'fp2_best_free_practice_sec', 
                    'fp2_free_practice_delta_to_best_lap_sec', 'fp2_free_practice_percent_of_best_lap_sec', 'fp3_lap_count', 'fp3_lap_mean', 'fp3_lap_std',
                    'fp3_lap_best', 'fp3_lap_median', 'fp3_air_temp_mean', 'fp3_air_temp_std', 'fp3_track_temp_mean', 'fp3_track_temp_std', 'fp3_humidity_mean', 
                    'fp3_humidity_std', 'fp3_wind_speed_mean', 'fp3_wind_speed_std', 'fp3_rain_any', 'fp3_rain_samples_ratio', 'fp3_yellow_count', 
                    'fp3_red_count', 'fp3_vsc_deployed_count', 'fp3_sc_deployed_count', 'fp3_disruption_ratio', 'fp3_best_free_practice_sec', 
                    'fp3_free_practice_delta_to_best_lap_sec', 'fp3_free_practice_percent_of_best_lap_sec', 'quali_lap_count', 'quali_lap_mean', 
                    'quali_lap_std', 'quali_lap_best', 'quali_lap_median', 'quali_air_temp_mean', 'quali_air_temp_std', 'quali_track_temp_mean', 
                    'quali_track_temp_std', 'quali_humidity_mean', 'quali_humidity_std', 'quali_wind_speed_mean', 'quali_wind_speed_std', 'quali_rain_any',
                    'quali_rain_samples_ratio', 'quali_yellow_count', 'quali_red_count', 'quali_vsc_deployed_count', 'quali_sc_deployed_count', 'quali_disruption_ratio', 
                    'quali_reached_q1', 'quali_reached_q2', 'quali_reached_q3', 'quali_best_quali_seconds', 'quali_quali_delta_to_pole', 'quali_quali_percent_of_pole', 'quali_quali_finish_position']

categorical_features = ['grand_prix', 'event_id', 'row_id', 'driver_id', 'abbreviation', 'team_name', 'team_id']

features = numeric_features + categorical_features 

X = df[features]
y = df[target]

# preprocessing

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scalar', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocess = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features),
    ('spe', 'passthrough', special_feature)
], remainder='drop')

# train/test split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8, test_size=.2, random_state=1, )