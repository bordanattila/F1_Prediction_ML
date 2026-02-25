import pandas as pd
from ..utils.utils import CYAN, RESET

def aggregate_weather_data(weather_df: pd.DataFrame):
    """
    Aggregates weather data by calculating mean, min, max, and variability for key weather parameters.
    """
    # Aggregate weather data 
    print(f'{CYAN}INFO: Aggregating weather data{RESET}')
    print(f'{CYAN}************ Weather DataFrame Head ***********{RESET}')
    print(weather_df.head())

    weather_df = weather_df.drop(columns=['unnamed:_0'], errors='ignore')

    weather_aggregated = weather_df.copy().groupby('session_key', as_index=False).agg(
        air_temp_mean=('air_temp', 'mean'),
        air_temp_min=('air_temp', 'min'),
        air_temp_max=('air_temp', 'max'),
        track_temp_mean=('track_temp', 'mean'),
        track_temp_min=('track_temp', 'min'),
        track_temp_max=('track_temp', 'max'),
        humidity_mean=('humidity', 'mean'),
        humidity_min=('humidity', 'min'),
        humidity_max=('humidity', 'max'),
        wind_speed_mean=('wind_speed', 'mean'),
        wind_speed_min=('wind_speed', 'min'),
        wind_speed_max=('wind_speed', 'max'), 
        rain_any=('rainfall', 'max'),                 # True if any True exists
        rain_samples_ratio=('rainfall', 'mean'),      # Ratio of True samples
        rain_samples=('rainfall', 'size'),
    )

    # Calculate weather variability (standard deviation)
    weather_variability = (
        weather_df.copy().groupby('session_key', as_index=False)
        .agg(
            air_temp_std=('air_temp', 'std'),
            track_temp_std=('track_temp', 'std'),
            humidity_std=('humidity', 'std'),
            wind_speed_std=('wind_speed', 'std'),
            )
    )

    # Merge aggregated weather data with variability
    weather_aggregated = weather_aggregated.merge(weather_variability, on='session_key', how='left')

    return weather_aggregated