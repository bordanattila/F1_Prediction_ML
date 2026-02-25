import pandas as pd
from ..utils.utils import CYAN, RESET

def aggregate_laps_data(laps_df: pd.DataFrame):
        """
        Aggregates laps data by calculating statistics for lap times and other performance metrics.
        """
        # Aggragate laps data
        print(f'{CYAN}INFO: Aggregating laps data{RESET}')
        print(f'{CYAN}************ Laps DataFrame ***********{RESET}')
        print(laps_df.head())

        # Drop columns that are not needed for aggregation
        laps_df = laps_df.drop(columns=['unnamed:_0','deleted_reason','lap_start_date','fast_f1_generated'], errors='ignore')

        # Convert time object columns to seconds
        laps_aggregated = laps_df.copy()
        TIME_COLS = ['lap_time', 'pit_out_time', 'pit_in_time', 'sector1_time', 'sector2_time', 'sector3_time',
            'sector1_session_time', 'sector2_session_time', 'sector3_session_time', 'lap_start_time',
        ]
        
        for col in TIME_COLS:
            laps_aggregated[col] = pd.to_timedelta(laps_aggregated[col], errors='coerce')
            laps_aggregated[col + '_seconds'] = laps_aggregated[col].dt.total_seconds()
       
        # Group by session_key and driver to get per-driver statistics
        laps_aggregated = (laps_aggregated.groupby(['session_key', 'driver_number'], as_index=False)
            .agg(
                lap_count=('lap_number', 'max'),
                lap_mean=('lap_time_seconds', 'mean'),
                lap_std=('lap_time_seconds', 'std'),
                lap_best=('lap_time_seconds', 'min'),
                lap_median=('lap_time_seconds', 'median'),
            )
        )

        laps_aggregated = laps_aggregated.drop(columns='driver_number_x', errors='ignore')

        return laps_aggregated