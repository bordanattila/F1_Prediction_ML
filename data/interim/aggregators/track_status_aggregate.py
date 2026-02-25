import pandas as pd
from ..utils.utils import CYAN, RESET

def aggregate_track_status_data(track_status_df: pd.DataFrame):
        """
        Aggregates track status data by calculating counts and durations of different track statuses.
        """
        # Convert Time column of track_status_df
        print(f'{CYAN}INFO: Aggregating track status data{RESET}')
        print(f'{CYAN}************ Track Status DataFrame Head ***********{RESET}')
        print(track_status_df.head())

        ts = track_status_df.copy()
        ts = ts.drop(columns=['unnamed:_0'], errors='ignore')
        ts['time'] = pd.to_timedelta(ts['time'], errors='raise')

        # Sort and compute duration each status lasted
        ts = ts.sort_values(by=['session_key', 'time'])
        ts['next_time'] = ts.groupby('session_key')['time'].shift(-1)
        ts['duration'] = (ts['next_time'] - ts['time'])
        # Replace the last row with 0 seconds.
        ts['duration'] = ts['duration'].fillna(pd.Timedelta(seconds=0))

        # Aggregate track status durations
        ts['yellow'] = ts['status'].astype(str).eq('2')
        ts['red'] = ts['status'].astype(str).eq('5')
        ts['vsc_deployed'] = ts['status'].astype(str).eq('6')
        ts['vsc_ending'] = ts['status'].astype(str).eq('7')
        ts['sc_deployed'] = ts['status'].astype(str).eq('4')
        ts['not_green'] = ts['status'].astype(str).ne('1')


        track_status_aggregated = (ts.groupby('session_key', as_index=False)
            .agg(
                yellow_count=('yellow', 'sum'),
                red_count=('red', 'sum'),
                vsc_deployed_count=('vsc_deployed', 'sum'),
                sc_deployed_count=('sc_deployed', 'sum'),
                vsc_duration=('duration', lambda x: x[ts.loc[x.index,'vsc_deployed']].sum()),
                vsc_ending_duration=('duration', lambda x: x[ts.loc[x.index,'vsc_ending']].sum()),
                sc_duration=('duration', lambda x: x[ts.loc[x.index,'sc_deployed']].sum()),
                not_green_duration=('duration', lambda x: x[ts.loc[x.index,'not_green']].sum()),
                total_duration=('duration', 'sum'),
            )
        )

        # Convert timedeltas into seconds
        track_status_aggregated['vsc_duration_in_seconds'] = track_status_aggregated['vsc_duration'].dt.total_seconds()
        track_status_aggregated['vsc_ending_duration_in_seconds'] = track_status_aggregated['vsc_ending_duration'].dt.total_seconds()
        track_status_aggregated['sc_duration_in_seconds'] = track_status_aggregated['sc_duration'].dt.total_seconds()
        track_status_aggregated['not_green_duration_in_seconds'] = track_status_aggregated['not_green_duration'].dt.total_seconds()
        track_status_aggregated['total_duration_in_seconds'] = track_status_aggregated['total_duration'].dt.total_seconds()

        track_status_aggregated['disruption_ratio'] = track_status_aggregated['not_green_duration_in_seconds'] / track_status_aggregated['total_duration_in_seconds'].replace(0, 1)  # Avoid division by zero

        return track_status_aggregated