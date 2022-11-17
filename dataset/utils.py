from typing import Union, Optional, List, Dict, Tuple, Any
from datetime import datetime, timedelta
import os
import requests

import numpy as np
import pandas as pd

from tqdm.auto import tqdm


# Please set the environment variable or change the code.
BASE_URL = os.environ.get('API_URL', '')
PUSH_URL = BASE_URL + "/push-data"
LOGIN_URL = BASE_URL + "/login-data"
DISABLE_TQDM = os.environ.get('DISABLE_TQDM', False)


def ceil_dt(dt: datetime, delta: timedelta):
    """Round up datetime to the nearest multiple of delta."""
    return dt + (datetime.min - dt) % delta


def round_pd_timestamp(dt: pd.Timestamp, delta: timedelta) -> pd.Timestamp:
    """Round up pd.Timestamp to the nearest multiple of delta."""
    return dt + (datetime.min - dt.to_pydatetime()) % delta


def drop_duplicates_by_time(df: pd.DataFrame, delta: timedelta, keep: str = "first") -> pd.DataFrame:
    """Drop duplicates in a dataframe to keep only the first or last entry within a time window.

    Args:
        df (pd.DataFrame): dataframe to drop duplicates which has columns "timestamp".
        delta (timedelta): round up timestamp to the nearest multiple of delta and drop duplicates.
        keep (str): "first" or "last"

    Returns:
        pd.DataFrame: dataframe with duplicates dropped
    """
    _df = df.copy()

    _df["timestamp"] = _df["timestamp"].apply(lambda x: round_pd_timestamp(x, delta))
    mask = _df.duplicated(keep=keep)
    
    return df[~mask].copy()


def download_login(
    start_date: Union[str, datetime], 
    end_date: Union[str, datetime], 
    save_dir: str,
    overwrite: bool=False, 
    verbose: bool=False
) -> pd.DataFrame:
    """Download login data from thebackend server and preprocess it.
    Login data is stored in a csv file for each day.
    We download all the csv files between start_date and end_date and preprocess them.

    Args:
        start_date (str, datetime): start date (YYYY-MM-DD)
        end_date (str, datetime): end date (YYYY-MM-DD)
        save_dir (str): directory to save login data
        overwrite (bool): overwrite existing login data (default: False)
        verbose (bool): print progress (default: False)

    Returns:
        pd.DataFrame: login data
    """
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        if verbose:
            print(f"Created directory {save_dir}")

    if not isinstance(start_date, datetime):
        start_date = datetime.fromisoformat(start_date)
    if not isinstance(end_date, datetime):
        end_date = datetime.fromisoformat(end_date)
    current_date = start_date

    csv_files = []
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        current_login_url = LOGIN_URL + f"?date={date_str}"

        try:
            csv_file = os.path.join(save_dir, f"{date_str}.csv")
            if os.path.exists(csv_file) and not overwrite:
                csv_files.append(csv_file)
                if verbose:
                    print(f"File {csv_file} already exists. Skipping...")
            else:
                r = requests.get(current_login_url)
                current_login_df = pd.read_json(r.text)
                current_login_df.to_csv(csv_file, index=False)
                csv_files.append(csv_file)
                if verbose:
                    print(f"Saved {date_str}.csv")
        except:
            if verbose:
                print(f"Error in {date_str}")
        current_date += timedelta(1)

    # Reload all csv files and preprocess it
    login_df = preprocess_login_csv(csv_files)
    login_file = os.path.join(save_dir, f"login_start={start_date.strftime('%Y-%m-%d')}_end={end_date.strftime('%Y-%m-%d')}.parquet.gzip")
    login_df.to_parquet(login_file, compression='gzip', engine='pyarrow', index=False)

    return login_df


def download_push(save_dir: str, duplicate_mins: int=10, overwrite: bool=False, verbose: bool=False) -> pd.DataFrame:
    """
    Download push data from thebackend server and preprocess it.
    Push data is stored in a parquet file for the entire period.
    We download the raw log and preprocess it, which includes removing duplicates.

    Args:
        save_dir (str): directory to save push data
        duplicate_mins (int): round up timestamp to the nearest multiple of delta and drop duplicates. (default: 10)
        overwrite (bool): overwrite existing push data
        verbose (bool): print progress

    Returns:
        pd.DataFrame: push data
    """
    push_file = os.path.join(save_dir, "push.parquet.gzip")
    if os.path.exists(push_file) and not overwrite:
        if verbose:
            print(f"File {push_file} already exists. Skipping...")
        push_df = pd.read_parquet(push_file)

    else:
        r = requests.get(PUSH_URL)
        push_df = pd.read_json(r.text)
        push_df.rename({"pushTime": "timestamp"}, axis=1, inplace=True)
        push_df['timestamp'] = pd.to_datetime(push_df['timestamp'], utc=False)
        push_df['timestamp'] = push_df['timestamp'].dt.tz_localize(None)

        # To reduce memory footprint
        push_df['is_ad']   = pd.Categorical(push_df['is_ad'].astype(bool))
        push_df['game_id'] = pd.Categorical(push_df['game_id'])

        push_df.sort_values("timestamp", inplace=True)
        # Drop duplicates (if any) -- possible if the same push is sent multiple times (e.g. Android and iOS)
        push_df = drop_duplicates_by_time(push_df, delta=timedelta(minutes=duplicate_mins), keep='first')
        push_df.reset_index(drop=True, inplace=True)

        push_df.to_parquet(push_file, index=False, engine='pyarrow', compression='gzip')

    return push_df


def preprocess_login_csv(csv_files: List[str], duplicate_secs: int=10) -> pd.DataFrame:
    """
    Preprocess login csv files and make them into a dataframe
    
    Args:
        csv_files (List[str]): a list of csv files containing login logs
        duplicate_secs (int): consider two (or more) login logs as duplicated within the given seconds (default: 10)
    
    Returns:
        pd.DataFrame: concatenated login data
    """
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if df.columns[0] == "Unnamed: 0":
            # might contain an extra column of index
            df = df.drop("Unnamed: 0", axis=1)
            df.to_csv(csv_file, index=False)
            # resave the file for future use
        df = preprocess_login(df, duplicate_secs=duplicate_secs)
        dfs.append(df)
    df = pd.concat(dfs, axis=0)
    df['gamer_id'] = pd.Categorical(df['gamer_id'].astype(str))
    df['game_id'] = pd.Categorical(df['game_id'])
    df.reset_index(drop=True, inplace=True)
    
    return df


def load_login_from_parquet(parquet_file: str, duplicate_secs: int=10) -> pd.DataFrame:
    """Load login data from a parquet file and preprocess it.

    Args:
        parquet_file (str): parquet file to load

        duplicate_secs (int): round up timestamp to the nearest multiple of delta and drop duplicates. (default: 10)

    Returns:
        pd.DataFrame: login data
    """
    df = pd.read_parquet(parquet_file, engine='pyarrow')
    df = preprocess_login(df, duplicate_secs=duplicate_secs)

    return df


def preprocess_login(df: pd.DataFrame, duplicate_secs: int) -> pd.DataFrame:
    df.rename({"inDate": "timestamp", "indate": "timestamp"}, axis=1, inplace=True)
    
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
    # just in case (UTC and non-UTC are not comparable)
    df['timestamp'] = df['timestamp'].dt.tz_localize(None)
    df['gamer_id'] = df['gamer_id'].str.upper()

    df = drop_duplicates_by_time(df, delta=timedelta(seconds=duplicate_secs), keep='first')
    df.sort_values("timestamp", inplace=True)
    
    df = df.loc[:, ["timestamp", "game_id", "gamer_id"]]
    return df



def get_available_pushes(df: pd.DataFrame, before_hour: int, after_hour: int, sampling_day: int=7, return_ids: bool=False) -> pd.DataFrame:
    """
    Get pushes without another push between the given timeframe for T=1 
    and without any push between the same timeframe before `sampling_day`s ago for T=0.

                       sampling days before  <-----------------------  Push here!!!               
    |------------------T = 0-----|------------||------------------T = 1-----|------------|
    |             X              |    Y(0)    ||              X             |    Y(1)    |
                   | before_hour | after_hour |               | before_hour | after_hour |
                        No push in between                      Only one push in between

    Args:
        df: The pushes dataframe. Duplicateds need to be dropped beforehand.
        before_hour: The hour before which no push should be
        after_hour: The hour after which no push should be
        sampling_day: The day of the month to sample (default: 7)
        return_ids: return the ids of the pushes instead of the dataframe (default: False)

    Returns:
        The filtered dataframe
    """
    
    before_delta = timedelta(hours=before_hour)
    after_delta  = timedelta(hours=after_hour)
    sampling_delta = timedelta(days=sampling_day)
    
    available_ids = []

    for idx, row in df.iterrows():
        push_time = row['timestamp']
        game_id = row['game_id']
        before_time_t1 = push_time - before_delta
        after_time_t1  = push_time + after_delta
        
        subset_t1 = df[(df['timestamp'] > before_time_t1) & (df['timestamp'] < after_time_t1)]
        subset_t1 = subset_t1[subset_t1['game_id'] == game_id]

        nopush_time = push_time - sampling_delta
        before_time_t0 = nopush_time - before_delta
        after_time_t0  = nopush_time + after_delta

        subset_t0 = df[(df['timestamp'] > before_time_t0) & (df['timestamp'] < after_time_t0)]
        subset_t0 = subset_t0[subset_t0['game_id'] == game_id]
        
        if (len(subset_t1) == 1) and (len(subset_t0) == 0):
            available_ids.append(idx)

    if return_ids:
        return available_ids
    else:
        return df.loc[available_ids]


def create_dataset(
    timestamp: datetime, 
    game_id: int, 
    duration: int,
    after_hours: List[int],
    before_day: int,
    login_df: pd.DataFrame, 
    crud_df: pd.DataFrame, 
) -> List[Dict[str, pd.DataFrame]]:
    """
    Create dataset for a given timestamp and game_id.
    
    Args:
        timestamp (datetime): datetime object of the target push.
        
        game_id (int): game id.

        duration (int): duration of the time series in days.
        
        after_hours (List[int]): a list of hours after the push for targets (Y). 
        Multiple values will create multiple targets.

        before_day (int): number of days before the push for features (X|T=0).
        
        push_df (pd.DataFrame): push dataframe
        
        login_df (pd.DataFrame): login dataframe
        
        crud_df (pd.DataFrame): crud dataframe
    
    Returns:
        List[Dict[str, pd.DataFrame]]: a list of dictionaries containing X, Y, T.
    """

    max_after_hour = max(after_hours)

    # Get the time window
    push_t1  = timestamp
    start_t1 = push_t1 - timedelta(days=duration)
    end_t1   = push_t1 + timedelta(hours=max_after_hour)

    push_t0  = push_t1 - timedelta(days=before_day)
    start_t0 = push_t0 - timedelta(days=before_day)
    end_t0   = push_t0 + timedelta(hours=max_after_hour)
    
    # Deepcopy the subset of the dataframes
    crud_t1 = crud_df[(crud_df['game_id'] == game_id) & (crud_df['timestamp'] >= start_t1) & (crud_df['timestamp'] < push_t1)].copy().drop(columns=['game_id'])
    crud_t0 = crud_df[(crud_df['game_id'] == game_id) & (crud_df['timestamp'] >= start_t0) & (crud_df['timestamp'] < push_t0)].copy().drop(columns=['game_id'])

    login_t1 = login_df[(login_df['game_id'] == game_id) & (login_df['timestamp'] >= start_t1) & (login_df['timestamp'] < push_t1)].copy().drop(columns=['game_id'])
    login_t0 = login_df[(login_df['game_id'] == game_id) & (login_df['timestamp'] >= start_t0) & (login_df['timestamp'] < push_t0)].copy().drop(columns=['game_id'])

    y_t1 = login_df[(login_df['game_id'] == game_id) & (login_df['timestamp'] >= push_t1) & (login_df['timestamp'] < end_t1)].copy().drop(columns=['game_id'])
    y_t0 = login_df[(login_df['game_id'] == game_id) & (login_df['timestamp'] >= push_t0) & (login_df['timestamp'] < end_t0)].copy().drop(columns=['game_id'])

    # To concatenate the dataframes
    login_t1['method'] = 'LOGIN'
    login_t0['method'] = 'LOGIN'

    login_t1['action'] = 1  # 1 indicates login
    login_t0['action'] = 1  # 0 is left for <MASK>

    crud_t1['action'] += 2  # originally started from 0.
    crud_t0['action'] += 2  # Now 2, 3, 4, ...

    all_data = []

    # T=1
    gamer_ids = login_t1['gamer_id'].unique().tolist()
    print(f"Number of gamers for T=1 in game_id {game_id} on {timestamp.strftime('%Y-%m-%d')}: {len(gamer_ids)}")

    for gamer_id in tqdm(gamer_ids, disable=DISABLE_TQDM):
        # Create X features
        X_crud = crud_t1[crud_t1['gamer_id'] == gamer_id].drop(columns=['gamer_id']).copy()
        X_login = login_t1[login_t1['gamer_id'] == gamer_id].drop(columns=['gamer_id']).copy()
        X = pd.concat([X_crud, X_login], axis=0)
        X.sort_values(by=['timestamp'], inplace=True)
        X.reset_index(drop=True, inplace=True)

        # Create (possibly) multiple Y targets
        Y = np.zeros(len(after_hours))
        for idx, hour in enumerate(after_hours):
            end = push_t1 + timedelta(hours=hour)
            y = y_t1[(y_t1['gamer_id'] == gamer_id) & (y_t1['timestamp'] >= push_t1) & (y_t1['timestamp'] < end)]
            y = 1 if len(y) > 0 else 0
            Y[idx] = y

        T = 1

        all_data.append({'gamer_id': gamer_id, 'X': X, 'Y': Y, 'T': T})

    # T=0
    gamer_ids = login_t0['gamer_id'].unique().tolist()
    print(f"Number of gamers for T=0 in game_id {game_id} on {timestamp.strftime('%Y-%m-%d')}: {len(gamer_ids)}")

    for gamer_id in tqdm(gamer_ids, disable=DISABLE_TQDM):
        # Create X features
        X_crud = crud_t0[crud_t0['gamer_id'] == gamer_id].drop(columns=['gamer_id']).copy()
        X_login = login_t0[login_t0['gamer_id'] == gamer_id].drop(columns=['gamer_id']).copy()
        X = pd.concat([X_crud, X_login], axis=0)
        X.sort_values(by=['timestamp'], inplace=True)
        X.reset_index(drop=True, inplace=True)

        # Create (possibly) multiple Y targets
        Y = np.zeros(len(after_hours))
        for idx, hour in enumerate(after_hours):
            end = push_t0 + timedelta(hours=hour)
            y = y_t0[(y_t0['gamer_id'] == gamer_id) & (y_t0['timestamp'] >= push_t0) & (y_t0['timestamp'] < end)]
            y = 1 if len(y) > 0 else 0
            Y[idx] = y

        T = 0

        all_data.append({'gamer_id': gamer_id, 'X': X, 'Y': Y, 'T': T})

    return all_data