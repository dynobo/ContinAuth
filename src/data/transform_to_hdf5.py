"""Stores CSV files from dataset into a single hdf5 blob.

HDF5 is more space efficient, faster to read and write [1], and we can
also use Queries for subsets of the data [2].

The sensors CSVs get joined into a single table by interpolating some missing
values, joining on the frequency (10ms) and dropping NaN at beginning / end.

[1] http://matthewrocklin.com/blog/work/2015/03/16/Fast-Serialization
[2] https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html

RUNTIME:
- For HMOG ~60min (on SSD)
- For ISROS TODO: Add time need for h5 transform

CREATED IN 2019 BY:
Buech, Holger

"""

# Standard
from pathlib import Path
import logging
import os

# Extra
from tqdm import tqdm
import pandas as pd

# Some constants
ACTIVITY_FILE = "Activity.csv"
ACTIVITY_FILE_COLUMNS = [
    "activity_id",
    "subject",
    "session_number",
    "start_time",
    "end_time",
    "relative_start_time",
    "relative_end_time",
    "gesture_scenario",
    "task_id",
    "content_id",
]

DATA_FILES = ["Accelerometer.csv", "Gyroscope.csv", "Magnetometer.csv"]
DATA_FILE_COLUMNS = [
    "sys_time",
    "event_time",
    "activity_id",
    "x",
    "y",
    "z",
    "orientation",
]

# Those sessions are lacking Accelerometer Data and break the workflow
EXCLUDE_SESSIONS = [
    "733162_session_9",
    "733162_session_10",
    "733162_session_11",
    "733162_session_12",
    "733162_session_13",
    "733162_session_14",
]

# Mapping of task_ids to task_types, according to data_description.pdf
HMOG_TASK_IDS_TYPES = {
    7: 1,  # Reading + Sitting = 1
    13: 1,
    19: 1,
    8: 2,  # Reading + Walking = 2
    14: 2,
    20: 2,
    9: 3,  # Writing + Sitting = 3
    15: 3,
    21: 3,
    10: 4,  # Writing + Walking = 4
    16: 4,
    22: 4,
    11: 5,  # Map + Sitting = 5
    17: 5,
    23: 5,
    12: 6,  # Map + Walking = 6
    18: 6,
    24: 6,
}


def dataset_to_hdf5(raw_path: Path, target_path: Path):
    """Transform a HMOG-like Dataset into a nested HDF5 file. Main function.

    Arguments:
        raw_path {Path}    -- Root datasetfolder, contains subject folders
        target_path {Path} -- Filename of target HDF5 file

    """
    _prepare_target_dir(target_path)

    logging.info("Creating hdf5 blob file...")
    with pd.HDFStore(target_path) as store:
        logging.info("Converting session by session...")
        session_paths = [p for p in raw_path.glob("*/*/")]

        # loop all session folders
        sessions_index = []
        for p in tqdm(session_paths):
            # Skip files (.DS_Store) and excluded session
            if (not os.path.isdir(p)) or (p.name in EXCLUDE_SESSIONS):
                logging.debug(f"Skipping {p.resolve()}")
                continue

            # Derive subject and session from path
            subject = p.parent.name
            session = p.name
            session_no = session.split("_")[-1]  #

            # Read
            df_act = _read_activity(p)
            df_sens = _read_sensors(p)

            # Join task/scenario information to sensor data
            df_sens = _join_activity(df_act, df_sens)

            # Save to hdf5. Renaming, because keys can't start with digits
            store.put(
                f"subject_{subject}/session_{subject}_{session_no}/activity",
                df_act,
                format="f",
            )
            store.put(
                f"subject_{subject}/session_{subject}_{session_no}/sensors_100hz",
                df_sens,
                format="f",
            )

            # Compose index table
            sessions_index.append(
                {
                    "subject": subject,
                    "session": f"{subject}_session_{session_no}",
                    "key": f"subject_{subject}/session_{subject}_{session_no}",
                    "task_type": df_sens["task_type"].max(),
                }
            )

        # Save index table to hdf5
        df_index = pd.DataFrame(sessions_index)
        store.put(f"index", df_index, format="f")


def _prepare_target_dir(file_path: Path):
    """Check if file exists and if yes, offers to delete it.

    Also ensures the directory of the file is existing.

    Arguments:
        file_path {Path} -- File to check for existence

    """
    if os.path.isfile(file_path):
        delete = input(f"File '{file_path.resolve()}' exists. Delete? (Y/n) ")
        if delete == "Y":
            os.remove(file_path)
            logging.info("File removed.")
        else:
            logging.info("Won't overwrite existing file. Quit.")
            quit()
    else:
        file_path.parents[0].mkdir(parents=True, exist_ok=True)


def _read_activity(session_path: Path):
    """Read CSVs of intertial sensor and activity description to dataframes.

    The three sensors CSVs are joined by row into a single table!

    Arguments:
        session_path {Path} -- Path of a Session, containing the CSVs

    Returns:
        {pd.DataFrame} -- DF containing activity.csv

    """
    # Read activity file
    df_act = pd.read_csv(
        session_path / ACTIVITY_FILE,
        names=ACTIVITY_FILE_COLUMNS,
        usecols=[
            "subject",
            "session_number",
            "start_time",
            "end_time",
            "gesture_scenario",
            "task_id",
        ],
        header=None,
        engine="c",
    )
    # Timestamps as additional datetime columns
    df_act["start_time_dt"] = pd.to_datetime(df_act["start_time"], unit="ms")
    df_act["end_time_dt"] = pd.to_datetime(df_act["end_time"], unit="ms")

    return df_act


def _join_on_millisec(dfs: list):
    """Join dataframes by resampling on milliseconds and join on datetimeindex.

    Arguments:
        dfs {list} - List of sensor dataframes to join.

    Returns:
        {pd.Dataframe} - Joined sensor dataframes.

    """
    # Resample to milliseconds befor joining
    for idx, df in enumerate(dfs):
        df["sys_time_dt"] = pd.to_datetime(df["sys_time"], unit="ms")
        df = df.set_index("sys_time_dt")
        df = df.drop(columns=["sys_time"])
        df = df[~df.index.duplicated(keep="last")]  # Remove index dups
        dfs[idx] = df.resample("10ms").interpolate(method="time")

    # Join resampled sensor data, drop NaNs, that might be generated for
    # start or end of session, because not all sensors start/end at same time
    df_joined = pd.concat(dfs, axis=1).dropna()

    # Add datetimeindex as ms
    df_joined["sys_time"] = df_joined.index.astype("int64") // 10 ** 6

    # Reset index to save memory
    df_joined = df_joined.reset_index(drop=True)

    return df_joined


def _read_sensors(session_path: Path):
    """Read CSVs of intertial sensor to dataframes.

    The three sensors CSVs red from disk and  joined by timestamp into a
    single table.

    Arguments:
        session_path {Path} -- Path of a Session, containing the CSVs.

    Returns:
        {pd.DataFrame} -- DF containing joined sensor readings.

    """
    subject = session_path.parent.name
    session = session_path.name

    # Read all sensor files
    sensor_dfs = []
    for filename in DATA_FILES:
        df = pd.read_csv(
            session_path / filename,
            names=DATA_FILE_COLUMNS,
            usecols=["x", "y", "z", "sys_time"],
            dtype="float64",
            header=None,
            engine="c",
        )
        col_prefix = filename[:3].lower() + "_"
        df.columns = [
            col_prefix + c if c != "sys_time" else c for c in df.columns
        ]
        sensor_dfs.append(df)

    # Join into single DF
    df_sensors = _join_on_millisec(sensor_dfs)
    df_sensors["subject"] = subject
    df_sensors["session"] = session

    return df_sensors


def _join_activity(df_activity: pd.DataFrame, df_sens: pd.DataFrame):
    """Add meta info to sensors DF based on timestamps in activity DF.

    Also maps the 24 different task_ids into 6 different task types.

    Arguments:
        df_activity {pd.DataFrame} -- Activity information
        df_sens {pd.DataFrame}     -- Combined sensor data

    Returns:
        {pd.DataFrame} -- Sensor DF with additional "task" cols

    """
    df_sens["task_id"] = 0
    for idx, row in df_activity.iterrows():
        df_sens.loc[
            (df_sens["sys_time"] >= row["start_time"])
            & (df_sens["sys_time"] <= row["end_time"]),
            "task_id",
        ] = row["task_id"]

    # Map 24 task ids down to 6 task types
    df_sens["task_type"] = (
        df_sens["task_id"].astype("int8").replace(HMOG_TASK_IDS_TYPES)
    )
    df_sens = df_sens.drop(columns=["task_id"])

    return df_sens


if __name__ == "__main__":
    log_level = logging.INFO
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(funcName)s (L.%(lineno)s) |"
        + " %(message)s",
        datefmt="%H:%M:%S",
        level=log_level,
    )

    # ---------------------
    # Transform into HDF5
    # ---------------------
    path_dataset = Path.cwd() / "data" / "external" / "hmog_dataset"
    path_target_hdf5 = Path.cwd() / "data" / "processed" / "hmog_dataset.hdf5"

    logging.info(f"Start converting Dataset from {path_dataset.resolve()}...")

    dataset_to_hdf5(path_dataset, path_target_hdf5)

    logging.info("Done.")
