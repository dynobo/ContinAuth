"""Module used to resample sensor data session wise."""

# Standard
from pathlib import Path
import logging

# Extra
from tqdm.auto import tqdm
import pandas as pd
import numpy as np


def load_index(hdf5_file):
    """Retrieve index table with sessions and users stored hdf5 file.

    Arguments:
        hdf5_file {Path} -- Path to hdf5 File

    Returns:
        pd.DataFrame -- Index with hdf5 keys, sessions and subjects

    """
    logging.info("Opening HDF5 file...")
    with pd.HDFStore(hdf5_file, mode="r") as store:
        logging.info("Looking up index table...")
        df_index = store.select("index")
    return df_index


def resample_session(df_session, sampling):
    """Resample single user session.

    Uses mean() for sensor values, and min() for other fields (only relevant
    for "task_type").

    Arguments:
        df_session {pd.DataFrame} -- Session data with 100 hz frequency
        sampling {int}            -- [description]

    Returns:
        pd.DataFrame -- Resampled session data

    """
    df_resampled = df_session.groupby(by=lambda x: x // sampling, axis=0).agg(
        {
            "acc_x": np.mean,
            "acc_y": np.mean,
            "acc_z": np.mean,
            "gyr_x": np.mean,
            "gyr_y": np.mean,
            "gyr_z": np.mean,
            "mag_x": np.mean,
            "mag_y": np.mean,
            "mag_z": np.mean,
            "sys_time": np.min,
            "subject": np.min,
            "session": np.min,
            "task_type": np.min,
        }
    )
    return df_resampled


def resample_all_sessions(hdf5_file, sampling):
    """Read session tables from hdf5 file and save back resampled versions.

    The expected input tables are read from [session key]/sensors_100hz and
    stored as [session key]/sensors_[new freq]hz.

    Arguments:
        hdf5_file {Path} -- Path to hdf5 File
        sampling {int}   -- Sampling rate. E.g. if "4", four sensor values are
                            downsampled to 1 new value.
    """
    df_index = load_index(hdf5_file)
    freq = 100 // sampling
    logging.info(f"Resampling sessions to {freq} Hz...")
    with pd.HDFStore(hdf5_file) as store:
        for index, row in tqdm(df_index.iterrows(), total=len(df_index)):
            df_session = store.get(row["key"] + "/sensors_100hz")
            df_resampled = resample_session(df_session, sampling)
            store.put(f"{row.key}/sensors_{freq}hz", df_resampled, format="f")
    logging.info("Done.")


if __name__ == "__main__":
    """Example for resampling sensor data."""
    log_level = logging.DEBUG
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(funcName)s (L.%(lineno)s) | "
        + "%(message)s",
        datefmt="%H:%M:%S",
        level=log_level,
    )

    hdf5_file = Path.cwd() / "data" / "processed" / "hmog_dataset.hdf5"

    resample_all_sessions(hdf5_file, 4)
