"""Wrapper to perform various steps to download and transfrom dataset.

CREATED IN 2019 BY:
Buech, Holger

"""

# Standard
import logging
from pathlib import Path

# Own
from .unzip_hmog_dataset import unzip_hmog
from .transform_to_hdf5 import dataset_to_hdf5
from .resample_dataset import resample_all_sessions

if __name__ == "__main__":
    log_level = logging.INFO
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(funcName)s (L.%(lineno)s) |"
        + " %(message)s",
        datefmt="%H:%M:%S",
        level=log_level,
    )

    path_source_hmog_zip = Path.cwd() / "hmog_dataset.zip"
    path_target_hmog_dir = Path.cwd() / "data" / "external" / "hmog_dataset"
    path_target_hdf5 = Path.cwd() / "data" / "processed" / "hmog_dataset.hdf5"

    logging.info("Start extracting HMOG Dataset...")
    unzip_hmog(path_source_hmog_zip, path_target_hmog_dir)
    logging.info("Done.")

    logging.info(
        f"Start converting {path_target_hmog_dir.resolve()} to HDF5..."
    )
    dataset_to_hdf5(path_target_hmog_dir, path_target_hdf5)
    logging.info("Done.")

    logging.info(f"Start resampling dataset {path_target_hdf5} to 25hz...")
    resample_all_sessions(path_target_hdf5, 4)
    logging.info("Done.")
