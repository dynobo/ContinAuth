"""Unzip CSV files from HMOG-Dataset.

Unzips HMOG-Dataset-Zip to local directory.

NOTES:
- The _extracted_ CSV-files needs additionally ~30GB.

CREATED IN 2019 BY:
Buech, Holger

"""

# Standard
from multiprocessing import Pool
from pathlib import Path
import logging
import shutil
import zipfile


def extract_and_remove(zip_file: Path):
    """Extract zip file to parent directory and remove it.

    Arguments:
        zip_file {Path} -- Path to zip file

    """
    # Unzip file
    zip_ref = zipfile.ZipFile(zip_file, "r")
    zip_ref.extractall(zip_file.parent)
    zip_ref.close()
    # Remove sessions zip file
    zip_file.unlink()


def unzip_hmog(source_path: Path, target_path: Path):
    """Extract hmog_dataset.zip and its containing session zip files.

    Arguments:
        source_path {Path} -- Path to hmog_dataset.zip
        target_path {Path} -- Path to target folder for extracted files

    """
    # Create target path, if not exists
    if target_path.exists() and (len(list(target_path.glob("*"))) > 0):
        raise FileExistsError(
            "Target folder exists and is not empty: "
            + str(target_path.resolve())
        )
    target_path.mkdir(parents=True, exist_ok=True)

    # Unzip hmog_dataset.zip
    logging.info(
        f"Extracting {source_path.resolve()} "
        + f"to {target_path.resolve()}..."
    )
    zip_ref = zipfile.ZipFile(source_path, "r")
    zip_ref.extractall(target_path)
    zip_ref.close()

    # Move all files one level up
    files = target_path.glob("**/*")
    for f in files:
        f.rename(target_path / f.name)
    target_path.joinpath("public_dataset").rmdir()

    # Unzip session files
    logging.info(f"Extracting and removing session zip files...")
    zip_files = target_path.glob("**/*.zip")
    Pool().map(extract_and_remove, zip_files)

    # Rename folder (subject-folder name has mixed up digits, change to actual
    # subject id)
    wrong_dir = target_path / "207969"
    right_dir = target_path / "207696"
    wrong_dir.rename(right_dir)

    # Remove MacOS-specific folder, as not needed
    shutil.rmtree(target_path.joinpath("__MACOSX"), ignore_errors=True)


if __name__ == "__main__":
    log_level = logging.DEBUG
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(funcName)s (L.%(lineno)s) |"
        + " %(message)s",
        datefmt="%H:%M:%S",
        level=log_level,
    )

    # ---------------------
    # Download & Extract HMOG Dataset
    # ---------------------
    path_hmog_zip = Path.cwd() / "hmog_dataset.zip"
    path_target_data_folder = Path.cwd() / "data" / "external" / "hmog_dataset"

    logging.info("Start extracting HMOG Dataset...")
    unzip_hmog(path_hmog_zip, path_target_data_folder)

    logging.info("Done.")
