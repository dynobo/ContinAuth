"""Wrapper to load DataFrames of HMOG Dataset or similar datasets from HDF5.

CREATED IN 2019 BY
Buech, Holger

"""

# Standard
from pathlib import Path
import logging
import random

# Extra
from tqdm.auto import tqdm
import numpy as np
import pandas as pd


class DatasetLoader:
    """Wrapper around HMOG-Dataset or equally structured datasets."""

    def __init__(
        self,
        hdf5_file: Path,
        table_name: str,
        max_subjects: int,
        task_types: list,
        exclude_subjects: list = [],
        exclude_cols: list = [],
        seed: int = 712,
    ):
        """Init with arguments.

        Arguments:
            hdf5_file {Path}    -- Path of hdf5 file containing the data
            table_name {str}    -- Name of table (key) in hdf5 store
            max_subjects {int}  -- Maximum different subjects to load
            task_types {list}   -- Task types to include. If None, include all.

        Keyword Arguments:
            exclude_subjects {list} -- Subject ids to exclude (default: {[]})
            exclude_cols {list}     -- Columns to drop to save memory
                                       (default: {[]})
            seed {int}              -- Seed to use for all random operations
                                       (default: {712})

        Returns:
            [DatasetLoader] -- Self

        """
        self.hdf5_file = hdf5_file
        self.table_name = table_name
        self.max_subjects = max_subjects
        self.task_types = task_types
        self.exclude_subjects = exclude_subjects
        self.seed = seed
        self.excluded_features = []
        self.exclude_cols = exclude_cols

        # Will host DataFrames
        self.all = None
        self.index = None
        self.train = None
        self.valid = None
        self.test = None
        self.valid_train = None
        self.valid_test = None
        self.test_train = None
        self.test_test = None

        logging.info("Opening HDF5 file...")
        with pd.HDFStore(self.hdf5_file, mode="r") as store:
            logging.info("Looking up index table...")
            self.index = store.select("index")
            self.index = self.index[
                ~self.index["subject"].isin(self.exclude_subjects)
            ]
        logging.info("Done.")

        self._read_data()

    def data_summary(self):
        """Show DataFrames currently available in class and basic infos.

        Returns:
            pd.DataFrame -- DataFrame showing different dataspits including
                            session counts and subject counts

        """
        summary = []
        for name, df in (
            ("all", self.all),
            ("index", self.index),
            ("train", self.train),
            ("valid", self.valid),
            ("test", self.test),
            ("valid_train", self.valid_train),
            ("valid_test", self.valid_test),
            ("test_train", self.test_train),
            ("test_test", self.test_test),
        ):
            if type(df) is not pd.DataFrame:
                continue
            summary.append(
                {
                    "DataFrame": name,
                    "Rows": df.shape[0],
                    "Columns": df.shape[1],
                    "Memory (MB)": (
                        df.memory_usage().sum() / 1024 / 1024
                    ).round(2),
                    "Subjects": df["subject"].nunique(),
                    "Sessions": df["session"].nunique(),
                }
            )
        df = pd.DataFrame(summary)
        df = df[
            [
                "DataFrame",
                "Memory (MB)",
                "Rows",
                "Columns",
                "Subjects",
                "Sessions",
            ]
        ]
        return df

    def dfs_from_hdf5(self, keys: list):
        """Load data from list of hdf5 keys into DataFrames.

        Arguments:
            keys {list} -- Rable names to read from hdf5

        Returns:
            [list] -- List of sessions DataFrames

        """
        dfs = []
        with pd.HDFStore(self.hdf5_file, mode="r") as store:
            for idx, key in enumerate(tqdm(keys, desc="Loading sessions")):
                df = store.get(key)
                dfs.append(df)
        return dfs

    def _read_data(self) -> pd.DataFrame:
        """Retrieve data from hdf5."""
        df_get = self.index

        # Filter subjects
        subjects_to_get = df_get["subject"].unique()
        if len(self.exclude_subjects) > 0:
            subjects_to_get = np.array(
                [s for s in subjects_to_get if s not in self.exclude_subjects]
            )
        if self.max_subjects and self.max_subjects < len(subjects_to_get):
            np.random.seed(self.seed)
            subjects_to_get = np.random.choice(
                subjects_to_get, size=self.max_subjects, replace=False
            )
            df_get = df_get[df_get["subject"].isin(subjects_to_get)]

        # Filter task type
        if self.task_types:
            df_get = df_get[df_get["task_type"].isin(self.task_types)]

        # Filter sessions
        sessions_to_get = []
        for subject in subjects_to_get:
            sessions = df_get[df_get["subject"] == subject]["session"].unique()
            sessions_to_get.extend(list(sessions))
        df_get = df_get[df_get["session"].isin(sessions_to_get)]

        keys = [f"{v}/{self.table_name}" for v in df_get["key"].values]

        # Start reading data from hdf5
        df = pd.concat(self.dfs_from_hdf5(keys))

        # Drop columns, if passed as class argument
        df = df.drop(columns=self.exclude_cols)

        self.all = df

    def split_train_valid_test(
        self, n_test_subjects, n_valid_subjects, cleanup=True
    ):
        """Create three splits of the datasets by splitting by subjects.

        The main dataset is splitted into three datasets. The splitting is
        performed by splitting the subjects to ensure every single subject is
        only included in one of the three sets.

        This step does not include selecting an owner, which is done in
        a second step.
        """
        df = self.all
        subjects = df["subject"].unique().tolist()

        random.seed(self.seed)
        subjects_tests = random.sample(subjects, n_test_subjects)
        subjects = [s for s in subjects if s not in subjects_tests]

        random.seed(self.seed)
        subjects_valid = random.sample(subjects, n_valid_subjects)

        subjects_train = [  # NOQA
            s for s in subjects if s not in subjects_valid
        ]
        self.test = df.query("subject in @subjects_tests").copy()
        self.valid = df.query("subject in @subjects_valid").copy()
        self.train = df.query("subject in @subjects_train").copy()

        if cleanup:
            del self.all
            self.all = None

    def split_train_valid_train_test(
        self,
        n_valid_train,
        n_valid_test,
        n_test_train,
        n_test_test,
        cleanup=True,
    ):
        """Create three splits of the datasets by splitting by subjects.

        The main dataset is splitted into three datasets. The splitting is
        performed by splitting the subjects to ensure every single subject is
        only included in one of the three sets.

        This step does not include selecting an owner, which is done in
        a second step.
        """
        df = self.all
        subjects = df["subject"].unique().tolist()

        random.seed(self.seed)
        subjects_t_test = random.sample(subjects, n_test_test)
        subjects = [s for s in subjects if s not in subjects_t_test]

        random.seed(self.seed)
        subjects_t_train = random.sample(subjects, n_test_train)
        subjects = [s for s in subjects if s not in subjects_t_train]

        random.seed(self.seed)
        subjects_v_train = random.sample(subjects, n_valid_train)
        subjects_v_test = [  # NOQA
            s for s in subjects if s not in subjects_v_train
        ]

        self.valid_train = df.query("subject in @subjects_v_train").copy()
        self.valid_test = df.query("subject in @subjects_v_test").copy()
        self.test_train = df.query("subject in @subjects_t_train").copy()
        self.test_test = df.query("subject in @subjects_t_test").copy()

        if cleanup:
            del self.all
            self.all = None

    def split_train_test(self, n_test_subjects, cleanup=True):
        """Create two splits of the datasets by splitting by subjects.

        The main dataset is splitted into two datasets. The splitting is
        performed by splitting the subjects to ensure every single subject is
        only included in one of the two sets.

        This step does not include selecting an owner, which is done in
        a second step.

        """
        df = self.all
        subjects = df["subject"].unique().tolist()

        random.seed(self.seed)
        subjects_tests = random.sample(subjects, n_test_subjects)
        subjects_train = [  # NOQA
            s for s in subjects if s not in subjects_tests
        ]
        self.test = df.query("subject in @subjects_tests").copy()
        self.train = df.query("subject in @subjects_train").copy()

        if cleanup:
            del self.all
            self.all = None

    def split_sessions(self, df, sessions_per_type=1, seed=None):
        """Split input DataFrame by sessions.

        The splitting is stratified by task_type and subject, so both splits
        contain (appr.) sessions with the same tasks and same subjects.

        """
        if not seed:
            seed = self.seed

        sessions_all = set(df["session"].unique())
        sessions_test = set()

        df_task_types = (
            df.drop_duplicates(["task_type", "session"])
            .query("task_type != 0")
            .groupby(["subject", "task_type"])["session"]
            .unique()
        )

        for row in df_task_types:
            row = set(row)
            random.seed(seed)
            sessions_test.update(
                random.sample(row, min(len(row), sessions_per_type))
            )

        sessions_train = sessions_all - sessions_test  # NOQA
        df_train = df.query("session in @sessions_train").copy()
        df_test = df.query("session in @sessions_test").copy()

        return df_train, df_test


if __name__ == "__main__":
    """Just an example how to use the class."""
    log_level = logging.DEBUG
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(funcName)s (L.%(lineno)s) | "
        + "%(message)s",
        datefmt="%H:%M:%S",
        level=log_level,
    )

    path_hdf5 = Path.cwd() / "data" / "processed" / "hmog_dataset.hdf5"

    logging.info(f"\n{'='*50}\nInstantiate DatasetLoader...\n{'-'*50}")
    dl_hmog = DatasetLoader(
        hdf5_file=path_hdf5,
        table_name="sensors_100hz",
        max_subjects=30,
        task_types=None,
        split_ratio=0.3,
        exclude_subjects=[],
        exclude_cols=[],
        seed=712,
    )
    logging.info(f"\n{'='*50}\nCurrently available datasets:\n{'-'*50}")
    print(dl_hmog.data_summary())

    dl_hmog.split_train_valid_test()
    print(dl_hmog.data_summary())

    logging.info(f"\n{'='*50}\nCurrently available datasets:\n{'-'*50}")
    print(dl_hmog.data_summary())

    logging.info(
        f"\n{'='*50}\nHead of datasets for parameter tuning "
        + f"(train / valid):\n{'-'*50}"
    )
    print(dl_hmog.train.head())
    print(dl_hmog.valid.head())

    logging.info(f"\n{'='*50}\nSubjects picked as owners:\n{'-'*50}")
    print(f"For validation: {dl_hmog.valid_owner}")
    print(f"For testing: {dl_hmog.test_owner}")
