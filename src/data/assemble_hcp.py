"""Convert HCP data to HDF5 format"""

import argparse
import os

import h5py
import pandas as pd
import numpy as np

from pathlib import Path

timeseries_dir = "/home/mschoett/projects/rrg-pbellec/mschoett/HCP_fMRI_timeseries"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output",
    type=str,
    default="inputs/connectomes/hcp.h5",
    help="path to save the output",
)
args = parser.parse_args()
print(args)

path_concat = str(args.output)
for task in ["rest1", "rest2", "wm", "emotion", "gambling", "motor", "language", "relational", "social"]:
    print(f"Loading task {task}")
    subjects = os.listdir(Path(timeseries_dir, task, "derivatives", "preproc"))
    subjects.sort()
    for subject in subjects:
        for ses in ["ses-01", "ses-02"]:
            for scale in range(1,6):
                ts_sub_dir = Path(timeseries_dir, task, "derivatives", "preproc", subject, ses, f"{subject}_task-{task}_{ses}_desc-timeseries_scale-{scale}.csv")
                # skip missing data
                if os.path.exists(ts_sub_dir) == False:
                    continue
                timeseries = pd.read_csv((ts_sub_dir), index_col=0)
                with h5py.File(path_concat, "a") as h5file:
                    h5file.create_dataset(f"{task}/{subject}/{ses}/{subject}_task-{task}_{ses}_desc-timeseries_scale-{scale}", data=timeseries.values)