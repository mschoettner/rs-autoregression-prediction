"""Convert HCP data to HDF5 format"""

import argparse
import os

import h5py
import pandas as pd
import numpy as np

from pathlib import Path

from src.data.load_data import load_hcp_subjects

timeseries_dir = "/home/mschoett/projects/rrg-pbellec/mschoett/HCP_fMRI_timeseries"


out_folder = "/home/mschoett/projects/rrg-pbellec/mschoett/hcp_h5"
path_subjects = "/home/mschoett/projects/rrg-pbellec/mschoett/rs-autoregression-prediction/inputs/subjects/hcp_complete_rest.csv"
# path_subjects = "/home/mschoett/projects/rrg-pbellec/mschoett/rs-autoregression-prediction/inputs/subjects/hcp_complete_rest_test.csv"

factors = [0.25, 0.5, 1, 2, 4]

subjects = load_hcp_subjects(path_subjects)
sessions = ["ses-01", "ses-02"]
tasks = ["rest1", "rest2"]

for factor in factors:
    print(factor)
    for subject in subjects:
        subject = str(subject)
        if factor <= 1:
            task = "rest1"
            ses = "ses-01"
            ts_file = f"sub-{subject}_task-{task}_{ses}_desc-timeseries_scale-3.csv"
            ts_folder = Path(timeseries_dir, task, "derivatives", "preproc", f"sub-{subject}", ses)
            ts_path = Path(ts_folder, ts_file)
            ts = pd.read_csv((ts_path), index_col=0)
            time_points_total = ts.shape[0]
            time_points_shortened = int(time_points_total // (1 / factor))
            timeseries = ts[:time_points_shortened]
        elif factor == 2:
            task = "rest1"
            ts_list = []
            for ses in sessions:
                ts_file = f"sub-{subject}_task-{task}_{ses}_desc-timeseries_scale-3.csv"
                ts_folder = Path(timeseries_dir, task, "derivatives", "preproc", f"sub-{subject}", ses)
                ts_path = Path(ts_folder, ts_file)
                ts = pd.read_csv((ts_path), index_col=0)
                ts_list.append(ts)
            timeseries = pd.concat(ts_list, axis=0)
        elif factor == 4:
            ts_list = []
            for task in tasks:
                for ses in sessions:
                    ts_file = f"sub-{subject}_task-{task}_{ses}_desc-timeseries_scale-3.csv"
                    ts_folder = Path(timeseries_dir, task, "derivatives", "preproc", f"sub-{subject}", ses)
                    ts_path = Path(ts_folder, ts_file)
                    ts = pd.read_csv((ts_path), index_col=0)
                    ts_list.append(ts)
            timeseries = pd.concat(ts_list, axis=0)
        h5_sub_dir = f"rest/sub-{subject}/{factor}-sessions/sub-{subject}_task-rest_sessions-{factor}_desc-timeseries_scale-3"
        h5file = h5py.File(Path(out_folder, f"hcp_{factor}.h5"), 'a')
        # check if time series already exists
        if h5_sub_dir in h5file:
            h5file.close()
            continue
        h5file.create_dataset(h5_sub_dir, data=timeseries.values)
        h5file.close()
    


# for task in ["rest1", "rest2"]: # , "wm", "emotion", "gambling", "motor", "language", "relational", "social"]:
#     print(f"Loading task {task}")
#     subjects = os.listdir(Path(timeseries_dir, task, "derivatives", "preproc"))
#     subjects.sort()
#     for subject in subjects:
#         for ses in ["ses-01", "ses-02"]:
#             for scale in range(1,6):
#                 ts_path = Path(timeseries_dir, task, "derivatives", "preproc", subject, ses, f"{subject}_task-{task}_{ses}_desc-timeseries_scale-{scale}.csv")
#                 h5_sub_dir = f"{task}/{subject}/{ses}/{subject}_task-{task}_{ses}_desc-timeseries_scale-{scale}"
#                 # skip missing data
#                 if os.path.exists(ts_path) == False:
#                     continue
#                 h5file = h5py.File(Path(out_folder, f"hcp_{factor}.h5"), 'a')
#                 # check if time series already exists
#                 if h5_sub_dir in h5file:
#                     h5file.close()
#                     continue
#                 else:
#                     timeseries = pd.read_csv((ts_path), index_col=0)
#                     h5file.create_dataset(h5_sub_dir, data=timeseries.values)
#                     h5file.close()