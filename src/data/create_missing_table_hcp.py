"""Create table to find missing values in HCP data"""

import h5py
import pandas as pd

from data.load_data import load_hcp_subjects

path_subjects = "/home/mschoett/rs-autoregression-prediction/inputs/subjects/hcp_subjects.csv"
path_h5 = "/home/mschoett/rs-autoregression-prediction/inputs/connectomes/hcp.h5"

# load subject list
subjects = load_hcp_subjects(path_subjects)
tasks = ["rest1", "rest2", "wm", "emotion", "gambling", "motor", "language", "relational", "social"]
sessions = ["ses-01", "ses-02"]

h5file = h5py.File(path_h5)

missing_list = []

for subject in subjects:
    print(subject)
    for task in tasks:
        for ses in sessions:
            for scale in range(1,6):
                h5_sub_dir = f"{task}/sub-{subject}/{ses}/sub-{subject}_task-{task}_{ses}_desc-timeseries_scale-{scale}"
                if h5_sub_dir not in h5file:
                    missing_list.append([subject, task, ses, scale])
                        
h5file.close()

missing = pd.DataFrame(missing_list, columns=["subject", "task", "session", "scale"])
missing.to_csv("/home/mschoett/rs-autoregression-prediction/inputs/subjects/hcp_missing.csv")