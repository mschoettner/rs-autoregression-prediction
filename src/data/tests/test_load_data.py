import pandas as pd
import h5py

from data.load_data import load_hcp_dset_path, load_data

path_restricted = "/home/mschoett/rs-autoregression-prediction/inputs/behavior/hcp_behavioral_RESTRICTED.csv"
path_subjects = "/home/mschoett/rs-autoregression-prediction/inputs/subjects/hcp_subjects.csv"
path_rest_subjects = "/home/mschoett/rs-autoregression-prediction/inputs/subjects/hcp_complete_rest.csv"
path_h5 = "/home/mschoett/rs-autoregression-prediction/inputs/connectomes/hcp.h5"


# data_dict = load_hcp_dset_path(path_restricted, path_subjects)
# data_dict = load_hcp_dset_path(path_restricted, path_subjects, n_sample=-1)
# data_dict = load_hcp_dset_path(path_restricted, path_subjects, tasks=["emotion", "wm"])
# data_dict = load_hcp_dset_path(path_restricted, path_subjects, sessions=["ses-02"])
# data_dict = load_hcp_dset_path(path_restricted, path_subjects, sessions=["ses-01", "ses-02"])
# data_dict = load_hcp_dset_path(path_restricted, path_subjects, random_state=42)
# data_dict2 = load_hcp_dset_path(path_restricted, path_subjects, random_state=8)
# data_dict = load_hcp_dset_path(path_restricted, path_subjects, test_set=0.1)
# data_dict = load_hcp_dset_path(path_restricted, path_subjects, val_set=0.1)
# data_dict = load_hcp_dset_path(path_restricted, path_subjects, atlas_scale=1)
data_dict = load_hcp_dset_path(path_restricted, path_rest_subjects, n_sample=100, val_set=0.15, test_set=0.15)

# with h5py.File(path_h5, "r") as h5file:
#     for p in data_dict["train"]:
#         print(h5file[p])
#         # data_list.append(h5file[p][:])

data = load_data(path_h5, data_dict['train'])

print("finish")