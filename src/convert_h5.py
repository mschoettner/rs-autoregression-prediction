import os
import yaml
import h5py
import numpy as np

result_folder = "/home/mschoett/projects/rrg-pbellec/mschoett/rs-autoregression-prediction/outputs/extract_full_scaling/multiruns/2024-09-26_07-37-05_failed_run"
out_folder = "/home/mschoett/projects/rrg-pbellec/mschoett/rs-autoregression-prediction/outputs/gcn_features"

subfolders = os.listdir(result_folder)
# subfolders = [str(i) for i in range(0, 2)]

for subfolder in subfolders:
    if subfolder == ".submitit":
        continue
    config_path = os.path.join(result_folder, subfolder, ".hydra", "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    sessions = config["sessions"]
    fraction = config["fraction"]
    random_state = config["random_state"]
    print(f"Processing {subfolder} with {sessions} sessions, fraction {fraction} and random state {random_state}")

    h5_conv_path = os.path.join(result_folder, subfolder, f"feature_convlayers_sessions-{sessions}.h5")
    h5_r2_path = os.path.join(result_folder, subfolder, f"feature_horizon-1_sessions-{sessions}.h5")

    features = {}
    with h5py.File(h5_conv_path, "r") as f:
        for sub in f['rest']:
            features[sub] = {}
            for session in f['rest'][sub].keys():
                for measure in f['rest'][sub][session].keys():
                    for measure_name in ["1dconv", "average", "max", "std"]:
                        if measure_name == "std":
                            std = f['rest'][sub][session][measure][()]
                            std = np.squeeze(std)
                            features[sub][measure_name] = std
                        else:
                            features[sub][measure_name] = f['rest'][sub][session][measure][()]
    with h5py.File(h5_r2_path, "r") as f:
        for sub in f['rest']:
            for session in f['rest'][sub].keys():
                for measure in f['rest'][sub][session].keys():
                    for measure_name in ["r2", "mse"]:
                        features[sub][measure_name] = np.squeeze(f['rest'][sub][session][measure][()])
    out_path = os.path.join(out_folder, f"features_sessions-{sessions}_fraction-{fraction}_random-{random_state}.h5")
    with h5py.File(out_path, "w") as f:
        for sub in features.keys():
            f.create_group(sub)
            for measure in features[sub].keys():
                f[sub].create_dataset(measure, data=features[sub][measure])