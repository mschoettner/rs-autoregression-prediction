"""Extracts the GCN features from the full scaling run."""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import h5py
import hydra
import torch

from fmri_autoreg.models.predict_model import predict_horizon
from fmri_autoreg.tools import load_model
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

log = logging.getLogger(__name__)

@hydra.main(version_base="1.3", config_path="../config", config_name="extract_full_scaling")
def main(params: DictConfig) -> None:
    """Extracts the GCN features from the full scaling run."""

    from src.model.extract_features import (
        extract_convlayers,
        pooling_convlayers,
    )
    # Load path and parameters
    model_folder = Path(params["model_folder"])
    model_subfolder = Path(f"++data.sessions={params['sessions']},++data.split.fraction={params['fraction']},++random_state={params['random_state']}")
    model_path = Path(model_folder, model_subfolder, "model.pkl")
    model_config = OmegaConf.load(model_path.parent / ".hydra/config.yaml")
    params = OmegaConf.merge(model_config, params)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    log.info(f"Working on {device}.")
    compute_edge_index = "Chebnet" in params["model"]["model"]
    thres = params["model"]["edge_index_thres"] if compute_edge_index else None
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    output_dir = Path(output_dir)
    log.info(f"Current working directory : {os.getcwd()}")
    log.info(f"Output directory  : {output_dir}")
    horizon = params["horizon"]

        # load test set subject path from the training
    with open(model_path.parent / "train_test_split.json", "r") as f:
        subj = json.load(f)

    subj_list = subj["train"]+subj["test"]

    # filter out participants IDs from file paths
    participant_id = [
        p.split("/")[-1].split("sub-")[-1].split("_")[0] for p in subj_list
    ]

    # subject path template
    subject_path_template = "/rest/sub-{sub}/sessions-{n}/sub-{sub}_task-rest_sessions-{n}_desc-timeseries_scale-3"
    
    # save test data path to a text file for easy future reference
    with open(output_dir / "test_set_connectome.txt", "w") as f:
        for item in subj_list:
            f.write("%s\n" % item)

    log.info("Load model")
    with torch.no_grad():
        model = load_model(model_path)
        if isinstance(model, torch.nn.Module):
            model.to(torch.device(device)).eval()
    
        n = params["sessions"]
        
        # data file for this n
        data_file = f"inputs/connectomes/sourcedata/hcp_h5/hcp_{n}.h5"
        # subject list for this n
        subj_list = []
        for sub in participant_id:
            cur_sub_path = subject_path_template.format(sub=sub, n=n)
            subj_list.append(cur_sub_path)
        
        # save file path for this n
        with open(output_dir / f"test_set_connectome_sessions-{n}.txt", "w") as f:
            for item in subj_list:
                f.write("%s\n" % item)
        
        output_horizon_path = (
            Path(output_dir) / f"feature_horizon-{horizon}_sessions-{n}.h5"
        )
        with h5py.File(output_horizon_path, "a") as f:
            f.attrs["complied_date"] = str(datetime.today())
            f.attrs["based_on_model"] = str(model_path)
            f.attrs["horizon"] = horizon
        
        log.info(f"Predicting t+{horizon} of each subject, using {n} sessions")
        for h5_dset_path in tqdm(subj_list):
            # get the prediction of t+1
            r2, _, _ = predict_horizon(
                model=model,
                seq_length=params["model"]["seq_length"],
                horizon=horizon,
                data_file=data_file,
                dset_path=h5_dset_path,
                batch_size=params["model"]["batch_size"],
                stride=params["model"]["time_stride"],
                standardize=False,  # the ts is already standardized
            )
            # save the original output to a h5 file
            with h5py.File(output_horizon_path, "a") as f:
                new_ds_path = h5_dset_path.replace("timeseries", "r2map")
                f[new_ds_path] = r2

        output_conv_path = Path(output_dir) / f"feature_convlayers_sessions-{n}.h5"
        # save the model parameters in the h5 files
        with h5py.File(output_conv_path, "a") as f:
            f.attrs["complied_date"] = str(datetime.today())
            f.attrs["based_on_model"] = str(model_path)

        log.info("extract convo layers")
        for h5_dset_path in tqdm(subj_list):
            convlayers = extract_convlayers(
                data_file=data_file,
                h5_dset_path=h5_dset_path,
                model=model,
                seq_length=params["model"]["seq_length"],
                time_stride=params["model"]["time_stride"],
                lag=params["model"]["lag"],
                compute_edge_index=compute_edge_index,
                thres=thres,
            )
            # save the original output to a h5 file
            # EDIT: don't save the convolution layers, they're too big
            # with h5py.File(output_conv_path, "a") as f:
            #     new_ds_path = h5_dset_path.replace("timeseries", "convlayers")
            #     f[new_ds_path] = convlayers.numpy()
            convlayers_F = [
                int(F)
                for i, F in enumerate(params["model"]["FK"].split(","))
                if i % 2 == 0
            ]
            # get the pooling features of the assigned layer
            for method in ["average", "max", "std", "1dconv"]:
                features = pooling_convlayers(
                    convlayers=convlayers,
                    pooling_methods=method,
                    pooling_target="parcel",
                    layer_index=params["convlayer_index"],
                    layer_structure=convlayers_F,
                )
                # save the original output to a h5 file
                with h5py.File(output_conv_path, "a") as f:
                    new_ds_path = h5_dset_path.replace("timeseries", method)
                    f[new_ds_path] = features

        # save the original output to a h5 file
        with h5py.File(output_conv_path, "a") as f:
            f.attrs["convolution_layers_F"] = convlayers_F


if __name__ == "__main__":
    main()