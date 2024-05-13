"""
Scaling of the downstream prediction. Taking the outputs from the pretrained model,
how well does performance of the downstream prediction scale? 
"""

import json
import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd

from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    Ridge,
    RidgeClassifier,
)
from sklearn.model_selection import StratifiedKFold, GroupKFold, GroupShuffleSplit
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import LinearSVC, LinearSVR

from src.data.load_data import load_hcp_groups

baseline_details = {
    "connectome": {
        "data_file": None,
        "data_file_pattern": None,
        "plot_label": "Connectome",
    },
    "conv_avg": {
        "data_file": None,
        "data_file_pattern": "average",
        "plot_label": "Conv layers \n avg pooling",
    },
    "conv_std": {
        "data_file": None,
        "data_file_pattern": "std",
        "plot_label": "Conv layers \n std pooling",
    },
    "conv_max": {
        "data_file": None,
        "data_file_pattern": "max",
        "plot_label": "Conv layers \n max pooling",
    },
    "conv_conv1d": {
        "data_file": None,
        "data_file_pattern": "1dconv",
        "plot_label": "Conv layers \n 1D convolution",
    },
    "avgr2": {
        "data_file": None,
        "data_file_pattern": "r2map",
        "plot_label": "t+1\n average R2",
    },
    "r2map": {
        "data_file": None,
        "data_file_pattern": "r2map",
        "plot_label": "t+1\nR2 map",
    },
}


log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../config", config_name="downstream_scaling")
def main(params: DictConfig) -> None:
    from src.data.load_data import get_model_data, load_h5_data_path
    
    # parse parameters
    n_sessions = params["n_sessions_list"]
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    output_dir = Path(output_dir)
    log.info(f"Output data {output_dir}")
    feature_path = Path(params["feature_path"])
    phenotype_file = Path(params["phenotype_file"])
    path_restricted = Path(params["path_restricted"])
    # groups = pd.read_csv(group_file, index_col=0)["Family_ID"]

    k_splits = params["k_splits"]
    random_splits = params["random_splits"]

    for n in n_sessions:
        convlayers_path = feature_path / f"feature_convlayers_sessions-{n}.h5"
        feature_t1_file = feature_path / f"feature_horizon-{params['horizon']}_sessions-{n}.h5"
        test_subjects = feature_path / f"test_set_connectome_sessions-{n}.txt"
        # data file for this n
        data_file = f"inputs/connectomes/sourcedata/hcp_h5/hcp_{n}.h5"
        # model_config = OmegaConf.load(feature_path / ".hydra/config.yaml")
        model_config = OmegaConf.load(
            feature_path.parent / "model/.hydra/config.yaml"
        )
        # combine the autoregression parameters with the other parameters
        params = OmegaConf.merge(model_config, params)
        log.info(params)
        
        # load test set subject path from the training
        with open(test_subjects, "r") as f:
            subj = f.read().splitlines()
            
        # filter out participants IDs from file paths
        participant_id = [
        p.split("/")[-1].split("sub-")[-1].split("_")[0] for p in subj
        ]

        groups = load_hcp_groups(path_restricted=path_restricted,
                                 subjects=list(map(int, participant_id)))
            
        
        # add additional infos to the baseline features depending on which are present
        for key in baseline_details:
            if "r2" in key:
                baseline_details[key]["data_file"] = feature_t1_file
            elif "conv" in key:
                baseline_details[key]["data_file"] = convlayers_path
            elif "connectome" in key:
                baseline_details[key]["data_file"] = data_file
                baseline_details[key]["data_file_pattern"] = subj
            else:
                pass
        
        log.info(f"Predicting {params['predict_variable']}, using {n} sessions.")
        
        
        if params["predict_variable"] == "sex" or params["predict_variable"] == "gender":
            # four baseline models for sex
            svm = LinearSVC(C=100, penalty="l2", max_iter=1000000, random_state=42)
            lr = LogisticRegression(
                penalty="l2", max_iter=100000, random_state=42, n_jobs=-1
            )
            rr = RidgeClassifier(random_state=42, max_iter=100000)
            mlp = MLPClassifier(
                hidden_layer_sizes=(64, 64),
                max_iter=100000,
                random_state=42,
            )
            clf_names = ["SVM", "LogisticR", "Ridge", "MLP"]

        elif params["predict_variable"] == "age":  # need to fix this
            # four baseline models for age
            svm = LinearSVR(C=100, max_iter=1000000, random_state=42)
            lr = LinearRegression(n_jobs=-1)
            rr = Ridge(random_state=42, max_iter=100000)
            mlp = MLPRegressor(
                hidden_layer_sizes=(64, 64),
                max_iter=100000,
                random_state=42,
            )
            clf_names = ["SVM", "LinearR", "Ridge", "MLP"]
            
        elif params["predict_variable"] in ["mental_health", "cognition", "processing_speed", "substance_use"]:
            # four baseline models for factor scores
            svm = LinearSVC(C=100, penalty="l2", max_iter=1000000, random_state=42)
            lr = LogisticRegression(
                penalty="l2", max_iter=100000, random_state=42, n_jobs=-1
            )
            rr = RidgeClassifier(random_state=42, max_iter=100000)
            mlp = MLPClassifier(
                hidden_layer_sizes=(64, 64),
                max_iter=100000,
                random_state=42,
            )
            clf_names = ["SVM", "LogisticR", "Ridge", "MLP"]

        else:
            raise ValueError("predict_variable must be either sex, gender, age, or one of the factor scores")

        baselines_df = {
            "feature": [],
            "score": [],
            "classifier": [],
            "fold": [],
        }

        for measure in baseline_details:
            log.info(f"Start training {measure}")
            log.info(f"Load data {baseline_details[measure]['data_file']}")
            if measure == "connectome":
                dset_path = baseline_details[measure]["data_file_pattern"]
            else:
                dset_path = load_h5_data_path(
                    baseline_details[measure]["data_file"],
                    baseline_details[measure]["data_file_pattern"],
                    shuffle=True,
                    random_state=params["random_state"],
                )
            log.info(f"found {len(dset_path)} subjects with {measure} data.")
            dataset = get_model_data(
                baseline_details[measure]["data_file"],
                dset_path=dset_path,
                phenotype_file=phenotype_file,
                measure=measure,
                label=params["predict_variable"],
                log=log,
            )
            log.info("Start training...")

            tng, tst = next(
                StratifiedKFold(n_splits=5, shuffle=True).split(
                    dataset["data"], dataset["label"]
                )
            )  # only one fold

            if random_splits:
                cv = GroupShuffleSplit(n_splits=k_splits, random_state=42)
            else:
                cv = GroupKFold(n_splits=k_splits)

            for k, (tng, tst) in enumerate(cv.split(dataset["data"], dataset["label"], groups=groups)):
                log.info(f"Fold {k+1} out of {k_splits}...")

                for clf_name, clf in zip(clf_names, [svm, lr, rr, mlp]):
                    clf.fit(dataset["data"][tng], dataset["label"][tng])
                    score = clf.score(dataset["data"][tst], dataset["label"][tst])
                    log.info(f"{measure} - {clf_name} score: {score:.3f}")
                    baselines_df["feature"].append(measure)
                    baselines_df["score"].append(score)
                    baselines_df["classifier"].append(clf_name)
                    baselines_df["fold"].append(k)

        # save the results
        # json for safe keeping
        with open(
            output_dir / f"simple_classifiers_{params['predict_variable']}_sessions-{n}_splits-{k_splits}.json",
            "w",
        ) as f:
            json.dump(baselines_df, f, indent=4)

        baselines_df = pd.DataFrame(baselines_df)
        baselines_df.to_csv(
            output_dir / f"simple_classifiers_{params['predict_variable']}_sessions-{n}_splits-{k_splits}.tsv",
            sep="\t",
        )


if __name__ == "__main__":
    main()
