# Using one session
python src/extract.py --multirun \
    ++model_path='/home/mschoett/projects/rrg-pbellec/mschoett/rs-autoregression-prediction/outputs/train/multiruns/2024-04-10_14-59-15_scaling_700+900+all/++data.n_sample\=-1/model.pkl' \
    ++hydra.launcher.account=rrg-pbellec \
    ++hydra.launcher.timeout_min=180 \
    ++hydra.launcher.gpus_per_node=1 \
    # hydra/launcher=submitit_slurm \

# Using all sessions
python src/extract.py --multirun \
    ++model_path='/home/mschoett/projects/rrg-pbellec/mschoett/rs-autoregression-prediction/outputs/train/multiruns/2024-04-16_14-18-28_scaling_all/++data.n_sample\=-1\,++data.split.sessions\=\['ses-01'\,'ses-02'\]\,++data.split.tasks\=\['rest1'\,'rest2'\]\,++random_state\=1/model.pkl' \
    ++hydra.launcher.account=rrg-pbellec \
    ++hydra.launcher.timeout_min=180 \
    ++hydra.launcher.gpus_per_node=1

# Using best parameters so far
python src/extract.py --multirun \
    ++model_path='/home/mschoett/projects/rrg-pbellec/mschoett/rs-autoregression-prediction/outputs/train/multiruns/2024-04-16_14-18-28_scaling_all/++data.n_sample\=-1\,++data.split.sessions\=\['ses-01'\,'ses-02'\]\,++data.split.tasks\=\['rest1'\,'rest2'\]\,++random_state\=1/model.pkl' \
    ++hydra.launcher.account=rrg-pbellec \
    ++hydra.launcher.timeout_min=180 \
    ++hydra.launcher.gpus_per_node=1