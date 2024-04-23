# train on 100 HCP subjects with different random seeds
python src/train.py --multirun  \
  hydra/launcher=submitit_slurm \
  ++hydra.launcher.account=rrg-pbellec \
  ++hydra.launcher.timeout_min=90 \
  ++hydra.launcher.mem_gb=4 \
  ++hydra.launcher.gpus_per_node=1 \
  ++hydra.launcher.cpus_per_task=4 \
  ++random_state=1,2,3,5,8,13,21,34,55,89

# train one model on all subjects
python src/train.py --multirun  \
  hydra/launcher=submitit_slurm \
  ++data.n_sample=-1 \
  ++hydra.launcher.account=rrg-pbellec \
  ++hydra.launcher.timeout_min=90 \
  ++hydra.launcher.mem_gb=32 \
  ++hydra.launcher.gpus_per_node=1 \
  ++hydra.launcher.cpus_per_task=4 \

# scaling over subjects
python src/train.py --multirun  \
  hydra/launcher=submitit_slurm \
  ++data.n_sample=100,200,300,400,500,600,700,800,900,-1 \
  ++hydra.launcher.account=rrg-pbellec \
  ++hydra.launcher.timeout_min=720 \
  ++hydra.launcher.mem_gb=4 \
  ++hydra.launcher.gpus_per_node=1 \
  ++hydra.launcher.cpus_per_task=4 \
  ++random_state=1,2,3,5,8,13,21,34,55,89 \
  ++data.split.sessions="['ses-01','ses-02']" \
  ++data.split.tasks="['rest1','rest2']" \

# rerun those that ran out of time
python src/train.py --multirun  \
  hydra/launcher=submitit_slurm \
  ++data.n_sample=900 \
  ++hydra.launcher.account=rrg-pbellec \
  ++hydra.launcher.timeout_min=1000 \
  ++hydra.launcher.mem_gb=4 \
  ++hydra.launcher.gpus_per_node=1 \
  ++hydra.launcher.cpus_per_task=4 \
  ++random_state=1 \
  ++data.split.sessions="['ses-01','ses-02']" \
  ++data.split.tasks="['rest1','rest2']"

# train model on 100 subjects and half the time points
python src/train.py --multirun  \
  hydra/launcher=submitit_slurm \
  ++data.fraction_timepoints=0.5 \
  ++hydra.launcher.account=rrg-pbellec \
  ++hydra.launcher.timeout_min=90 \
  ++hydra.launcher.mem_gb=4 \
  ++hydra.launcher.gpus_per_node=1 \
  ++hydra.launcher.cpus_per_task=4

# run hyperparameter tuning with one session
python src/train.py --multirun \
  hydra=hyperparameters_old \
  ++data.n_sample=-1 \

# run hyperparameter tuning with all sessions
python src/train.py --multirun \
  hydra=hyperparameters_old \
  # ++data.n_sample=-1 \
  # ++data.split.tasks='["rest1", "rest2"]' \
  # ++data.split.sessions='["ses-01", "ses-02"]'

# Hao-Ting's examples
# use a small set to make sure the parameter tuning is doing things
python src/train.py --multirun hydra=hyperparameters ++hydra.launcher.timeout_min=480 ++data.n_sample=-1

# debug
python src/train.py --multirun hydra=scaling \
  ++hydra.launcher.timeout_min=180 \
  ++data.n_sample=-1 \
  ++model.FK=\'128,32,128,32,128,32,128,32\' \
  ++model.M=\'32,16,8,1\' \
  ++model.batch_size=256 \
  ++model.lag=1 \
  ++model.lr=0.04966 \
  ++model.lr_thres=0.4105 \
  ++model.dropout=0.02249 \
  ++model.seq_length=29

# scaling
python src/full_experiment.py --multirun  \
  ++hydra.launcher.mem_gb=8 \
  ++data.n_sample=16000,20000,-1 \
  ++random_state=0,1,2,4,8,42
  # ++data.n_sample=100,250,500,1000,2000,3000,4000,5000,6000,8000,10000,16000,20000,-1 \
  # ++random_state=0,1,2,4,8,42

# extraction - create symlink to model
# outputs/ccn2024/model/ -> to training results
python src/extract.py --multirun model_path=outputs/ccn2024/model/model.pkl
# outputs/ccn2024/extract/ -> to extraction results from outputs/ccn2024/model/
python src/predict.py --multirun model_path=outputs/ccn2024/extract

# one script for all scaling
python src/full_experiment.py --multirun  \
  ++hydra.launcher.timeout_min=600 \
  ++data.n_sample=-1,10000,5000,2500,1250,625,300,150,75 \
  ++random_state=0,1,2,4,8,42
