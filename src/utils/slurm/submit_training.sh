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
  ++hydra.launcher.timeout_min=180 \
  ++hydra.launcher.mem_gb=4 \
  ++hydra.launcher.gpus_per_node=1 \
  ++hydra.launcher.cpus_per_task=4 \
  ++random_state=1,2,3,5,8,13,21,34,55,89

python src/train.py --multirun  \
  hydra/launcher=submitit_slurm \
  ++data.n_sample=800 \
  ++hydra.launcher.account=rrg-pbellec \
  ++hydra.launcher.timeout_min=180 \
  ++hydra.launcher.mem_gb=4 \
  ++hydra.launcher.gpus_per_node=1 \
  ++hydra.launcher.cpus_per_task=4 \
  ++random_state=1,2,3,5,8,13,21,34,55,89

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
  hydra=hyperparameters \
  ++data.n_sample=-1 \
  # ++data.split.tasks='["rest1", "rest2"]' \
  # ++data.split.sessions='["ses-01", "ses-02"]'

# Hao-Ting's examples
# use a small set to make sure the parameter tuning is doing things
python src/train.py --multirun hydra=hyperparameters ++hydra.launcher.timeout_min=480 ++data.n_sample=-1

# debug
python src/train.py \
  ++data.n_sample=-1 \
  ++model.FK=\'8,3,8,3,8,3\' \
  ++model.M=\'8,1\' \
  ++model.batch_size=127 \
  ++model.lag=3 \
  ++model.lr=0.65 \
  ++model.lr_thres=0.702 \
  ++model.seq_length=52

# train small default model with the full ukbb
python src/train.py --multirun hydra=hyperparameters \
  ++hydra.launcher.timeout_min=240 \
  ++hydra.launcher.mem_gb=4 \
  ++torch_device=cpu \
  ++data.n_sample=-1

# scaling
python src/train.py --multirun  \
  hydra=scaling \
  model=ccn_abstract \
  ++data.n_sample=100,250,500,1000,2000,3000,4000,5000,6000,8000,10000,16000,20000,-1 \
  ++random_state=0,1,2,4,8,42

# extraction - create symlink to model
python src/extract.py --multirun model_path=outputs/ccn2024/best_model/model.pkl
