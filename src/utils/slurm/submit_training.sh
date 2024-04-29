# train on 100 HCP subjects, one session, interactively
python src/train.py --multirun  \
  ++data.n_sample=100 \
  ++data.sessions=0.25

# train on 100 HCP subjects, one session
python src/train.py --multirun  \
  hydra/launcher=submitit_slurm \
  ++data.n_sample=100 \
  ++data.sessions=1 \
  ++hydra.launcher.account=rrg-pbellec \
  ++hydra.launcher.timeout_min=90 \
  ++hydra.launcher.mem_gb=4 \
  ++hydra.launcher.gpus_per_node=1 \
  ++hydra.launcher.cpus_per_task=4

# train one model on all subjects
python src/train.py --multirun  \
  hydra/launcher=submitit_slurm \
  ++hydra.launcher.account=rrg-pbellec \
  ++hydra.launcher.timeout_min=900 \
  ++hydra.launcher.mem_gb=4 \
  ++hydra.launcher.gpus_per_node=1 \
  ++hydra.launcher.cpus_per_task=4 \

# scaling over subjects
python src/train.py --multirun  \
  hydra/launcher=submitit_slurm \
  ++data.n_sample=100,200,300,400,500,600,700,800,900,-1 \
  ++hydra.launcher.account=rrg-pbellec \
  ++hydra.launcher.timeout_min=1000 \
  ++hydra.launcher.mem_gb=4 \
  ++hydra.launcher.gpus_per_node=1 \
  ++hydra.launcher.cpus_per_task=4 \
  ++random_state=1,2,3,5,8,13,21,34,55,89 \

# Scaling over time points
python src/train.py --multirun \
  hydra/launcher=submitit_slurm \
  ++hydra.launcher.account=rrg-pbellec \
  ++hydra.launcher.timeout_min=1000 \
  ++hydra.launcher.mem_gb=4 \
  ++hydra.launcher.gpus_per_node=1 \
  ++hydra.launcher.cpus_per_task=4 \
  ++data.sessions=0.25,0.5,1,2,4 \
  ++random_state=1,2,3,5,8,13,21,34,55,89

# run hyperparameter tuning with old parameters
python src/train.py --multirun \
  hydra=hyperparameters_old \

# run hyperparameter tuning with new parameters
python src/train.py --multirun \
  hydra=hyperparameters

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
