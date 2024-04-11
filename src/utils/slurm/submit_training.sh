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
  ++data.n_sample=100,200,300,400,500,600,700,900,-1 \
  ++hydra.launcher.account=rrg-pbellec \
  ++hydra.launcher.timeout_min=90 \
  ++hydra.launcher.mem_gb=32 \
  ++hydra.launcher.gpus_per_node=1 \
  ++hydra.launcher.cpus_per_task=4 \
  # ++random_state=1,2,3,5,8,13,21,34,55,89

# higher subject numbers need more time
python src/train.py --multirun  \
  hydra/launcher=submitit_slurm \
  ++data.n_sample=700,900,-1 \
  ++hydra.launcher.account=rrg-pbellec \
  ++hydra.launcher.timeout_min=180 \
  ++hydra.launcher.mem_gb=32 \
  ++hydra.launcher.gpus_per_node=1 \
  ++hydra.launcher.cpus_per_task=4



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

python src/train.py --multirun  \
  hydra/launcher=submitit_slurm \
  ++hydra.launcher.account=rrg-pbellec \
  ++hydra.launcher.timeout_min=90 \
  ++hydra.launcher.mem_gb=4 \
  ++hydra.launcher.gpus_per_node=1 \
  ++data.n_sample=-1
