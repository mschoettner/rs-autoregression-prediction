# Scaling over time points
# rest1, ses-01, fractions 0.25, 0.5, 1.0
python src/train.py --multirun  \
  hydra/launcher=submitit_slurm \
  ++data.n_sample=-1 \
  ++hydra.launcher.account=rrg-pbellec \
  ++hydra.launcher.timeout_min=180 \
  ++hydra.launcher.mem_gb=4 \
  ++hydra.launcher.gpus_per_node=1 \
  ++hydra.launcher.cpus_per_task=4 \
  ++data.fraction_timepoints=0.25,0.5,1.0 \
  ++random_state=1,2,3,5,8,13,21,34,55,89
# rest1, ses-01, ses-02
python src/train.py --multirun  \
  hydra/launcher=submitit_slurm \
  ++data.n_sample=-1 \
  ++hydra.launcher.account=rrg-pbellec \
  ++hydra.launcher.timeout_min=360 \
  ++hydra.launcher.mem_gb=4 \
  ++hydra.launcher.gpus_per_node=1 \
  ++hydra.launcher.cpus_per_task=4 \
  ++data.split.sessions='["ses-01", "ses-02"]' \
  ++random_state=1,2,3,5,8,13,21,34,55,89
# rest1, rest2, ses-01, ses-02
python src/train.py --multirun  \
  hydra/launcher=submitit_slurm \
  ++data.n_sample=-1 \
  ++hydra.launcher.account=rrg-pbellec \
  ++hydra.launcher.timeout_min=720 \
  ++hydra.launcher.mem_gb=4 \
  ++hydra.launcher.gpus_per_node=1 \
  ++hydra.launcher.cpus_per_task=4 \
  ++data.split.sessions='["ses-01", "ses-02"]' \
  ++data.split.tasks='["rest1", "rest2"]' \
  ++random_state=1,2,3,5,8,13,21,34,55,89