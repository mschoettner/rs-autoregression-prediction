# test run
python src/extract_full_scaling.py --multirun

# full run
python src/extract_full_scaling.py --multirun \
  ++hydra.launcher.account=rrg-pbellec \
  ++hydra.launcher.timeout_min=360 \
  ++hydra.launcher.mem_gb=16 \
  ++hydra.launcher.gpus_per_node=1 \
  ++hydra.launcher.cpus_per_task=4 \
  ++fraction=0.2,0.4,0.6,0.8,1.0 \
  ++random_state=1,2,3,5,8,13,21,34,55,89 \
  ++sessions=0.25,0.5,1,2,4

# Debug error for more than one session
python src/extract_full_scaling.py --multirun \
  ++fraction=0.2 \
  ++random_state=1 \
  ++sessions=0.5