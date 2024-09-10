# Minimal example
python src/extract_one_pipe.py --multirun \
    ++model_path='/home/mschoett/projects/rrg-pbellec/mschoett/rs-autoregression-prediction/outputs/train/multiruns/2024-04-29_13-51-25_minimal_example/++data.n_sample\=100\,++data.sessions\=0.25/model.pkl'

# Medium model
python src/extract_one_pipe.py --multirun \
    ++model_path='/home/mschoett/projects/rrg-pbellec/mschoett/rs-autoregression-prediction/outputs/train/multiruns/2024-05-28_09-56-51_scaling_medium_rs/++data.n_sample\=-1\,++data.sessions\=4\,++random_state\=1/model.pkl'

# Medium model with several random seeds
python src/extract_one_pipe.py --multirun \
    ++random_state=1,2,3,5,8,13,21,34,55,89 \
    ++model_path='/home/mschoett/projects/rrg-pbellec/mschoett/rs-autoregression-prediction/outputs/train/multiruns/2024-05-28_09-56-51_scaling_medium_rs/++data.n_sample\=-1\,++data.sessions\=4\,++random_state\=${random_state}/model.pkl'

# Model with best hyperparameters August 13
python src/extract_one_pipe.py --multirun \
    ++model_path='/home/mschoett/projects/rrg-pbellec/mschoett/rs-autoregression-prediction/outputs/train/multiruns/2024-08-13_09-25-10_hyperparameter/best/model.pkl'