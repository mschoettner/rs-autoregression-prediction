python src/predict.py --multirun \
    ++feature_path='/home/mschoett/projects/rrg-pbellec/mschoett/rs-autoregression-prediction/outputs/extract/multiruns/2024-04-24_10-21-07/0' \
    ++model_path='/home/mschoett/projects/rrg-pbellec/mschoett/rs-autoregression-prediction/outputs/train/multiruns/2024-04-16_14-18-28_scaling_all/++data.n_sample\=-1\,++data.split.sessions\=\['ses-01'\,'ses-02'\]\,++data.split.tasks\=\['rest1'\,'rest2'\]\,++random_state\=1/model.pkl' \

# minimal example
python src/predict.py --multirun \
    ++feature_path='/home/mschoett/projects/rrg-pbellec/mschoett/rs-autoregression-prediction/outputs/minimal_example/extract' \
    ++model_path='/home/mschoett/projects/rrg-pbellec/mschoett/rs-autoregression-prediction/outputs/minimal_example/model'

# one session
# minimal example
python src/predict.py --multirun \
    ++feature_path='/home/mschoett/projects/rrg-pbellec/mschoett/rs-autoregression-prediction/outputs/test_one_session/extract' \
    ++model_path='/home/mschoett/projects/rrg-pbellec/mschoett/rs-autoregression-prediction/outputs/test_one_session/model'

# predicting gender
python src/predict.py --multirun \
    ++feature_path='/home/mschoett/projects/rrg-pbellec/mschoett/rs-autoregression-prediction/outputs/hcp_prediction/extract' \
    ++model_path='/home/mschoett/projects/rrg-pbellec/mschoett/rs-autoregression-prediction/outputs/hcp_prediction/model'

# predicting age
python src/predict.py --multirun \
    ++feature_path='/home/mschoett/projects/rrg-pbellec/mschoett/rs-autoregression-prediction/outputs/hcp_prediction/extract' \
    ++model_path='/home/mschoett/projects/rrg-pbellec/mschoett/rs-autoregression-prediction/outputs/hcp_prediction/model' \
    ++predict_variable='age'