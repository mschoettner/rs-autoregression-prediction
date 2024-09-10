# minimal example
python src/predict_one_pipe.py --multirun \
    ++feature_path='/home/mschoett/projects/rrg-pbellec/mschoett/rs-autoregression-prediction/outputs/minimal_example_one_pipe/extract' \
    ++model_path='/home/mschoett/projects/rrg-pbellec/mschoett/rs-autoregression-prediction/outputs/minimal_example_one_pipe/model'

# medium model, first random seed
python src/predict_one_pipe.py --multirun \
    ++feature_path='/home/mschoett/projects/rrg-pbellec/mschoett/rs-autoregression-prediction/outputs/medium_model_10_seeds/extract' \
    ++model_path='/home/mschoett/projects/rrg-pbellec/mschoett/rs-autoregression-prediction/outputs/medium_model_10_seeds/model'