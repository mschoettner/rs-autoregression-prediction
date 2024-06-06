# best model from hyperparameter tuning
python src/downstream_scaling.py --multirun \
    ++feature_path='/home/mschoett/projects/rrg-pbellec/mschoett/rs-autoregression-prediction/outputs/downstream_scaling/extract' \
    ++model_path='/home/mschoett/projects/rrg-pbellec/mschoett/rs-autoregression-prediction/outputs/downstream_scaling/model'

# rerun failed job
python src/downstream_scaling.py --multirun \
    ++feature_path='/home/mschoett/projects/rrg-pbellec/mschoett/rs-autoregression-prediction/outputs/downstream_scaling/extract' \
    ++model_path='/home/mschoett/projects/rrg-pbellec/mschoett/rs-autoregression-prediction/outputs/downstream_scaling/model' \
    ++predict_variable='age' \
    ++n_sessions=2 \
    ++fraction_sample=0.9 \
    ++hydra.launcher.timeout_min=360


# large model
python src/downstream_scaling.py --multirun \
    ++feature_path='/home/mschoett/projects/rrg-pbellec/mschoett/rs-autoregression-prediction/outputs/downstream_scaling/extract_large_2' \
    ++model_path='/home/mschoett/projects/rrg-pbellec/mschoett/rs-autoregression-prediction/outputs/downstream_scaling/large_model'