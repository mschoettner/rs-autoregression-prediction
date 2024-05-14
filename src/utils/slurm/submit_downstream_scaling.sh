# best model from hyperparameter tuning
python src/downstream_scaling.py --multirun \
    ++feature_path='/home/mschoett/projects/rrg-pbellec/mschoett/rs-autoregression-prediction/outputs/downstream_scaling/extract' \
    ++model_path='/home/mschoett/projects/rrg-pbellec/mschoett/rs-autoregression-prediction/outputs/downstream_scaling/model'

# predict age
python src/downstream_scaling.py --multirun \
    ++feature_path='/home/mschoett/projects/rrg-pbellec/mschoett/rs-autoregression-prediction/outputs/downstream_scaling/extract' \
    ++model_path='/home/mschoett/projects/rrg-pbellec/mschoett/rs-autoregression-prediction/outputs/downstream_scaling/model' \
    ++predict_variable='age'