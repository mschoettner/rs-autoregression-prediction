# best model from hyperparameter tuning
python src/downstream_scaling.py --multirun \
    ++feature_path='/home/mschoett/projects/rrg-pbellec/mschoett/rs-autoregression-prediction/outputs/downstream_scaling/extract' \
    ++model_path='/home/mschoett/projects/rrg-pbellec/mschoett/rs-autoregression-prediction/outputs/downstream_scaling/model'

# predict factor scores and age
python src/downstream_scaling.py --multirun \
    ++feature_path='/home/mschoett/projects/rrg-pbellec/mschoett/rs-autoregression-prediction/outputs/downstream_scaling/extract' \
    ++model_path='/home/mschoett/projects/rrg-pbellec/mschoett/rs-autoregression-prediction/outputs/downstream_scaling/model' \
    ++predict_variable='mental_health','cognition','processing_speed','substance_use','age'