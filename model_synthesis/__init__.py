from .train_random_forest_optuna import train_random_forest_with_optuna
from .train_xgboost_with_optuna import train_xgboost_with_optuna

# Models selectable by name in the pipeline settings (TRAIN_SETTINGS["model"])
TRAINERS = {
    "random_forest": train_random_forest_with_optuna,
    "xgboost": train_xgboost_with_optuna,
}
