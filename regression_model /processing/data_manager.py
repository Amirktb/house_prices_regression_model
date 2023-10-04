import typing as t
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from regression_model import __version__ as _version
from regression_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config


def load_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    dataframe["MSSubClass"] = dataframe["MSSubClass"].astype("O")

    # rename variables beginning with numbers to avoid syntax error 
    transformed = dataframe.rename(columns=config.model_config.variables_to_rename)
    return transformed

def save_pipeline(*, pipeline_to_save: Pipeline) -> None:
    """Save the pipeline.
    Saving the versioned model and overwrite 
    the previous saved models. Therefore, one
    trained models can be called when the package 
    is published.
    """
    save_file_name = f"{config.app_config.pipeline_save_file}{_version}"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipeline(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_save, save_path)

def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a saved pipeline"""
    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model

def remove_old_pipeline(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines.
    Ensures one-to-one mapping between the 
    package version and the model version
    to be imported and used by other 
    applications.
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()