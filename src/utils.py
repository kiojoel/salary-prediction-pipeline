import os
import sys

import numpy as np
import pandas as pd
import dill

from sklearn.metrics import r2_score


from src.exception import CustomException

def save_object(file_path, obj):
  try:
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)
    with open(file_path, 'wb') as file_obj:
      dill.dump(obj, file_obj)

  except Exception as e:
    raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models):
    """
    Evaluate the performance of different regression models.

    Parameters:
    - X_train: Training feature set
    - y_train: Training target variable
    - X_test: Testing feature set
    - y_test: Testing target variable
    - models: Dictionary of model names and their instances

    Returns:
    - model_report: Dictionary with model names as keys and their R2 scores as values
    """
    try:
      report = {}

      for i in range(len(list(models))):
        model = list(models.values())[i]
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        train_model_score = r2_score(y_train, y_train_pred)
        test_model_score = r2_score(y_test, y_test_pred)
        report[list(models.keys())[i]] = test_model_score

      return report
    except Exception as e:
      raise CustomException(e, sys)


def load_object(file_path):
    """
    Load a Python object from a file.

    Parameters:
    - file_path: Path to the file containing the object

    Returns:
    - obj: The loaded object
    """
    try:
      with open(file_path, 'rb') as file_obj:
        return dill.load(file_obj)
    except Exception as e:
      raise CustomException(e, sys)