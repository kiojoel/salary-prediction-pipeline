import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import(
  AdaBoostRegressor,
  GradientBoostingRegressor,
  RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRFRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_models



@dataclass
class ModelTrainerConfig:
  trained_model_file_path = os.path.join('artifacts','model.pkl')


class ModelTrainer:
  def __init__(self):
    self.model_trainer_config=ModelTrainerConfig()

  def initiate_model_trainer(self,train_array,test_array):
    try:
      logging.info('Split training and test input data')
      X_train,y_train,X_test,y_test = (
        train_array[:,:-1],
        train_array[:,-1],
        test_array[:,:-1],
        test_array[:,-1]
      )
      models={
        'LinearRegression': LinearRegression(),
        'DecisionTreeRegressor': DecisionTreeRegressor(),
        'RandomForestRegressor': RandomForestRegressor(),
        'GradientBoostingRegressor': GradientBoostingRegressor(),
        'AdaBoostRegressor': AdaBoostRegressor(),
        'KNeighborsRegressor': KNeighborsRegressor(),
        'XGBRFRegressor': XGBRFRegressor()
      }

      model_report:dict = evaluate_models(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        models=models
      )

      model_report:dict=evaluate_models(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        models=models
      )

      best_model_score = max(sorted(model_report.values()))
      best_model_name = list(model_report.keys())[
        list(model_report.values()).index(best_model_score)
      ]
      best_model = models[best_model_name]

      if best_model_score < 0.6:
        raise CustomException("No best model found with sufficient accuracy", sys)
      else:
        logging.info(f'Best model found: {best_model_name} with score: {best_model_score}')
        save_object(
          file_path=self.model_trainer_config.trained_model_file_path,
          obj=best_model
        )

        predicted = best_model.predict(X_test)
        r2_square = r2_score(y_test, predicted)



        return best_model_name, best_model_score
    except Exception as e:
      logging.error(f'Error in model training: {str(e)}')
      raise CustomException(e, sys)
