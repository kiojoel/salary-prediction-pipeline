import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
  def __init__(self):
    pass

  def predict(self, features):
    try:
      model_path = 'artifacts\model.pkl'
      preprocessor_path = 'artifacts\preprocessor.pkl'
      model= load_object(file_path=model_path)
      preprocessor = load_object(file_path=preprocessor_path)
      data_scaled = preprocessor.transform(features)
      pred = model.predict(data_scaled)
      return pred

    except Exception as e:
      raise CustomException(e, sys)


class CustomData:
  def __init__(self,
               age:int,
               gender:str,
               education:str,
               job:str,
               experience:int,):
    self.age = age
    self.gender = gender
    self.education = education
    self.job = job
    self.experience = experience


  def get_data_as_dataframe(self):
    try:
      custom_data_input_dict = {
            "Age": [self.age],
            "Gender": [self.gender],
            "Education Level": [self.education],
            "Job Title": [self.job],
            "Years of Experience": [self.experience]
        }
      df = pd.DataFrame(custom_data_input_dict)
      return df
    except Exception as e:
      raise CustomException(e, sys)