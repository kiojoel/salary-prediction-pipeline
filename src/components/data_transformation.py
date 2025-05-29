import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocesor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_tranformer_object(self):
        '''
        This function transforms the data
        '''
        try:
            numerical_columns = ['Age', 'Years of Experience']
            categorical_columns = ['Gender', 'Education Level', 'Job Title']

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            # Add max_categories to prevent too many features
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore', max_categories=50))
                ]
            )

            logging.info('Numerical column standard scaling completed')
            logging.info('categorical column encoding completed')

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns),
                ],
                remainder='drop'  # Explicitly drop remaining columns
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train dataframe shape: {train_df.shape}')
            logging.info(f'Test dataframe shape: {test_df.shape}')

            # Check for data integrity
            logging.info(f'Train df columns: {train_df.columns.tolist()}')
            logging.info(f'Train df dtypes:\n{train_df.dtypes}')

            preprocessing_obj = self.get_data_tranformer_object()
            target_column = 'Salary'

            # Clean data - remove rows with NaN values
            initial_train_shape = train_df.shape[0]
            initial_test_shape = test_df.shape[0]

            train_df_clean = train_df.dropna()
            test_df_clean = test_df.dropna()

            logging.info(f'Dropped {initial_train_shape - train_df_clean.shape[0]} rows from train set due to NaN')
            logging.info(f'Dropped {initial_test_shape - test_df_clean.shape[0]} rows from test set due to NaN')
            logging.info(f'Clean train shape: {train_df_clean.shape}')
            logging.info(f'Clean test shape: {test_df_clean.shape}')

            # Separate features and target
            input_feature_train_df = train_df_clean.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_df_clean[target_column]

            input_feature_test_df = test_df_clean.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df_clean[target_column]

            logging.info(f'Features train shape: {input_feature_train_df.shape}')
            logging.info(f'Target train shape: {target_feature_train_df.shape}')
            logging.info(f'Features test shape: {input_feature_test_df.shape}')
            logging.info(f'Target test shape: {target_feature_test_df.shape}')

            # Check unique values in categorical columns
            for col in ['Gender', 'Education Level', 'Job Title']:
                unique_count = input_feature_train_df[col].nunique()
                logging.info(f'{col} has {unique_count} unique values')
                if unique_count > 100:
                    logging.warning(f'{col} has too many categories ({unique_count}), consider reducing')

            logging.info('Starting preprocessing transformation...')

            # Apply preprocessing with error handling
            try:
                input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
                logging.info('Training data transformation completed')
                logging.info(f'Transformed train array shape: {input_feature_train_arr.shape}')
                logging.info(f'Transformed train array type: {type(input_feature_train_arr)}')

                input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
                logging.info('Test data transformation completed')
                logging.info(f'Transformed test array shape: {input_feature_test_arr.shape}')
                logging.info(f'Transformed test array type: {type(input_feature_test_arr)}')

            except Exception as e:
                logging.error(f'Error during preprocessing: {str(e)}')
                raise CustomException(f'Preprocessing failed: {str(e)}', sys)

            # Convert sparse matrix to dense if needed
            if hasattr(input_feature_train_arr, 'toarray'):
                input_feature_train_arr = input_feature_train_arr.toarray()
                logging.info('Converted sparse matrix to dense array for training data')

            if hasattr(input_feature_test_arr, 'toarray'):
                input_feature_test_arr = input_feature_test_arr.toarray()
                logging.info('Converted sparse matrix to dense array for test data')

            # Ensure arrays are numpy arrays and 2D
            input_feature_train_arr = np.asarray(input_feature_train_arr)
            input_feature_test_arr = np.asarray(input_feature_test_arr)

            if input_feature_train_arr.ndim == 1:
                input_feature_train_arr = input_feature_train_arr.reshape(-1, 1)
                logging.info('Reshaped 1D train array to 2D')

            if input_feature_test_arr.ndim == 1:
                input_feature_test_arr = input_feature_test_arr.reshape(-1, 1)
                logging.info('Reshaped 1D test array to 2D')

            logging.info(f'Final input train array shape: {input_feature_train_arr.shape}')
            logging.info(f'Final input test array shape: {input_feature_test_arr.shape}')

            # Prepare target arrays
            target_feature_train_arr = np.asarray(target_feature_train_df).reshape(-1, 1)
            target_feature_test_arr = np.asarray(target_feature_test_df).reshape(-1, 1)

            logging.info(f'Target train array shape: {target_feature_train_arr.shape}')
            logging.info(f'Target test array shape: {target_feature_test_arr.shape}')

            # Final validation
            if input_feature_train_arr.shape[0] != target_feature_train_arr.shape[0]:
                error_msg = f"Row mismatch: features {input_feature_train_arr.shape[0]} vs target {target_feature_train_arr.shape[0]}"
                logging.error(error_msg)
                raise CustomException(error_msg, sys)

            # Concatenate arrays
            logging.info('Concatenating features and target...')
            train_arr = np.concatenate([input_feature_train_arr, target_feature_train_arr], axis=1)
            test_arr = np.concatenate([input_feature_test_arr, target_feature_test_arr], axis=1)

            logging.info(f'Final train array shape: {train_arr.shape}')
            logging.info(f'Final test array shape: {test_arr.shape}')

            # Save preprocessing object
            save_object(
                file_path=self.data_transformation_config.preprocesor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info('Preprocessing object saved successfully')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocesor_obj_file_path,
            )

        except Exception as e:
            logging.error(f'Error in data transformation: {str(e)}')
            logging.error(f'Error type: {type(e).__name__}')
            raise CustomException(e, sys)