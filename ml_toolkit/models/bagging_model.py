from typing import Any, Dict
import numpy as np
import pandas as pd
import pickle
from sklearn.utils import resample

from .base_model import BaseModel
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


class BaggingRegressor(BaseModel):
    """
    Bagging ensemble regressor that uses a combination of XGBoost and RandomForest models.

    This regressor creates an ensemble of XGBoost and RandomForest models by training them
    on different random subsets of the data (with replacement), then averages their predictions
    to produce the final prediction.

    Attributes:
    -----------
    xgb_params : dict
        Parameters for the XGBoost regressor models.
    rf_params : dict
        Parameters for the RandomForest regressor models.
    n_estimators : int
        The number of base estimators in the ensemble.
    bagging_fraction : float
        The fraction of the training data to be used for training each base estimator.
    xgb_models : list of XGBRegressor
        The list of trained XGBoost regressor models.
    rf_models : list of RandomForestRegressor
        The list of trained RandomForest regressor models.

    Methods:
    --------
    train(X_train, y_train) -> None:
        Trains the ensemble of models on the given dataset.

    predict(X) -> pd.Series:
        Predicts regression targets for given input features.

    save(path) -> None:
        Saves the trained ensemble model to the specified path.

    load(path) -> None:
        Loads the ensemble model from the specified path.
    """

    def __init__(self, params: Dict[str, Dict[str, Any]]):
        """
        Initializes the BaggingRegressor with a set of parameters for XGBoost and RandomForest models,
        as well as bagging-specific parameters.

        The 'params' dictionary is expected to contain sub-dictionaries for 'xgboost' and 'random_forest'
        keys with their respective model parameters, and bagging parameters under 'n_estimators' and
        'bagging_fraction'.

        Parameters:
        -----------
        params : dict
            A nested dictionary where:
            - The 'xgboost' key contains parameters for XGBoost models.
            - The 'random_forest' key contains parameters for RandomForest models.
            - The 'n_estimators' key specifies the number of base estimators in the ensemble.
            - The 'bagging_fraction' key determines the fraction of the dataset to use for training each base estimator.

        Raises:
        -------
        ValueError
            If the required keys are missing in the 'params' dictionary or if 'bagging_fraction' is not in the range (0, 1].

        Examples:
        --------
        >>> params ={
                'xgboost': {
                    'n_estimators': 100,
                    'max_depth': 3,
                    'learning_rate': 0.1,
                    #... other XGBoost parameters
                },
                'random_forest': {
                    'n_estimators': 50,
                    'max_depth': 5,
                    #... other RandomForest parameters
                },
                'n_estimators': 10,  # Total number of base estimators in the bagging ensemble
                'bagging_fraction': 0.8  # Percentage of data to use for each base model (0.8 means 80%)
            }
        """
        # Validation for 'params' structure and values
        if 'xgboost' not in params or 'random_forest' not in params:
            raise ValueError(
                "The 'params' dictionary must contain 'xgboost' and 'random_forest' keys."
            )
        if 'n_estimators' not in params or not isinstance(params['n_estimators'], int):
            raise ValueError("'n_estimators' must be provided as an integer value.")
        if 'bagging_fraction' not in params or not (
            0 < params['bagging_fraction'] <= 1
        ):
            raise ValueError("'bagging_fraction' must be a float in the range (0, 1].")

        self.xgb_params = params['xgboost']
        self.rf_params = params['random_forest']
        self.n_estimators = params['n_estimators']
        self.bagging_fraction = params['bagging_fraction']
        self.xgb_models = []
        self.rf_models = []

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train the bagging ensemble of XGBoost and RandomForest models.

        Each model is trained on a random subset of the provided training data.

        Parameters:
        -----------
        X_train : pd.DataFrame
            The input features for training.
        y_train : pd.Series
            The target values for training.
        """
        n_samples = int(len(X_train) * self.bagging_fraction)

        # Train XGBoost models
        for _ in range(self.n_estimators):
            X_sample, y_sample = resample(X_train, y_train, n_samples=n_samples)
            xgb_model = XGBRegressor(**self.xgb_params)
            xgb_model.fit(X_sample, y_sample)
            self.xgb_models.append(xgb_model)

        # Train RandomForest models
        for _ in range(self.n_estimators):
            X_sample, y_sample = resample(X_train, y_train, n_samples=n_samples)
            rf_model = RandomForestRegressor(**self.rf_params)
            rf_model.fit(X_sample, y_sample)
            self.rf_models.append(rf_model)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict regression targets using the ensemble of XGBoost and RandomForest models.

        The predictions from all models are averaged to produce the final output.

        Parameters:
        -----------
        X : pd.DataFrame
            The input features for which to predict the target values.

        Returns:
        --------
        pd.Series
            The predicted target values.
        """
        xgb_predictions = np.mean(
            [model.predict(X) for model in self.xgb_models], axis=0
        )
        rf_predictions = np.mean([model.predict(X) for model in self.rf_models], axis=0)
        return pd.Series((xgb_predictions + rf_predictions) / 2)

    def save(self, path: str) -> None:
        """
        Save the trained ensemble model to a file.

        Parameters:
        -----------
        path : str
            The file path where the model will be saved.
        """
        with open(path, 'wb') as f:
            pickle.dump((self.xgb_models, self.rf_models), f)

    def load(self, path: str) -> None:
        """
        Load the ensemble model from a file.

        Parameters:
        -----------
        path : str
            The file path from which to load the model.
        """
        with open(path, 'rb') as f:
            self.xgb_models, self.rf_models = pickle.load(f)
