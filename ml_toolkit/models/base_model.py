import pandas as pd
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    Abstract base class for machine learning models.

    This class provides a general interface for training, prediction, saving, and loading
    machine learning models. Any specific model type (e.g., linear regression, decision tree,
    neural network) that inherits from this class should provide concrete implementations
    for the `train`, `predict`, `save` and `load` methods, as they are marked as abstract
    and are required to be implemented. The `validate` method is optional but should be
    implemented if relevant for the specific model type.

    Methods:
    --------
    train(X_train, y_train) -> None:
        Train the model on the provided dataset.

    predict(X) -> pd.Series:
        Predict the target variable based on the input features.

    save(path) -> None:
        Save the model to the specified path.

    load(path) -> None:
        Load a trained model from the specified path.

    validate(X_test, y_test) -> dict:
        Validate the model's performance on test data and return metrics.
        (Optional, but will be useful for model evaluation)

    """

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train the model on given data.

        Parameters:
        -----------
        X_train : pd.DataFrame
            Features for training the model.
        y_train : pd.Series
            Target variable for training.

        Raises:
        -------
        NotImplementedError
            If the child class does not implement this method.
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict the target values for given features.

        Parameters:
        -----------
        X : pd.DataFrame
            Features for making predictions.

        Raises:
        -------
        NotImplementedError
            If the child class does not implement this method.
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the trained model to the specified path.

        Parameters:
        -----------
        path : str
            Path to save the trained model.

        Raises:
        -------
        NotImplementedError
            If the child class does not implement this method.
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load a trained model from the specified path.

        Parameters:
        -----------
        path : str
            Path to load the trained model from.

        Raises:
        -------
        NotImplementedError
            If the child class does not implement this method.
        """
        pass

    def validate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Validate the model's performance on test data.

        Parameters:
        -----------
        X_test : pd.DataFrame
            Features for testing the model.
        y_test : pd.Series
            True target values for testing.

        Returns:
        --------
        dict
            Dictionary containing various performance metrics.

        Raises:
        -------
        NotImplementedError
            If the child class does not implement this method.
        """
        raise NotImplementedError
