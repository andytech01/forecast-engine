import importlib
from typing import Dict, Any
from .models import BaseModel


def get_model_instance(model_name: str, parameters: Dict[str, Any]) -> BaseModel:
    """
    Dynamically import and instantiate a model from the `ml_toolkit.models` module based on the given model_name.

    Parameters
    ----------
    model_name : str
        The name of the model class to be instantiated, e.g., 'XGBoostRegressor'.
    parameters : dict
        A dictionary containing the parameters to be passed to the model's constructor.

    Returns
    -------
    model_instance : BaseModel
        An instance of the specified model class, initialized with the provided parameters. The returned instance
        is a subclass of `BaseModel`.

    Raises
    ------
    ImportError
        If the `ml_toolkit.models` module does not exist or if there's an issue importing it.
    AttributeError
        If the specified model is not found within the `ml_toolkit.models` module or if the model is not a
        subclass of `BaseModel`.

    Examples
    --------
    >>> parameters = {
            "max_depth": 8,
            "learning_rate": 0.05,
            ...
        }
    >>> model = get_model_instance("XGBoostRegressor", parameters)
    """

    try:
        module = importlib.import_module(".models", package="ml_toolkit")
        model_class = getattr(module, model_name)

        if not issubclass(model_class, BaseModel):
            raise AttributeError(
                f"Model '{model_name}' is not a subclass of 'BaseModel'."
            )

        return model_class(parameters)
    except ImportError:
        raise ImportError(f"Module 'ml_toolkit.models' does not exist.")
    except AttributeError:
        raise AttributeError(
            f"Model '{model_name}' is not found in 'ml_toolkit.models' or is not a subclass of 'BaseModel'."
        )
