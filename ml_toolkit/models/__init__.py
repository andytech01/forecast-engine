"""
ml_toolkit.models
-----------------
This module provides various machine learning models for the `ml_toolkit` package.

Usage and Conventions:
----------------------
- All model classes within this module should inherit from the `BaseModel` class defined in `base_model.py`.
- This inheritance ensures that all models adhere to a consistent interface and structure, making it easier to switch between models or build customized models.
- To add a new model to this module, please ensure the following:
    1. Your model class should be defined in a separate file within the `models` folder.
    2. Your model class should inherit from `BaseModel`.
    3. Import your model class in `models/__init__.py` file to make it available for external imports.
    4. Update the documentation as necessary to reflect any specific behaviors or requirements of your model.

Note:
-----
Failing to adhere to the convention of inheriting from `BaseModel` might result in unexpected behaviors or errors when integrating with other parts of the `ml_toolkit` package.

"""

from .base_model import BaseModel
from .xgboost_model import XGBoostRegressor
from .bagging_model import BaggingRegressor
from .moving_average_model import MARegressor

# # Dynamically import all the models in the current package
# import pkgutil
# for _, module_name, _ in pkgutil.walk_packages(__path__):
#     # Exclude the private classes
#     if not module_name.startswith('_'):
#         __import__(module_name, locals(), globals())

# # Remove unnecessary items from the namespace
# del pkgutil, module_name
