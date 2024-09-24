from typing import Any, Dict, List
import pandas as pd
import pickle
from .base_model import BaseModel
import xgboost as xgb


class XGBoostRegressor(BaseModel):
    def __init__(self, params: Dict[str, Any]):
        super().__init__()
        self._model = xgb.XGBRegressor(**params)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self._model.fit(X_train, y_train)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return self._model.predict(X)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self._model, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            self._model = pickle.load(f)

    def get_features(self) -> List[str]:
        return self._model.get_booster().feature_names
