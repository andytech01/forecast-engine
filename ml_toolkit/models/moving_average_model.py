import pandas as pd


class MARegressor:
    def __init__(self, window_size: int = 3):
        super().__init__()
        self.window_size = window_size

    def predict(self, y_history: pd.Series, forecast_len: int) -> pd.Series:
        return y_history.rolling(window=self.window_size).mean().iloc[-forecast_len:]
