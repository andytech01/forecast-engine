import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")  # Ignore warnings


class DataProcessor:
    """
    Processes time-series data, offering functionality like NaN value handling, Gaussian smoothing,
    and time segments creation.

    Attributes
    ----------
    data : pandas.DataFrame
        The dataframe containing the data to be processed.

    Methods
    -------
    fill_na(column_name, value=0)
        Fill NaN values in a specified column with a given value.
    gaussian_smooth(column_name, new_column_name, window_size=11, sigma=2.5)
        Apply Gaussian smoothing to a column.
    create_minute_segments(hour_column, minute_column, new_column_name, segment_size=10)
        Segment time based on given segment_size.
    get_processed_data()
        Get the processed dataframe.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'time': pd.date_range(start='1/1/2023', periods=3, freq='H'),
    ...                    'value': [1, np.nan, 3]})
    >>> processor = DataProcessor(df)
    >>> processed_df = processor.fill_na('value').gaussian_smooth('value', 'smoothed_value').get_processed_data()
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the DataProcessor with a DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            A pandas DataFrame with time-series data to process.
        """
        self.data = df.copy()

    def fill_na(self, column_name: str, value: float = 0) -> "DataProcessor":
        """
        Fill NaN values in a specified column with the provided value.

        Parameters
        ----------
        column_name : str
            The name of the column in which to fill NaN values.
        value : int or float, optional
            The value with which to fill NaNs. Defaults to 0.

        Returns
        -------
        DataProcessor
            The instance itself to enable method chaining.

        Examples
        --------
        >>> processor.fill_na(col_name, 0)
        """
        self.data[column_name].fillna(value, inplace=True)
        return self

    def _gaussian_weights(self, window: int, sigma: float) -> np.ndarray:
        """
        Calculate Gaussian weights for a smoothing window.

        Parameters
        ----------
        window : int
            The size of the smoothing window.
        sigma : float
            The standard deviation of the Gaussian kernel.

        Returns
        -------
        numpy.ndarray
            Array of weights for the Gaussian smoothing window.
        """
        distance_from_center = np.arange(-window // 2 + 1, window // 2 + 1)
        weights = np.exp(-(distance_from_center**2) / (2 * sigma**2))
        return weights / sum(weights)

    def gaussian_smooth(
        self,
        column_name: str,
        new_column_name: str,
        window_size: int = 11,
        sigma: float = 2.5,
    ) -> "DataProcessor":
        """
        Apply Gaussian smoothing to a column and save the result in a new column.

        Parameters
        ----------
        column_name : str
            The name of the column to smooth.
        new_column_name : str
            The name of the new column where the smoothed data will be saved.
        window_size : int, optional
            The size of the Gaussian smoothing window. Defaults to 11.
        sigma : float, optional
            The standard deviation for the Gaussian kernel. Defaults to 2.5.

        Returns
        -------
        DataProcessor
            The instance itself to enable method chaining.

        Examples
        --------
        >>> processor.gaussian_smooth(col_name, smoothed_col_name)
        """
        weights = self._gaussian_weights(window_size, sigma)
        smoothed_series = self.data[column_name].copy()
        half_window = window_size // 2

        for i in range(len(self.data)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(self.data), i + half_window + 1)

            window_values = self.data.iloc[start_idx:end_idx][column_name]
            valid_values = window_values.dropna()

            valid_weights = weights[
                half_window - (i - start_idx) : half_window + (end_idx - i)
            ]
            valid_weights = valid_weights[~window_values.isna()]
            valid_weights /= valid_weights.sum()

            smoothed_series.iloc[i] = np.sum(valid_values * valid_weights).round(2)

        self.data[new_column_name] = smoothed_series
        return self

    def create_minute_segments(
        self,
        hour_column: str,
        minute_column: str,
        new_column_name: str,
        segment_size: int = 10,
    ) -> "DataProcessor":
        """
        Create time segments based on hour and minute columns.

        Parameters
        ----------
        hour_column : str
            The name of the hour column.
        minute_column : str
            The name of the minute column.
        new_column_name : str
            The name of the new column to store segment data.
        segment_size : int, optional
            The size of each time segment in minutes. Defaults to 10.

        Returns
        -------
        DataProcessor
            The instance itself to enable method chaining.
        """
        self.data[new_column_name] = (
            self.data[hour_column] * 60 + self.data[minute_column]
        ) // segment_size
        return self

    def get_processed_data(self) -> pd.DataFrame:
        """
        Get the processed dataframe.

        Returns
        -------
        pandas.DataFrame
            The processed dataframe.
        """
        return self.data
