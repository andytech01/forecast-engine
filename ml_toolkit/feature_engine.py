from typing import Tuple, List
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import warnings

warnings.filterwarnings("ignore")  # Ignore warnings


class FeatureGenerator:
    """
    FeatureGenerator is responsible for generating features from the data for a given date range.

    Attributes:
    -----------
    df : DataFrame
        DataFrame from which features will be generated.

    Methods:
    --------
    generate_offline_features_for_date(date, columns) -> DataFrame:
        Extracts and combines features for a specific date.
    @staticmethod
    generate_diff_features(df, features, steps) -> DataFrame:
        Computes differences for given features over specified steps.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initializes FeatureExtractor with data from a DataLoader instance.

        Parameters:
        -----------
        df : DataFrame
            DataFrame from which features will be generated.
        """
        self.df = df

    def _extract_data_for_period(
        self, date_col: str, date: pd.Timestamp, period: str
    ) -> pd.DataFrame:
        """
        Extracts data for a specific period relative to a given date.

        Parameters:
        -----------
        date_col : str
            The name of the date column in the dataframe.
        date : Timestamp
            The reference date.
        period : str
            The period relative to the date ('yesterday', 'last_week_same_workday', 'last_week').

        Returns:
        --------
        DataFrame
            A DataFrame containing data for the specified period.
        """
        if period == "yesterday":
            return self.df[
                self.df[date_col] == date - pd.Timedelta(days=1)
            ].reset_index(drop=True)
        elif period == "last_week_same_workday":
            return self.df[
                self.df[date_col] == date - pd.Timedelta(days=7)
            ].reset_index(drop=True)
        elif period == "last_week":
            return self.df[
                (self.df[date_col] >= date - pd.Timedelta(days=7))
                & (self.df[date_col] <= date - pd.Timedelta(days=1))
            ].reset_index(drop=True)

    def _compute_aggregates(
        self,
        data: pd.DataFrame,
        column: str,
        period: str,
        group_keys: List[str] = ["zone_id", "10min_segment"],
    ) -> pd.DataFrame:
        """
        Computes aggregate metrics for a given column over a data period.

        Parameters:
        -----------
        data : DataFrame
            The data on which to compute aggregates.
        column : str
            The column name for which to compute aggregates.
        period : str
            The period during which data was extracted.
        primary_keys : list, optional
            The group keys based on which the data is grouped and aggregated.
            Default is ["zone_id", "10min_segment"].

        Returns:
        --------
        DataFrame
            A DataFrame with aggregated features.
        """
        features_df = (
            data.groupby(group_keys)[column]
            .agg(
                avg=lambda x: np.round(x.mean(), 2),
                median=lambda x: np.round(x.median(), 2),
                q_25=lambda x: np.round(x.quantile(0.25), 2),
                q_75=lambda x: np.round(x.quantile(0.75), 2),
                q_90=lambda x: np.round(x.quantile(0.9), 2),
            )
            .reset_index()
        )
        features_df.columns = group_keys + [
            f"{period}_{column}_{col}"
            for col in ["avg", "median", "q_25", "q_75", "q_90"]
        ]
        return features_df

    def generate_offline_features_for_date(
        self,
        date_col: str,
        date: pd.Timestamp,
        columns: List[str],
        join_keys: List[str] = ["zone_id", "10min_segment"],
        periods: List[str] = ["yesterday", "last_week_same_workday", "last_week"],
    ):
        """
        Extracts and combines features for a given date.

        Parameters:
        -----------
        date_col : str
            The name of the date column in the dataframe.
        date : Timestamp
            The date for which to extract features.
        columns : list
            List of columns for feature extraction.
        join_keys : list, optional
            Columns to join on during feature generation. Default is ["zone_id", "10min_segment"].
        periods : list, optional
            List of periods for feature generation. Default is ["yesterday", "last_week_same_workday", "last_week"].

        Returns:
        --------
        DataFrame
            A DataFrame containing generated features for the specified date.
        """
        periods = ["yesterday", "last_week_same_workday", "last_week"]
        features_df = None

        for column in columns:
            for period in periods:
                data = self._extract_data_for_period(date_col, date, period)
                df = self._compute_aggregates(data, column, period, join_keys)
                if features_df is None:
                    features_df = df
                else:
                    features_df = pd.merge(features_df, df, on=join_keys, how="left")

        features_df.insert(0, date_col, date)
        return features_df

    @staticmethod
    def generate_diff_features(df: pd.DataFrame, features: List[str], steps: int):
        """
        Calculate differences for specified features.

        Parameters:
        -----------
        df : pandas.DataFrame
            The dataframe to be processed.
        features : list
            List of features to calculate differences.
        steps : int
            Number of difference steps.

        Returns:
        --------
        pandas.DataFrame
            Processed dataframe with new difference features.
        """
        for col in features:
            for i in range(1, steps + 1):
                df[f"{col}_diff_{i}"] = df[col].diff(i)

        return df


class FeatureEncoder:
    """
    Handles the encoding tasks for datasets, transforming categorical variables
    into formats suitable for Machine Learning models.

    Methods:
    --------
    @staticmethod
    onehot_encode(df, columns, save_path=None) -> Tuple[DataFrame, List[str]]:
        One-hot encodes specified columns in the dataframe.

    @staticmethod
    label_encode(df, column, save_path=None) -> Tuple[DataFrame, str]:
        Label encodes a specified column in the dataframe.

    Note:
    -----
    Both methods are designed as static methods. They can be called on the class itself
    without requiring an instance of the class to be created.
    """

    @staticmethod
    def onehot_encode(
        df: pd.DataFrame,
        columns: List[str],
        save_path: str = None,
        is_train: bool = True,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        One-hot encodes specified columns in the dataframe.

        Parameters:
        -----------
        df : DataFrame
            The dataframe to be encoded.
        columns : list
            Columns to be one-hot encoded.
        save_path : str, optional
            Path to save the OneHotEncoder instance.
        is_train : bool, optional
            Flag to indicate if the encoder is being used for training or inference.
            Default is True.

        Returns:
        --------
        tuple:
            - DataFrame: DataFrame with one-hot encoded columns added.
            - list: List of names for the one-hot encoded columns.

        This method transforms specified columns in the dataframe into one-hot encoded format.
        If a save_path is provided, the OneHotEncoder instance will be saved to the specified path.
        The method also returns the names of the one-hot encoded columns.
        """
        if is_train:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
            encoded_data = encoder.fit_transform(df[columns])
            onehot_features = encoder.get_feature_names_out(columns).tolist()
            onehot_df = pd.DataFrame(
                encoded_data, columns=onehot_features, index=df.index
            )

            if save_path:
                with open(save_path, "wb") as f:
                    pickle.dump(encoder, f)
        else:
            with open(save_path, "rb") as f:
                encoder = pickle.load(f)
            encoded_data = encoder.transform(df[columns])
            onehot_features = encoder.get_feature_names_out(columns).tolist()
            onehot_df = pd.DataFrame(
                encoded_data, columns=onehot_features, index=df.index
            )

        return pd.concat([df, onehot_df], axis=1), onehot_features

    @staticmethod
    def label_encode(
        df: pd.DataFrame, column: str, save_path: str = None, is_train: bool = True
    ) -> Tuple[pd.DataFrame, str]:
        """
        Label encodes a specified column in the dataframe.

        Parameters:
        -----------
        df : DataFrame
            The dataframe to be encoded.
        column : str
            The column to be label encoded.
        save_path : str, optional
            Path to save the LabelEncoder instance.
        is_train : bool, optional
            Flag to indicate if the encoder is being used for training or inference.
            Default is True.

        Returns:
        --------
        tuple:
            - DataFrame: DataFrame with the label encoded column added.
            - str: Name of the newly added label encoded column.

        This method transforms a specified column in the dataframe into a label encoded format.
        The original column remains unchanged, and a new column with the suffix '_label' is added
        with encoded values. If a save_path is provided, the LabelEncoder instance will be saved
        to the specified path.
        """
        if is_train:
            encoder = LabelEncoder()
            df[f"{column}_label"] = encoder.fit_transform(df[column])

            if save_path:
                with open(save_path, "wb") as f:
                    pickle.dump(encoder, f)
        else:
            with open(save_path, "rb") as f:
                encoder = pickle.load(f)
            df[f"{column}_label"] = encoder.transform(df[column])

        return df, f"{column}_label"
