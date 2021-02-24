# modify this file and submit PR
import json
import warnings
from joblib import dump, load

import pandas as pd
import phik
from pandas.api.types import is_numeric_dtype
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import OneClassSVM


class OptionsNotSetUpError(Exception):
    pass


class NotFittedError(Exception):
    pass


class Imputer:
    """Reimplementation of SimpleImputer class for DataFrames.

    This class allows to impute empty values in any pandas DataFrame. It is inspired by sklearn's
    SimpleImputer class which operates on numpy arrays. It support various imputing strategies:
        - mean: use mean column value (available only for numerical columns)
        - median: use column median (available only for numerical columns)
        - most_frequent: use most frequent value
        - constant: use user provided value

    Before calling .fit method please provide options dictionary. Each key in options dictionary
    should correspond to single column name in imputed DataFrame. Each value should specify imputing
    strategy (also as a dictionary).

    Keys for imputing strategy dictionary are:
        strategy (str):
            One of available imputing strategies.
        missing_values (any single Python variable or list of variables; optional)
            Which values should be marked as missing. Default is float.nan.
        fill_value (float; optional):
            User provided fill value for missing values. Only appropriate when strategy is constant.
        add_indicator (bool; optional):
            Whether to create separate column indicating missing values. Default is False.

    Instance methods:
        fit(df):
            Compute imputed values and store them internally.
        transform(df, copy=True):
            Impute missing values in df.
        fit_transform(df, copy=True):
            Compute and impute missing values.
        save_options(filename):
            Save options dictionary in JSON format.

    Class methods:
        from_options_file(filename):
            Alternative constructor to recreate fitted instance from saved JSON file.
    """

    ALLOWED_OPTIONS_KWARGS = {"missing_values", "strategy", "fill_value", "add_indicator", "_fill"}
    ALLOWED_STRATEGIES = {"mean", "median", "most_frequent", "constant"}

    def __init__(self, options=None):
        self.options = options
        self._fitted = (
            False
            if options is None
            else all("_fill" in kwargs for kwargs in self._options.values())
        )

    @classmethod
    def from_options_file(cls, filename):
        with open(filename, "r") as f:
            deser = json.loads(f.read())
        return cls(options=deser)

    @property
    def options(self):
        return self._options

    @property
    def fitted(self):
        return self._fitted

    @options.setter
    def options(self, value):
        if not isinstance(value, dict):
            raise TypeError("options should be a dict")
        for k, v in value.items():
            if not isinstance(v, dict):
                raise TypeError("options values should be a dict")
            if not set(v.keys()).issubset(self.ALLOWED_OPTIONS_KWARGS):
                raise ValueError(
                    f"incorrect key in {v}, allowed keys: " + f"{self.ALLOWED_OPTIONS_KWARGS}"
                )
        self._options = value

    @classmethod
    def validate_options_kwargs(cls, kwargs):
        try:
            strategy = kwargs["strategy"]
        except KeyError:
            raise KeyError("strategy should be specified for each column")
        if strategy not in cls.ALLOWED_STRATEGIES:
            raise ValueError(
                f"not recognized strategy {strategy}" + f"allowed options: {cls.ALLOWED_STRATEGIES}"
            )

        fill_value = kwargs.get("fill_value", None)
        if fill_value is not None and strategy is not "constant":
            warnings.warn("specified fill value but strategy is not constant")

        add_indicator = kwargs.get("add_indicator", False)
        if not isinstance(add_indicator, bool):
            raise TypeError(
                "add_indicator should be either True or False" + f" but is {type(add_indicator)}"
            )

        missing_values = kwargs.get("missing_values", float("nan"))
        if not isinstance(missing_values, list):
            missing_values = [missing_values]

        return strategy, fill_value, add_indicator, missing_values

    def fit(self, df):
        if self.options is None:
            raise OptionsNotSetUpError("set up options before fitting")

        for col, kwargs in self.options.items():
            strategy, fill_value, add_indicator, missing_values = self.validate_options_kwargs(
                kwargs
            )

            if strategy in ("mean", "median") and not is_numeric_dtype(df[col]):
                raise TypeError(
                    f"{strategy} strategy is not suitable for non numerical data in {col}"
                )

            if strategy == "constant":
                fill = fill_value
            else:
                nan_indicator = df[col].isin(missing_values)
                if strategy == "mean":
                    fill = df.loc[~nan_indicator, col].mean()
                if strategy == "median":
                    fill = df.loc[~nan_indicator, col].median()
                if strategy == "most_frequent":
                    fill = df.loc[~nan_indicator, col].mode()[0]

            kwargs["add_indicator"] = add_indicator
            kwargs["missing_values"] = missing_values
            kwargs["_fill"] = fill

        self._fitted = True

    def transform(self, df, copy=True):
        if not isinstance(copy, bool):
            raise TypeError("copy should be either True or False")
        if copy:
            df = df.copy(deep=True)
        for col, kwargs in self._options.items():
            if col not in df:
                continue
            nan_indicator = df[col].isin(kwargs["missing_values"])
            df.loc[nan_indicator, col] = kwargs["_fill"]
            if kwargs["add_indicator"]:
                df[col + "_missing"] = nan_indicator.astype("int")
        return df

    def fit_transform(self, df, copy=True):
        self.fit(df)
        return self.transform(df, copy)

    def save_options(self, filename):
        if self.options is None:
            raise OptionsNotSetUpError("set up options before saving")
        with open(filename, "w") as f:
            f.write(json.dumps(self.options, indent=2))


def removeOutliers(data: pd.DataFrame, method: str, treshold: float, model_kwargs):
    """
    Function try to find outliers in given data set using one of available methods: DBSSCAN, OneClassSVN, IsolationForest.
    Next step is reduce number of outliers to given treshold. Where treshold is percentage of population.
    Value 0.05 means that function will return data set without 5% of population which were marked as outlier.
    If algorithm detected number of outliers less than treshold then funtion remove all detected outliers from population.

    Args:
        data (Pd.DataFrame): Data set which should contains only features.
        threshold (float): Value between 0 and 1.
        method (str): Method used to calculate corelation between columns. Available methods: DBSSCAN, SVN, IsolationForest
    Returns:
        Data Frame without observation marked as outliers.

    """
    dataCopy = data.copy()
    DBS_kwargs = IF_kwargs = SVN_kwargs = {}

    if method == "DBSSCAN":
        DBS_kwargs = model_kwargs
    if method == "SVN":
        SVN_kwargs = model_kwargs
    if method == "IsolationForest":
        IF_kwargs = model_kwargs

    methods = {
        "DBSSCAN": DBSCAN(**DBS_kwargs),
        "SVN": OneClassSVM(**SVN_kwargs),
        "IsolationForest": IsolationForest(**IF_kwargs),
    }

    model = methods[method]
    print(f"Model to detect outliers is {method} with parameters {model_kwargs}")
    outliers = model.fit_predict(dataCopy)

    lowerBound = np.floor(treshold * len(dataCopy))

    print(f"Detected {np.sum(outliers==-1)} outliers")
    print(f"Removing {int(min(lowerBound, np.sum(outliers==-1))) } outliers from oryginal data set")

    toRemove = dataCopy.loc[outliers == -1, :].index
    toRemove = toRemove[: int(min(lowerBound, len(toRemove)))]
    return dataCopy.drop(toRemove)


def removeHighlyCorreletedFeatures(data: pd.DataFrame, threshold: float, method: str):
    """
    Function calculates correlation between features and removes one of these features which are highly correleted.
    Warning: if two features are highly correleted then function take first feature from list data.columns.
    Highly correleted features are these which absolute value of correlation is bigger than treshold.
    Args:
        data (Pd.DataFrame): Data set which should contains only features.
        threshold (float): Value between 0 and 1.
        method (str): Method used to calculate corelation between columns. Available methods: {‘pearson’, ‘kendall’, ‘spearman’} or callable
    Returns:
        Data Frame with not correleted features between each other.

    """
    unCorrelatedFeatures = list()
    dataCopy = data.copy()
    originalColumns = dataCopy.columns

    if method in ["pearson", "kendall", "spearman"]:
        corr_data = dataCopy.corr(method=method)
    elif method == "phik":
        if interval_cols:
            corr_data = dataCopy.phik_matrix(interval_cols=interval_cols)
        else:
            corr_data = dataCopy.phik_matrix(interval_cols=data_copy.columns)
    else:
        raise KeyError("Provide a valid method name.")

    corr_data = corr_data[np.abs(corr_data) < threshold]
    unCorrelatedFeatures.append(corr_data.columns[0])
    columnsToCheck = corr_data[corr_data.columns[0]].dropna().index
    for column in columnsToCheck:
        previousColumns = corr_data.loc[corr_data.index.isin(unCorrelatedFeatures), column]
        if np.sum(previousColumns < threshold) == len(unCorrelatedFeatures):
            unCorrelatedFeatures.append(column)

    print(
        f"Columns {list(set(originalColumns) & set(unCorrelatedFeatures))} low correlated between each other"
    )

    return dataCopy[unCorrelatedFeatures]


def remove_uncorrelated_with_target(
    data,
    target_column: str,
    threshold: float = 0.3,
    method: str = "pearson",
    remove_anticorrelated: bool = True,
    interval_cols: list = None,
    verbose: bool = False,
):
    """
    This function removes features that are weakly correlated to the target column.

    Args:
        data (DataFrame): data to remove the uncorrelated columns from,

        target_column (str): target column to compare the other features with,

        threshold (float): correlation threshold,

        method (str): 'pearson', 'kendall', 'spearman', 'phik':
            - 'pearson' - standard correlation coefficient
            - 'kendall' - Kendall Tau correlation coefficient
            - 'spearman' - Spearman rank correlation
            - 'phik' - described at https://arxiv.org/abs/1811.11440)

        remove_anticorrelated (bool): determines if features that are anti-correlated
            above the given threshold should be removed from the dataframe (True, default)
            or should be left in the data (False)

        interval_cols (list) - column names of columns with interval variables, only relevant
            in the 'phik' method, the default behaviour is to assume that all columns
            are numerical

        verbose (bool) - this mode if turned on (True) provides details about the execution of the code

    Returns:
        data_copy (DataFrame): data after removing the weakly correlated features

        removed_columns (list): names of removed columns

    """

    data_copy = data.copy()

    if method in ["pearson", "kendall", "spearman"]:
        corr = data_copy.corr(method=method)
    elif method == "phik":
        if interval_cols:
            corr = data_copy.phik_matrix(interval_cols=interval_cols)
        else:
            corr = data_copy.phik_matrix(interval_cols=data_copy.columns)
    else:
        raise KeyError("Provide a valid method name.")

    # Show features with highest and lowest correlation with the target column
    corr_target = corr[target_column]
    corr_target = corr_target.sort_values(ascending=False)
    if verbose:
        print(
            f"Features with highest correlation with the target columns are: \n{corr_target.head(5)}"
        )
        print(
            f"Features with lowest correlation with the target columns are: \n{corr_target.tail(5)}"
        )

    # Remove features weakly correlating with target
    features_to_remove = corr_target[corr_target < threshold].index
    if remove_anticorrelated:
        features_to_keep = corr_target[abs(corr_target) > threshold].index
    else:
        features_to_keep = corr_target[corr_target > threshold].index

    data_copy = data_copy[features_to_keep]
    if verbose:
        print(
            f"({len(features_to_remove)} features correlated with target column ({target_column})",
            f"weaker than {threshold} have been removed",
            f"and the data now has the shape {data_copy.shape}.",
        )

    return data_copy, list(features_to_remove)


def remove_low_variance_features(data, threshold: float = 0.05):
    """
    This function removes features with low variance.

    Arguments:

    data (DataFrame): data to remove the low-variance features from
    threshold (float): minimum variance that the feature needs to have in order to stay in the dataset

    Returns:

    data_high_variance (DataFrame): data with the low-variant features removed
    removed_cols (list): removed columns

    """

    data_copy = data.copy()
    raw_variances = data_copy.var()
    means = data_copy.mean()
    # an attempt to normalize the variance: var/(mean**2) should be lower than the threshold
    cols_to_drop = [
        column
        for column in data.columns
        if raw_variances[column] / (means[column] ** 2) < threshold
    ]
    data_high_variance = data_copy.drop(columns=cols_to_drop)

    return data_high_variance, cols_to_drop

class Encode_categorical:
    """ Encode categorical features
    
    This class allows to encode binary columns. Before use method fit, provide column_names list
    
    Args for class:
        column_names (list): List of categorical columns to encode from dataframe
        method_drop (str): Optional, type of drop method: 'first' or 'binary'
    
    Instance methods:
    fit(df):
        Create columns for binary filling.
    transform(df, copy=True):
        Impute binary value to columns.
    fit_transform(df, copy=True):
        Equivalent to fit(df, copy=True).transform(df, copy=True) but more convenient.
    save(filename, name_of_instance):
        Save model fitted in joblib extension.
    from_file(filename):
        Read model fitted from file in joblib extension.
        
    """
    
    def __init__(self, column_names, method_drop='first'):
        self.column_names = column_names
        self.method_drop = method_drop
        
    def fit(self, df, copy=True):
        if copy:
            df = df.copy(deep=True)
        self.ohe = OneHotEncoder(drop=self.method_drop, 
                                 sparse=False).fit(df[self.column_names])
    
    def transform(self, df, copy=True):
        if copy:
            df = df.copy(deep=True)
        data_cat_transformed = pd.DataFrame(self.ohe.transform(df[self.column_names]), 
                                        columns = self.ohe.get_feature_names(
                                            input_features=self.column_names),
                                        index = df.index)
        df = pd.concat([df.drop(self.column_names, axis = 1), 
                               data_cat_transformed], axis = 1)
        return df

    def fit_transform(self, df):
        self.fit(df,copy=True)
        return self.transform(df, copy=True)
    
    def save(self, filename, name_of_instance):
        if not hasattr(name_of_instance, 'ohe'):
            raise TypeError("Fit model before saving")
        dump(self.ohe, filename) 
        
    def from_file(self, filename):
        if not os.path.isfile(filename):
            raise TypeError("Save model before reading")
        self.ohe = load(filename) 
