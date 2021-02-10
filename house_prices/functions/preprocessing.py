# modify this file and submit PR

import json
import warnings

import pandas as pd
from pandas.api.types import is_numeric_dtype


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
                    f"{strategy} strategy is not suitable " + f"for non numerical data in {col}"
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
