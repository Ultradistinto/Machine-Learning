"""
Data transformation utilities for feature engineering
"""
import numpy as np
import pandas as pd


class TransformationItem:
    """Wrapper for applying and reversing transformations"""

    def __init__(self, transform_fn, untransform_fn, columns=None):
        self.transform_fn = transform_fn
        self.untransform_fn = untransform_fn
        self.columns = columns

    def transform(self, data):
        if isinstance(data, pd.DataFrame):
            data = data.copy()
            cols_to_transform = (
                self.columns if self.columns is not None
                else data.select_dtypes(include=np.number).columns
            )
            for col in cols_to_transform:
                if col in data.columns:
                    data[col] = self.transform_fn(data[col].to_numpy())
            return data

        elif isinstance(data, (np.ndarray, pd.Series, list)):
            arr = np.asarray(data)
            return self.transform_fn(arr)
        else:
            raise TypeError(f"Unsupported data type for transform: {type(data)}")

    def untransform(self, data):
        if isinstance(data, pd.DataFrame):
            data = data.copy()
            cols_to_untransform = (
                self.columns if self.columns is not None
                else data.select_dtypes(include=np.number).columns
            )
            for col in cols_to_untransform:
                if col in data.columns:
                    data[col] = self.untransform_fn(data[col].to_numpy())
            return data

        elif isinstance(data, (np.ndarray, pd.Series, list)):
            arr = np.asarray(data)
            return self.untransform_fn(arr)
        else:
            raise TypeError(f"Unsupported data type for untransform: {type(data)}")


class TransformationArray:
    """Pipeline of transformations that can be applied and reversed"""

    def __init__(self, items):
        self.items = items

    def transform(self, df):
        for item in self.items:
            df = item.transform(df)
        return df

    def untransform(self, df):
        for item in reversed(self.items):
            df = item.untransform(df)
        return df
