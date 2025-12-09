"""
Data loading and preprocessing utilities
"""
import pandas as pd
from sklearn.model_selection import train_test_split


class DataLoader:
    """Handles loading and splitting data"""

    def __init__(self, train_path, test_path=None):
        self.train_path = train_path
        self.test_path = test_path

    def load_train_data(self):
        """Load training data from CSV"""
        return pd.read_csv(self.train_path)

    def load_test_data(self):
        """Load test data from CSV"""
        if self.test_path:
            return pd.read_csv(self.test_path)
        return None

    def split_data(self, X, y, test_size=0.2, val_size=0.2, random_state=42):
        """
        Split data into train, validation, and test sets

        Args:
            X: Features
            y: Target variable
            test_size: Proportion for test set
            val_size: Proportion of remaining data for validation
            random_state: Random seed for reproducibility

        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=random_state
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_statistics(self, df, target_col='price'):
        """Print basic statistics about the data"""
        stats = {
            'mean': df[target_col].mean(),
            'median': df[target_col].median(),
            'shape': df.shape,
            'missing_values': df.isnull().sum().sum()
        }
        return stats

    def remove_outliers_iqr(self, df, columns=None, lower_factor=1.5, upper_factor=1.5):
        """
        Remove outliers using IQR method

        Args:
            df: DataFrame
            columns: List of columns to check for outliers (default: all numeric)
            lower_factor: Multiplier for lower bound (Q1 - factor*IQR)
            upper_factor: Multiplier for upper bound (Q3 + factor*IQR)

        Returns:
            DataFrame without outliers
        """
        df_clean = df.copy()

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        print(f"Original shape: {df_clean.shape}")

        for col in columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - lower_factor * IQR
            upper_bound = Q3 + upper_factor * IQR

            # Count outliers before removing
            outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]

            # Remove outliers
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]

            print(f"  {col}: Removed {len(outliers)} outliers ({len(outliers)/len(df)*100:.2f}%)")

        print(f"Final shape: {df_clean.shape}")
        print(f"Total removed: {len(df) - len(df_clean)} rows ({(len(df) - len(df_clean))/len(df)*100:.2f}%)")

        return df_clean


    def remove_outliers_percentile(self, df, column, lower_percentile=0, upper_percentile=99):
        """
        Remove outliers based on percentiles

        Args:
            df: DataFrame
            column: Column to filter
            lower_percentile: Lower percentile threshold (0-100)
            upper_percentile: Upper percentile threshold (0-100)

        Returns:
            DataFrame without outliers
        """
        df_clean = df.copy()

        lower_bound = df_clean[column].quantile(lower_percentile / 100)
        upper_bound = df_clean[column].quantile(upper_percentile / 100)

        print(f"Original shape: {df_clean.shape}")
        print(f"Removing {column} outside [{lower_bound:.2f}, {upper_bound:.2f}]")

        df_clean = df_clean[(df_clean[column] >= lower_bound) & (df_clean[column] <= upper_bound)]

        print(f"Final shape: {df_clean.shape}")
        print(f"Total removed: {len(df) - len(df_clean)} rows ({(len(df) - len(df_clean))/len(df)*100:.2f}%)")

        return df_clean


    def remove_outliers_upper_only(self, df, column, upper_percentile=95):
        """
        Remove only upper outliers (common for price data)

        Args:
            df: DataFrame
            column: Column to filter
            upper_percentile: Upper percentile threshold (0-100)

        Returns:
            DataFrame without upper outliers
        """
        df_clean = df.copy()

        upper_bound = df_clean[column].quantile(upper_percentile / 100)

        print(f"Original shape: {df_clean.shape}")
        print(f"Removing {column} > {upper_bound:.2f} (top {100-upper_percentile}%)")

        outliers_removed = len(df_clean[df_clean[column] > upper_bound])
        df_clean = df_clean[df_clean[column] <= upper_bound]

        print(f"Final shape: {df_clean.shape}")
        print(f"Total removed: {outliers_removed} rows ({outliers_removed/len(df)*100:.2f}%)")

        return df_clean
