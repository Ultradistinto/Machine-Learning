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
