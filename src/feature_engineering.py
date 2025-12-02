"""
Feature engineering functions for Airbnb price prediction
"""
import numpy as np
import pandas as pd


class FeatureEngineer:
    """Handles all feature engineering operations"""

    def __init__(self, config):
        self.config = config
        self.bbaa_center = config.get('bbaa_center', [-34.59977951146896, -58.38320368379193])

    def add_distance_to_center(self, df):
        """Calculate distance from listing to Buenos Aires center"""
        df['distance_to_center'] = np.sqrt(
            (df['latitude'] - self.bbaa_center[0])**2 +
            (df['longitude'] - self.bbaa_center[1])**2
        )
        return df

    def add_time_features(self, df):
        """Add time-based features from last_review date"""
        time_diff = pd.to_datetime('today') - pd.to_datetime(
            df['last_review'].fillna('01-01-1970'), dayfirst=True
        )

        if self.config.get('days_since_last_review', True):
            df['days_since_last_review'] = time_diff / np.timedelta64(1, 'D')

        if self.config.get('weeks_since_last_review', True):
            df['weeks_since_last_review'] = time_diff / np.timedelta64(1, 'W')

        if self.config.get('months_since_last_review', True):
            df['months_since_last_review'] = (time_diff / np.timedelta64(1, 'D')) / 30.4375

        if self.config.get('quarters_since_last_review', True):
            df['quarters_since_last_review'] = (time_diff / np.timedelta64(1, 'D')) / 91.3125

        if self.config.get('years_since_last_review', True):
            df['years_since_last_review'] = (time_diff / np.timedelta64(1, 'D')) / 365.25

        return df

    def process_last_review_date(self, df):
        """Convert last_review to numeric or drop it"""
        if self.config.get('last_review_date', True):
            df['last_review'] = pd.to_datetime(
                df['last_review'].fillna('01-01-1970'),
                dayfirst=True
            ).astype('int64')
        else:
            df = df.drop(['last_review'], axis=1)
        return df

    def add_minimum_nights_category(self, df):
        """Categorize minimum nights into ranges"""
        df['minimum_nights_category'] = pd.cut(
            df['minimum_nights'],
            bins=[0, 7, 30, 180, 365, 10000],
            labels=['short', 'week', 'month', 'semi_year', 'long']
        )
        return df

    def add_host_multiple_listings(self, df):
        """Flag if host has multiple listings"""
        df['host_has_multiple_listings'] = df['calculated_host_listings_count'] > 1
        return df

    def add_reviews_ratio(self, df):
        """Calculate reviews per day ratio"""
        df['reviews_ratio'] = df['number_of_reviews'] / (df['days_since_last_review'] + 1)
        return df

    def drop_unnecessary_columns(self, df):
        """Remove columns not needed for modeling"""
        columns_to_drop = ['name', 'id', 'host_id', 'host_name', 'availability_365']
        existing_cols = [col for col in columns_to_drop if col in df.columns]
        return df.drop(existing_cols, axis=1)

    def fill_missing_values(self, df):
        """Fill missing values in reviews_per_month"""
        df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
        return df

    def encode_categorical(self, df):
        """Encode categorical variables"""
        night_mapping = {
            'short': 1,
            'week': 2,
            'month': 3,
            'semi_year': 4,
            'long': 5
        }

        if 'minimum_nights_category' in df.columns:
            df['minimum_nights_num'] = df['minimum_nights_category'].map(night_mapping)
            df = df.drop(['minimum_nights_category'], axis=1)

        categorical_cols = ['room_type', 'neighbourhood']
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        return df_encoded

    def apply_all_features(self, df, is_training=True):
        """Apply all feature engineering steps"""
        if self.config.get('distance_to_center', True):
            df = self.add_distance_to_center(df)

        df = self.add_time_features(df)
        df = self.process_last_review_date(df)

        if self.config.get('minimum_nights_category', True):
            df = self.add_minimum_nights_category(df)

        if self.config.get('host_multiple_listing', True):
            df = self.add_host_multiple_listings(df)

        if self.config.get('review_ratio', True):
            df = self.add_reviews_ratio(df)

        df = self.fill_missing_values(df)
        df = self.drop_unnecessary_columns(df)
        df = self.encode_categorical(df)

        return df
