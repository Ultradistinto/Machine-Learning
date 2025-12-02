"""
Model training and evaluation
"""
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class ModelTrainer:
    """Handles model training and evaluation"""

    def __init__(self, transform=None):
        self.transform = transform
        self.scaler = StandardScaler()
        self.models = {}

    def scale_features(self, X_train, X_val, X_test):
        """Apply standard scaling to features"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_val_scaled, X_test_scaled

    def train_linear_regression(self, X_train, y_train, scaled=False):
        """Train a linear regression model"""
        model = LinearRegression()
        model.fit(X_train, y_train)
        model_name = 'linear_regression_scaled' if scaled else 'linear_regression'
        self.models[model_name] = model
        return model

    def train_ridge(self, X_train, y_train, alpha=1.0):
        """Train a Ridge regression model"""
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        self.models['ridge'] = model
        return model

    def train_lasso(self, X_train, y_train, alpha=10.0, max_iter=10000):
        """Train a Lasso regression model"""
        model = Lasso(alpha=alpha, max_iter=max_iter)
        model.fit(X_train, y_train)
        self.models['lasso'] = model

        selected_features = np.sum(model.coef_ != 0)
        print(f"Features selected by LASSO: {selected_features}/{X_train.shape[1]}")

        return model

    def evaluate_model(self, model, X_sets, y_sets, set_names):
        """
        Evaluate model on multiple datasets

        Args:
            model: Trained model
            X_sets: List of feature sets [X_train, X_val, X_test]
            y_sets: List of target sets [y_train, y_val, y_test]
            set_names: List of names ['Train', 'Validation', 'Test']

        Returns:
            Dictionary with metrics for each set
        """
        results = {}

        for name, X, y in zip(set_names, X_sets, y_sets):
            y_pred = model.predict(X)

            if self.transform:
                y_pred_original = self.transform.untransform(y_pred)
                y_original = self.transform.untransform(y)
            else:
                y_pred_original = y_pred
                y_original = y

            mae = mean_absolute_error(y_original, y_pred_original)
            mse = mean_squared_error(y_original, y_pred_original)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_original, y_pred_original)

            results[name] = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2
            }

            print(f"\n{name}:")
            print(f"  MAE:  ${mae:,.2f}")
            print(f"  RMSE: ${rmse:,.2f}")
            print(f"  R²:   {r2:.6f}")

        return results

    def predict_test_set(self, model, X_test, test_ids, output_path):
        """Generate predictions for test set and save to CSV"""
        y_pred = model.predict(X_test)

        if self.transform:
            y_pred_original = self.transform.untransform(y_pred)
        else:
            y_pred_original = y_pred

        import pandas as pd
        predictions_df = pd.DataFrame({
            'id': test_ids,
            'price': y_pred_original
        })

        predictions_df.to_csv(output_path, index=False)
        print(f"\nPredictions saved to '{output_path}'")

        return predictions_df
