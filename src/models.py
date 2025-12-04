"""
Model training and evaluation
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV, RepeatedKFold, KFold


class ModelTrainer:
    """Handles model training and evaluation"""

    def __init__(self, transform=None, cv_config=None):
        self.transform = transform
        self.scaler = StandardScaler()
        self.models = {}
        self.best_params = {}
        self.cv_results = {}
        self.cv_config = cv_config or {}

    def _get_cv_splitter(self):
        """Get cross-validation splitter based on config"""
        method = self.cv_config.get('method', 'kfold')
        n_splits = self.cv_config.get('n_splits', 5)
        random_state = self.cv_config.get('random_state', 42)
        
        if method == 'repeated_holdout':
            n_repeats = self.cv_config.get('n_repeats', 3)
            return RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        else:  # kfold
            return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

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

    def train_decision_tree(self, X_train, y_train, config=None, use_grid_search=False):
        """Train a Decision Tree Regressor"""
        config = config or {}
        
        if use_grid_search and 'grid_search' in config:
            print("  Running GridSearchCV for Decision Tree...")
            base_model = DecisionTreeRegressor(random_state=42)
            param_grid = config['grid_search']
            
            grid_search = GridSearchCV(
                base_model, param_grid,
                cv=self._get_cv_splitter(),
                scoring=self.cv_config.get('scoring', 'neg_root_mean_squared_error'),
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            self.best_params['decision_tree'] = grid_search.best_params_
            self.cv_results['decision_tree'] = {
                'best_score': -grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
            print(f"  Best params: {grid_search.best_params_}")
            print(f"  Best CV RMSE: {-grid_search.best_score_:.2f}")
            
            model = grid_search.best_estimator_
        else:
            model = DecisionTreeRegressor(
                max_depth=config.get('max_depth', 10),
                min_samples_split=config.get('min_samples_split', 5),
                min_samples_leaf=config.get('min_samples_leaf', 2),
                random_state=42
            )
            model.fit(X_train, y_train)
        
        self.models['decision_tree'] = model
        return model

    def train_random_forest(self, X_train, y_train, config=None, use_grid_search=False):
        """Train a Random Forest Regressor"""
        config = config or {}
        
        if use_grid_search and 'grid_search' in config:
            print("  Running GridSearchCV for Random Forest...")
            base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
            param_grid = config['grid_search']
            
            grid_search = GridSearchCV(
                base_model, param_grid,
                cv=self._get_cv_splitter(),
                scoring=self.cv_config.get('scoring', 'neg_root_mean_squared_error'),
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            self.best_params['random_forest'] = grid_search.best_params_
            self.cv_results['random_forest'] = {
                'best_score': -grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
            print(f"  Best params: {grid_search.best_params_}")
            print(f"  Best CV RMSE: {-grid_search.best_score_:.2f}")
            
            model = grid_search.best_estimator_
        else:
            model = RandomForestRegressor(
                n_estimators=config.get('n_estimators', 100),
                max_depth=config.get('max_depth', 10),
                min_samples_split=config.get('min_samples_split', 2),
                min_samples_leaf=config.get('min_samples_leaf', 1),
                n_jobs=config.get('n_jobs', -1),
                random_state=42
            )
            model.fit(X_train, y_train)
        
        self.models['random_forest'] = model
        return model

    def train_gradient_boosting(self, X_train, y_train, config=None, use_grid_search=False):
        """Train a Gradient Boosting Regressor"""
        config = config or {}
        
        if use_grid_search and 'grid_search' in config:
            print("  Running GridSearchCV for Gradient Boosting...")
            base_model = GradientBoostingRegressor(random_state=42)
            param_grid = config['grid_search']
            
            grid_search = GridSearchCV(
                base_model, param_grid,
                cv=self._get_cv_splitter(),
                scoring=self.cv_config.get('scoring', 'neg_root_mean_squared_error'),
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            self.best_params['gradient_boosting'] = grid_search.best_params_
            self.cv_results['gradient_boosting'] = {
                'best_score': -grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
            print(f"  Best params: {grid_search.best_params_}")
            print(f"  Best CV RMSE: {-grid_search.best_score_:.2f}")
            
            model = grid_search.best_estimator_
        else:
            model = GradientBoostingRegressor(
                n_estimators=config.get('n_estimators', 100),
                learning_rate=config.get('learning_rate', 0.1),
                max_depth=config.get('max_depth', 5),
                min_samples_split=config.get('min_samples_split', 2),
                min_samples_leaf=config.get('min_samples_leaf', 1),
                random_state=42
            )
            model.fit(X_train, y_train)
        
        self.models['gradient_boosting'] = model
        return model

    def train_neural_network(self, X_train, y_train, config=None, use_grid_search=False):
        """Train a Neural Network (MLP) Regressor"""
        config = config or {}
        
        if use_grid_search and 'grid_search' in config:
            print("  Running GridSearchCV for Neural Network...")
            base_model = MLPRegressor(
                max_iter=config.get('epochs', 100),
                early_stopping=config.get('early_stopping', True),
                validation_fraction=0.1,
                random_state=42
            )
            
            # Convert hidden_layers to hidden_layer_sizes format
            param_grid = {}
            if 'hidden_layers' in config['grid_search']:
                param_grid['hidden_layer_sizes'] = [tuple(h) for h in config['grid_search']['hidden_layers']]
            if 'learning_rate' in config['grid_search']:
                param_grid['learning_rate_init'] = config['grid_search']['learning_rate']
            
            grid_search = GridSearchCV(
                base_model, param_grid,
                cv=self._get_cv_splitter(),
                scoring=self.cv_config.get('scoring', 'neg_root_mean_squared_error'),
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            self.best_params['neural_network'] = grid_search.best_params_
            self.cv_results['neural_network'] = {
                'best_score': -grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
            print(f"  Best params: {grid_search.best_params_}")
            print(f"  Best CV RMSE: {-grid_search.best_score_:.2f}")
            
            model = grid_search.best_estimator_
        else:
            hidden_layers = config.get('hidden_layers', [128, 64])
            model = MLPRegressor(
                hidden_layer_sizes=tuple(hidden_layers),
                learning_rate_init=config.get('learning_rate', 0.001),
                max_iter=config.get('epochs', 100),
                batch_size=config.get('batch_size', 32),
                early_stopping=config.get('early_stopping', True),
                validation_fraction=0.1,
                n_iter_no_change=config.get('patience', 10),
                random_state=42
            )
            model.fit(X_train, y_train)
        
        self.models['neural_network'] = model
        return model

    def cross_validate_model(self, model, X, y, model_name="model"):
        """Perform cross-validation on a model and return scores"""
        cv = self._get_cv_splitter()
        scoring = self.cv_config.get('scoring', 'neg_root_mean_squared_error')
        
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        
        # Convert negative scores to positive (for RMSE)
        if 'neg_' in scoring:
            scores = -scores
        
        print(f"\n  Cross-Validation Results for {model_name}:")
        print(f"    Scores: {scores}")
        print(f"    Mean: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")
        
        return {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std()
        }

    def evaluate_model(self, model, X_sets, y_sets, set_names, verbose=True):
        """
        Evaluate model on multiple datasets

        Args:
            model: Trained model
            X_sets: List of feature sets [X_train, X_val, X_test]
            y_sets: List of target sets [y_train, y_val, y_test]
            set_names: List of names ['Train', 'Validation', 'Test']
            verbose: Whether to print results

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
                'MSE': mse,
                'RMSE': rmse,
                'R2': r2
            }

            if verbose:
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

        predictions_df = pd.DataFrame({
            'id': test_ids,
            'price': y_pred_original
        })

        predictions_df.to_csv(output_path, index=False)
        print(f"\nPredictions saved to '{output_path}'")

        return predictions_df

    def get_results_summary(self):
        """Get a summary of all model results"""
        summary = []
        for model_name, results in self.cv_results.items():
            summary.append({
                'Model': model_name,
                'Best CV RMSE': results.get('best_score', None),
                'Best Params': self.best_params.get(model_name, {})
            })
        return pd.DataFrame(summary)
