"""
Main training script for Airbnb price prediction
"""
import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.transformations import TransformationItem, TransformationArray
from src.models import ModelTrainer


def load_config(config_path='configs/config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_transformation_pipeline(config):
    """Create transformation pipeline based on config"""
    columns = config['transformation']['columns_to_transform']
    transform = TransformationArray([
        TransformationItem(
            lambda df: np.log1p(df),
            lambda df: np.expm1(df),
            columns=columns
        )
    ])
    return transform


def main():
    print("="*70)
    print("AIRBNB PRICE PREDICTION - TRAINING")
    print("="*70)

    config = load_config()
    cv_config = config.get('cross_validation', {})
    use_grid_search = cv_config.get('grid_search', False)

    data_loader = DataLoader(
        train_path=config['data']['train_path'],
        test_path=config['data']['test_path']
    )

    print("\nLoading training data...")
    df = data_loader.load_train_data()

    stats = data_loader.get_statistics(df)
    print(f"\nOriginal Data Statistics:")
    print(f"  Mean:   ${stats['mean']:,.2f}")
    print(f"  Median: ${stats['median']:,.2f}")
    print(f"  Shape:  {stats['shape']}")

    df = data_loader.remove_outliers_upper_only(df, column='price', upper_percentile=99)

    print("\nApplying feature engineering...")
    feature_engineer = FeatureEngineer(config['features'])
    df = feature_engineer.apply_all_features(df)

    print("\nLoading test data...")
    df_test = data_loader.load_test_data()
    test_ids = df_test['id'].copy()
    df_test = feature_engineer.apply_all_features(df_test, is_training=False)

    print("\nApplying transformations...")
    transform = create_transformation_pipeline(config)
    df = transform.transform(df)
    df_test = transform.transform(df_test)

    print("\nSplitting data...")
    y = df['price']
    X = df.drop('price', axis=1)

    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.split_data(
        X, y,
        test_size=config['split']['test_size'],
        val_size=config['split']['val_size'],
        random_state=config['split']['random_state']
    )

    print(f"\n{'='*70}")
    print("DATA SPLIT")
    print(f"{'='*70}")
    print(f"Train:      {X_train.shape[0]:,} samples")
    print(f"Validation: {X_val.shape[0]:,} samples")
    print(f"Test:       {X_test.shape[0]:,} samples")
    print(f"Features:   {X_train.shape[1]}")

    # Initialize trainer with CV config
    trainer = ModelTrainer(transform=transform, cv_config=cv_config)

    # Store all results for summary table
    all_results = {}

    print("\n" + "="*70)
    print("TRAINING MODELS")
    print("="*70)

    # Scale features for models that need it
    X_train_scaled, X_val_scaled, X_test_scaled = trainer.scale_features(
        X_train, X_val, X_test
    )
    df_test_scaled = trainer.scaler.transform(df_test)

    # ==================== LINEAR MODELS ====================
    
    # 1. Linear Regression (No Scaling)
    print("\n1. Linear Regression (No Scaling)")
    print("-" * 50)
    model_lr_unscaled = trainer.train_linear_regression(X_train, y_train, scaled=False)
    all_results['Linear Regression (Unscaled)'] = trainer.evaluate_model(
        model_lr_unscaled,
        [X_train, X_val, X_test],
        [y_train, y_val, y_test],
        ['Train', 'Validation', 'Test']
    )

    # 2. Linear Regression (Scaled)
    print("\n2. Linear Regression (Scaled)")
    print("-" * 50)
    model_lr = trainer.train_linear_regression(X_train_scaled, y_train, scaled=True)
    all_results['Linear Regression (Scaled)'] = trainer.evaluate_model(
        model_lr,
        [X_train_scaled, X_val_scaled, X_test_scaled],
        [y_train, y_val, y_test],
        ['Train', 'Validation', 'Test']
    )

    # 3. Ridge Regression
    print("\n3. Ridge Regression")
    print("-" * 50)
    ridge_alpha = config['models']['ridge']['alpha']
    model_ridge = trainer.train_ridge(X_train_scaled, y_train, alpha=ridge_alpha)
    all_results['Ridge Regression'] = trainer.evaluate_model(
        model_ridge,
        [X_train_scaled, X_val_scaled, X_test_scaled],
        [y_train, y_val, y_test],
        ['Train', 'Validation', 'Test']
    )

    # 4. Lasso Regression
    print("\n4. Lasso Regression")
    print("-" * 50)
    lasso_alpha = config['models']['lasso']['alpha']
    lasso_max_iter = config['models']['lasso']['max_iter']
    model_lasso = trainer.train_lasso(
        X_train_scaled, y_train,
        alpha=lasso_alpha,
        max_iter=lasso_max_iter
    )
    all_results['Lasso Regression'] = trainer.evaluate_model(
        model_lasso,
        [X_train_scaled, X_val_scaled, X_test_scaled],
        [y_train, y_val, y_test],
        ['Train', 'Validation', 'Test']
    )

    # ==================== TREE-BASED MODELS ====================
    
    # 5. Decision Tree
    print("\n5. Decision Tree Regressor")
    print("-" * 50)
    model_dt = trainer.train_decision_tree(
        X_train_scaled, y_train,
        config=config['models']['decision_tree'],
        use_grid_search=use_grid_search
    )
    all_results['Decision Tree'] = trainer.evaluate_model(
        model_dt,
        [X_train_scaled, X_val_scaled, X_test_scaled],
        [y_train, y_val, y_test],
        ['Train', 'Validation', 'Test']
    )

    # 6. Random Forest
    print("\n6. Random Forest Regressor")
    print("-" * 50)
    model_rf = trainer.train_random_forest(
        X_train_scaled, y_train,
        config=config['models']['random_forest'],
        use_grid_search=use_grid_search
    )
    all_results['Random Forest'] = trainer.evaluate_model(
        model_rf,
        [X_train_scaled, X_val_scaled, X_test_scaled],
        [y_train, y_val, y_test],
        ['Train', 'Validation', 'Test']
    )

    # 7. Gradient Boosting
    print("\n7. Gradient Boosting Regressor")
    print("-" * 50)
    model_gb = trainer.train_gradient_boosting(
        X_train_scaled, y_train,
        config=config['models']['gradient_boosting'],
        use_grid_search=use_grid_search
    )
    all_results['Gradient Boosting'] = trainer.evaluate_model(
        model_gb,
        [X_train_scaled, X_val_scaled, X_test_scaled],
        [y_train, y_val, y_test],
        ['Train', 'Validation', 'Test']
    )

    # ==================== NEURAL NETWORK ====================
    
    # 8. Neural Network
    print("\n8. Neural Network (MLP)")
    print("-" * 50)
    model_nn = trainer.train_neural_network(
        X_train_scaled, y_train,
        config=config['models']['neural_network'],
        use_grid_search=use_grid_search
    )
    all_results['Neural Network'] = trainer.evaluate_model(
        model_nn,
        [X_train_scaled, X_val_scaled, X_test_scaled],
        [y_train, y_val, y_test],
        ['Train', 'Validation', 'Test']
    )

    # ==================== RESULTS SUMMARY ====================
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    # Create summary table
    summary_data = []
    for model_name, results in all_results.items():
        summary_data.append({
            'Model': model_name,
            'Train MAE': f"${results['Train']['MAE']:,.2f}",
            'Train RMSE': f"${results['Train']['RMSE']:,.2f}",
            'Train R²': f"{results['Train']['R2']:.4f}",
            'Val MAE': f"${results['Validation']['MAE']:,.2f}",
            'Val RMSE': f"${results['Validation']['RMSE']:,.2f}",
            'Val R²': f"{results['Validation']['R2']:.4f}",
            'Test MAE': f"${results['Test']['MAE']:,.2f}",
            'Test RMSE': f"${results['Test']['RMSE']:,.2f}",
            'Test R²': f"{results['Test']['R2']:.4f}",
        })
    
    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))
    
    # Save summary to CSV
    summary_df.to_csv(f"{config['output']['predictions_dir']}/model_comparison.csv", index=False)
    print(f"\nSummary saved to '{config['output']['predictions_dir']}/model_comparison.csv'")

    # Print best params if grid search was used
    if use_grid_search and trainer.best_params:
        print("\n" + "="*70)
        print("BEST HYPERPARAMETERS (from GridSearchCV)")
        print("="*70)
        for model_name, params in trainer.best_params.items():
            print(f"\n{model_name}:")
            for param, value in params.items():
                print(f"  {param}: {value}")

    # ==================== GENERATE PREDICTIONS ====================
    
    Path(config['output']['predictions_dir']).mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("GENERATING PREDICTIONS")
    print("="*70)

    # Linear Regression (Unscaled)
    trainer.predict_test_set(model_lr_unscaled, df_test, test_ids, 
                            f"{config['output']['predictions_dir']}/predictions_linear_regression_unscaled.csv")

    # Linear Regression (Scaled)
    trainer.predict_test_set(model_lr, df_test_scaled, test_ids, 
                            f"{config['output']['predictions_dir']}/predictions_linear_regression_scaled.csv")

    # Ridge Regression
    trainer.predict_test_set(model_ridge, df_test_scaled, test_ids, 
                            f"{config['output']['predictions_dir']}/predictions_ridge.csv")

    # Lasso Regression
    trainer.predict_test_set(model_lasso, df_test_scaled, test_ids, 
                            f"{config['output']['predictions_dir']}/predictions_lasso.csv")

    # Decision Tree
    trainer.predict_test_set(model_dt, df_test_scaled, test_ids, 
                            f"{config['output']['predictions_dir']}/predictions_decision_tree.csv")

    # Random Forest
    trainer.predict_test_set(model_rf, df_test_scaled, test_ids, 
                            f"{config['output']['predictions_dir']}/predictions_random_forest.csv")

    # Gradient Boosting
    trainer.predict_test_set(model_gb, df_test_scaled, test_ids, 
                            f"{config['output']['predictions_dir']}/predictions_gradient_boosting.csv")

    # Neural Network
    trainer.predict_test_set(model_nn, df_test_scaled, test_ids, 
                            f"{config['output']['predictions_dir']}/predictions_neural_network.csv")

    # ==================== BEST MODEL SUMMARY ====================
    
    # Find best model based on Test R²
    best_model_name = None
    best_r2 = float('-inf')
    best_metrics = None
    
    for model_name, results in all_results.items():
        if results['Test']['R2'] > best_r2:
            best_r2 = results['Test']['R2']
            best_model_name = model_name
            best_metrics = results
    
    print("\n" + "="*70)
    print("BEST MODEL FOUND")
    print("="*70)
    print(f"\n  Model: {best_model_name}")
    print(f"\n  Performance Metrics:")
    print(f"  {'-'*50}")
    print(f"    Test R²:   {best_metrics['Test']['R2']:.6f}")
    print(f"    Test MAE:  ${best_metrics['Test']['MAE']:,.2f}")
    print(f"    Test RMSE: ${best_metrics['Test']['RMSE']:,.2f}")
    print(f"    Val R²:    {best_metrics['Validation']['R2']:.6f}")
    print(f"    Val MAE:   ${best_metrics['Validation']['MAE']:,.2f}")
    
    # Map model names to their config keys for hyperparameters
    model_to_config_key = {
        'Linear Regression (Unscaled)': None,
        'Linear Regression (Scaled)': None,
        'Ridge Regression': 'ridge',
        'Lasso Regression': 'lasso',
        'Decision Tree': 'decision_tree',
        'Random Forest': 'random_forest',
        'Gradient Boosting': 'gradient_boosting',
        'Neural Network': 'neural_network',
    }
    
    config_key = model_to_config_key.get(best_model_name)
    
    # Print hyperparameters
    print(f"\n  Hyperparameters:")
    print(f"  {'-'*50}")
    
    if config_key and config_key in trainer.best_params:
        # Grid search was used and found best params
        print(f"    (from GridSearchCV)")
        for param, value in trainer.best_params[config_key].items():
            print(f"    {param}: {value}")
    elif config_key and config_key in config['models']:
        # Print config defaults
        model_config = config['models'][config_key]
        print(f"    (from config defaults)")
        for param, value in model_config.items():
            if param != 'grid_search' and param != 'use_scaling':
                print(f"    {param}: {value}")
    else:
        print(f"    No hyperparameters (linear regression)")
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETED")
    print("="*70)


if __name__ == "__main__":
    main()
