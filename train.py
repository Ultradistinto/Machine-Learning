"""
Main training script for Airbnb price prediction
"""
import sys
import yaml
import numpy as np
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

    trainer = ModelTrainer(transform=transform)

    print("\n" + "="*70)
    print("TRAINING MODELS")
    print("="*70)

    # 1. Linear Regression (No Scaling)
    print("\n1. Linear Regression (No Scaling)")
    print("-" * 50)
    model_lr_unscaled = trainer.train_linear_regression(X_train, y_train, scaled=False)
    results_lr_unscaled = trainer.evaluate_model(
        model_lr_unscaled,
        [X_train, X_val, X_test],
        [y_train, y_val, y_test],
        ['Train', 'Validation', 'Test']
    )

    # 2. Linear Regression (Scaled)
    print("\n2. Linear Regression (Scaled)")
    print("-" * 50)
    X_train_scaled, X_val_scaled, X_test_scaled = trainer.scale_features(
        X_train, X_val, X_test
    )
    df_test_scaled = trainer.scaler.transform(df_test)

    model_lr = trainer.train_linear_regression(X_train_scaled, y_train, scaled=True)
    results_lr = trainer.evaluate_model(
        model_lr,
        [X_train_scaled, X_val_scaled, X_test_scaled],
        [y_train, y_val, y_test],
        ['Train', 'Validation', 'Test']
    )

    print("\n2. Ridge Regression")
    print("-" * 50)
    ridge_alpha = config['models']['ridge']['alpha']
    model_ridge = trainer.train_ridge(X_train_scaled, y_train, alpha=ridge_alpha)
    results_ridge = trainer.evaluate_model(
        model_ridge,
        [X_train_scaled, X_val_scaled, X_test_scaled],
        [y_train, y_val, y_test],
        ['Train', 'Validation', 'Test']
    )

    print("\n3. Lasso Regression")
    print("-" * 50)
    lasso_alpha = config['models']['lasso']['alpha']
    lasso_max_iter = config['models']['lasso']['max_iter']
    model_lasso = trainer.train_lasso(
        X_train_scaled, y_train,
        alpha=lasso_alpha,
        max_iter=lasso_max_iter
    )
    results_lasso = trainer.evaluate_model(
        model_lasso,
        [X_train_scaled, X_val_scaled, X_test_scaled],
        [y_train, y_val, y_test],
        ['Train', 'Validation', 'Test']
    )

    Path(config['output']['predictions_dir']).mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("GENERATING PREDICTIONS")
    print("="*70)

    # Linear Regression (Unscaled) - uses unscaled test data
    predictions_path_lr_unscaled = f"{config['output']['predictions_dir']}/predictions_linear_regression_unscaled.csv"
    trainer.predict_test_set(model_lr_unscaled, df_test, test_ids, predictions_path_lr_unscaled)

    # Linear Regression (Scaled) - uses scaled test data
    predictions_path_lr_scaled = f"{config['output']['predictions_dir']}/predictions_linear_regression_scaled.csv"
    trainer.predict_test_set(model_lr, df_test_scaled, test_ids, predictions_path_lr_scaled)

    # Ridge Regression - uses scaled test data
    predictions_path_ridge = f"{config['output']['predictions_dir']}/predictions_ridge.csv"
    trainer.predict_test_set(model_ridge, df_test_scaled, test_ids, predictions_path_ridge)

    # Lasso Regression - uses scaled test data
    predictions_path_lasso = f"{config['output']['predictions_dir']}/predictions_lasso.csv"
    trainer.predict_test_set(model_lasso, df_test_scaled, test_ids, predictions_path_lasso)

    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print("="*70)


if __name__ == "__main__":
    main()
