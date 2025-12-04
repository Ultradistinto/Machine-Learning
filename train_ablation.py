"""
Feature ablation study script for Airbnb price prediction.
Trains models with different combinations of feature groups to understand
which features contribute most to predictive performance.
"""
import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from datetime import datetime

from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.transformations import TransformationItem, TransformationArray
from src.models import ModelTrainer


def load_config(config_path='configs/config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_feature_columns_for_groups(df, feature_groups, selected_groups):
    """
    Get list of column names for the selected feature groups.
    Handles one-hot encoded columns by matching prefixes.
    """
    selected_columns = []
    
    for group_name in selected_groups:
        if group_name not in feature_groups:
            continue
            
        group_config = feature_groups[group_name]
        if not group_config.get('enabled', True):
            continue
            
        for col in group_config.get('columns', []):
            # Check for exact match
            if col in df.columns:
                selected_columns.append(col)
            # Check for one-hot encoded columns (prefix match)
            else:
                prefix_matches = [c for c in df.columns if c.startswith(f"{col}_")]
                selected_columns.extend(prefix_matches)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_columns = []
    for col in selected_columns:
        if col not in seen:
            seen.add(col)
            unique_columns.append(col)
    
    return unique_columns


def get_all_feature_columns(df, feature_groups):
    """Get all feature columns from all enabled groups."""
    all_groups = [name for name, cfg in feature_groups.items() if cfg.get('enabled', True)]
    return get_feature_columns_for_groups(df, feature_groups, all_groups)


def filter_features(X, feature_columns):
    """Filter dataframe to only include specified columns."""
    # Get columns that exist in X
    available_columns = [col for col in feature_columns if col in X.columns]
    return X[available_columns]


def generate_all_combinations(group_names, min_size=1):
    """Generate all possible combinations of feature groups."""
    all_combos = []
    for r in range(min_size, len(group_names) + 1):
        all_combos.extend(combinations(group_names, r))
    return [list(combo) for combo in all_combos]


def train_model_for_combination(trainer, model_name, X_train, X_val, X_test, 
                                y_train, y_val, y_test, config, use_grid_search=False):
    """Train a single model and return results."""
    
    if model_name == 'linear_regression':
        model = trainer.train_linear_regression(X_train, y_train, scaled=True)
    elif model_name == 'ridge':
        model = trainer.train_ridge(X_train, y_train, alpha=config['models']['ridge']['alpha'])
    elif model_name == 'lasso':
        model = trainer.train_lasso(X_train, y_train, 
                                    alpha=config['models']['lasso']['alpha'],
                                    max_iter=config['models']['lasso']['max_iter'])
    elif model_name == 'decision_tree':
        model = trainer.train_decision_tree(X_train, y_train, 
                                           config=config['models']['decision_tree'],
                                           use_grid_search=use_grid_search)
    elif model_name == 'random_forest':
        model = trainer.train_random_forest(X_train, y_train,
                                           config=config['models']['random_forest'],
                                           use_grid_search=use_grid_search)
    elif model_name == 'gradient_boosting':
        model = trainer.train_gradient_boosting(X_train, y_train,
                                               config=config['models']['gradient_boosting'],
                                               use_grid_search=use_grid_search)
    elif model_name == 'neural_network':
        model = trainer.train_neural_network(X_train, y_train,
                                            config=config['models']['neural_network'],
                                            use_grid_search=use_grid_search)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Evaluate
    results = trainer.evaluate_model(
        model,
        [X_train, X_val, X_test],
        [y_train, y_val, y_test],
        ['Train', 'Validation', 'Test'],
        verbose=False
    )
    
    return model, results


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


def run_ablation_study(config, combinations_to_test=None, models_to_test=None, 
                       use_grid_search=False, verbose=True):
    """
    Run ablation study with different feature combinations.
    
    Args:
        config: Configuration dictionary
        combinations_to_test: List of feature group combinations to test.
                             Each combination is a list of group names.
                             Use ["all"] to include all enabled groups.
        models_to_test: List of model names to test
        use_grid_search: Whether to use grid search for hyperparameter tuning
        verbose: Whether to print progress
    
    Returns:
        DataFrame with all results
    """
    
    feature_groups = config.get('feature_groups', {})
    ablation_config = config.get('ablation', {})
    
    # Use config values if not provided
    if combinations_to_test is None:
        combinations_to_test = ablation_config.get('combinations', [["all"]])
    
    if models_to_test is None:
        models_to_test = ablation_config.get('models', ['linear_regression', 'random_forest'])
    
    cv_config = config.get('cross_validation', {})
    
    # Load data
    data_loader = DataLoader(
        train_path=config['data']['train_path'],
        test_path=config['data']['test_path']
    )
    
    if verbose:
        print("="*70)
        print("FEATURE ABLATION STUDY")
        print("="*70)
        print(f"\nLoading training data...")
    
    df = data_loader.load_train_data()
    
    if verbose:
        print(f"Applying feature engineering...")
    
    feature_engineer = FeatureEngineer(config['features'])
    df = feature_engineer.apply_all_features(df)
    
    if verbose:
        print(f"Applying transformations...")
    
    transform = create_transformation_pipeline(config)
    df = transform.transform(df)
    
    # Split data
    y = df['price']
    X = df.drop('price', axis=1)
    
    X_train_full, X_val_full, X_test_full, y_train, y_val, y_test = data_loader.split_data(
        X, y,
        test_size=config['split']['test_size'],
        val_size=config['split']['val_size'],
        random_state=config['split']['random_state']
    )
    
    if verbose:
        print(f"\nData split: Train={len(X_train_full)}, Val={len(X_val_full)}, Test={len(X_test_full)}")
        print(f"Total features available: {X_train_full.shape[1]}")
    
    # Get all enabled group names
    enabled_groups = [name for name, cfg in feature_groups.items() if cfg.get('enabled', True)]
    
    if verbose:
        print(f"Feature groups: {enabled_groups}")
        print(f"Models to test: {models_to_test}")
        print(f"Combinations to test: {len(combinations_to_test)}")
    
    # Store all results
    all_results = []
    # Store feature columns and best params for each combination
    feature_columns_map = {}
    best_params_map = {}
    
    for combo_idx, combo in enumerate(combinations_to_test):
        # Handle "all" special case
        if combo == ["all"]:
            selected_groups = enabled_groups
            combo_name = "ALL"
        else:
            selected_groups = combo
            combo_name = "+".join(combo)
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"COMBINATION {combo_idx + 1}/{len(combinations_to_test)}: {combo_name}")
            print(f"{'='*70}")
        
        # Get columns for this combination
        feature_columns = get_feature_columns_for_groups(X_train_full, feature_groups, selected_groups)
        
        if not feature_columns:
            if verbose:
                print(f"  WARNING: No features found for combination {combo_name}, skipping...")
            continue
        
        # Store feature columns for this combination
        feature_columns_map[combo_name] = feature_columns
        
        # Filter features
        X_train = filter_features(X_train_full, feature_columns)
        X_val = filter_features(X_val_full, feature_columns)
        X_test = filter_features(X_test_full, feature_columns)
        
        if verbose:
            print(f"  Features selected: {len(feature_columns)}")
            print(f"  Columns: {feature_columns[:5]}..." if len(feature_columns) > 5 else f"  Columns: {feature_columns}")
        
        # Create trainer and scale features
        trainer = ModelTrainer(transform=transform, cv_config=cv_config)
        X_train_scaled, X_val_scaled, X_test_scaled = trainer.scale_features(X_train, X_val, X_test)
        
        # Train each model
        for model_name in models_to_test:
            if verbose:
                print(f"\n  Training {model_name}...")
            
            try:
                model, results = train_model_for_combination(
                    trainer, model_name,
                    X_train_scaled, X_val_scaled, X_test_scaled,
                    y_train, y_val, y_test,
                    config, use_grid_search
                )
                
                # Get best params if available
                best_params = trainer.best_params.get(model_name, {})
                
                # Store results
                result_row = {
                    'combination': combo_name,
                    'groups': str(selected_groups),
                    'n_features': len(feature_columns),
                    'model': model_name,
                    'train_mae': results['Train']['MAE'],
                    'train_rmse': results['Train']['RMSE'],
                    'train_r2': results['Train']['R2'],
                    'val_mae': results['Validation']['MAE'],
                    'val_rmse': results['Validation']['RMSE'],
                    'val_r2': results['Validation']['R2'],
                    'test_mae': results['Test']['MAE'],
                    'test_rmse': results['Test']['RMSE'],
                    'test_r2': results['Test']['R2'],
                    'best_params': str(best_params) if best_params else 'N/A',
                }
                all_results.append(result_row)
                
                # Store best params map
                key = f"{combo_name}_{model_name}"
                best_params_map[key] = best_params
                
                if verbose:
                    print(f"    Test R²: {results['Test']['R2']:.4f}, Test MAE: ${results['Test']['MAE']:,.2f}")
                    
            except Exception as e:
                if verbose:
                    print(f"    ERROR: {e}")
                continue
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    return results_df, feature_columns_map, best_params_map


def main():
    """Main function for running ablation study."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Feature ablation study for Airbnb price prediction')
    parser.add_argument('--config', type=str, default='configs/config.yaml', 
                        help='Path to config file')
    parser.add_argument('--all-combinations', action='store_true',
                        help='Test all possible combinations of feature groups')
    parser.add_argument('--groups', type=str, nargs='+',
                        help='Specific groups to test (space-separated)')
    parser.add_argument('--models', type=str, nargs='+',
                        help='Models to test (space-separated)')
    parser.add_argument('--grid-search', action='store_true',
                        help='Use grid search for hyperparameter tuning')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file path')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    feature_groups = config.get('feature_groups', {})
    enabled_groups = [name for name, cfg in feature_groups.items() if cfg.get('enabled', True)]
    
    # Determine combinations to test
    if args.all_combinations:
        combinations_to_test = generate_all_combinations(enabled_groups)
        combinations_to_test.insert(0, ["all"])  # Add all features first
    elif args.groups:
        combinations_to_test = [args.groups]
    else:
        combinations_to_test = config.get('ablation', {}).get('combinations', [["all"]])
    
    # All available models
    ALL_MODELS = [
        'linear_regression', 'ridge', 'lasso', 
        'decision_tree', 'random_forest', 'gradient_boosting', 'neural_network'
    ]
    
    # Determine models to test
    if args.models:
        if args.models == ['all'] or 'all' in args.models:
            models_to_test = ALL_MODELS
        else:
            models_to_test = args.models
    else:
        models_to_test = config.get('ablation', {}).get('models', 
                                                        ['linear_regression', 'random_forest', 'gradient_boosting'])
    
    # Run ablation study
    results_df, feature_columns_map, best_params_map = run_ablation_study(
        config,
        combinations_to_test=combinations_to_test,
        models_to_test=models_to_test,
        use_grid_search=args.grid_search,
        verbose=True
    )
    
    # Print summary
    print("\n" + "="*70)
    print("ABLATION STUDY RESULTS")
    print("="*70)
    
    # Summary by combination
    print("\nBest Test R² by Feature Combination:")
    print("-" * 50)
    combo_summary = results_df.groupby('combination').agg({
        'test_r2': 'max',
        'test_mae': 'min',
        'n_features': 'first'
    }).sort_values('test_r2', ascending=False)
    print(combo_summary.to_string())
    
    # Summary by model
    print("\n\nBest Test R² by Model:")
    print("-" * 50)
    model_summary = results_df.groupby('model').agg({
        'test_r2': ['mean', 'max'],
        'test_mae': ['mean', 'min']
    }).round(4)
    print(model_summary.to_string())
    
    # Full results table
    print("\n\nFull Results Table:")
    print("-" * 50)
    display_df = results_df[['combination', 'model', 'n_features', 'test_r2', 'test_mae', 'val_r2']].copy()
    display_df['test_mae'] = display_df['test_mae'].apply(lambda x: f"${x:,.2f}")
    display_df = display_df.sort_values(['combination', 'model'])
    print(display_df.to_string(index=False))
    
    # Save results
    output_path = args.output or f"{config['output']['predictions_dir']}/ablation_results.csv"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    # Find best overall configuration
    best_idx = results_df['test_r2'].idxmax()
    best = results_df.loc[best_idx]
    best_combo = best['combination']
    best_model = best['model']
    best_key = f"{best_combo}_{best_model}"
    
    print(f"\n{'='*70}")
    print(f"BEST MODEL FOUND")
    print(f"{'='*70}")
    print(f"\n  Model:       {best_model}")
    print(f"  Combination: {best_combo}")
    print(f"  N Features:  {best['n_features']}")
    print(f"  Test R²:     {best['test_r2']:.6f}")
    print(f"  Test MAE:    ${best['test_mae']:,.2f}")
    print(f"  Test RMSE:   ${best['test_rmse']:,.2f}")
    print(f"  Val R²:      {best['val_r2']:.6f}")
    
    # Print feature columns used
    print(f"\n  Feature Columns ({best['n_features']} total):")
    print(f"  {'-'*50}")
    if best_combo in feature_columns_map:
        cols = feature_columns_map[best_combo]
        # Print in groups of 5 for readability
        for i in range(0, len(cols), 5):
            chunk = cols[i:i+5]
            print(f"    {', '.join(chunk)}")
    
    # Print hyperparameters if available
    if best_key in best_params_map and best_params_map[best_key]:
        print(f"\n  Hyperparameters:")
        print(f"  {'-'*50}")
        for param, value in best_params_map[best_key].items():
            print(f"    {param}: {value}")
    else:
        print(f"\n  Hyperparameters: Default (no grid search or linear model)")
    
    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
