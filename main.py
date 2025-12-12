"""
Main execution script for TabPFN tire quality prediction
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer

from utils import (
    load_config, create_submission, print_selection_summary, 
    print_final_summary, suppress_warnings
)
from data_loader import load_data, preprocess_data
from feature_engineering import engineer_features
from feature_selection import split_bias_free_feature_selection
from model_training import train_ensemble_tabpfn


def select_samples(result_df, config):
    """
    Select samples based on consensus strategy
    
    Args:
        result_df: Results dataframe with probabilities and consensus flags
        config: Configuration dictionary
        
    Returns:
        selected_ids, selection_breakdown
    """
    max_samples = config['selection']['max_samples']
    max_prob = config['selection']['max_probability']
    consensus_priority = config['selection']['consensus_priority']
    
    # ⭐ NEW: Prob std filter
    use_prob_std_filter = config['selection'].get('use_prob_std_filter', False)
    max_prob_std = config['selection'].get('max_prob_std', 0.08)
    
    # Apply prob_std filter if enabled
    if use_prob_std_filter:
        print(f"\n⚡ Applying probability std filter (max_prob_std={max_prob_std})")
        before = len(result_df)
        result_df = result_df[result_df['prob_std'] <= max_prob_std].copy()
        after = len(result_df)
        removed = before - after
        print(f"   Filtered out {removed} samples (std > {max_prob_std})")
        print(f"   Remaining: {after} samples")
        
        if after == 0:
            print(f"   ⚠️  No samples pass prob_std filter!")
            print(f"   Suggestion: Increase max_prob_std (current: {max_prob_std})")
    
    # Prepare consensus pools (sorted by final_probability ascending)
    consensus_pools = {
        'unanimous': result_df[result_df['unanimous_good']].copy().sort_values('final_probability'),
        'strong': result_df[result_df['strong_consensus'] & ~result_df['unanimous_good']].copy().sort_values('final_probability'),
        'majority': result_df[result_df['majority_good'] & ~result_df['strong_consensus']].copy().sort_values('final_probability')
    }
    
    print(f"\n[Available Samples by Consensus Level]")
    for level, pool in consensus_pools.items():
        if len(pool) > 0:
            print(f"  {level.capitalize():12s}: {len(pool):3d} samples "
                  f"(final_prob range: {pool['final_probability'].min():.4f} - "
                  f"{pool['final_probability'].max():.4f})")
            if use_prob_std_filter:
                print(f"                      "
                      f"(prob_std range: {pool['prob_std'].min():.4f} - "
                      f"{pool['prob_std'].max():.4f})")
        else:
            print(f"  {level.capitalize():12s}: {len(pool):3d} samples")
    
    # Select samples by priority
    selected_ids = []
    selection_breakdown = {}
    
    print(f"\n[Selection Strategy]")
    print(f"  Priority: {' + '.join(consensus_priority)}")
    if use_prob_std_filter:
        print(f"  Prob std filter: ENABLED (max_prob_std={max_prob_std})")
    print(f"  Selecting ALL samples from specified consensus levels (max {max_samples})")
    
    for priority in consensus_priority:
        pool = consensus_pools[priority]
        
        # Apply probability filter if specified (kept for backward compatibility)
        if max_prob < 1.0:
            pool = pool[pool['final_probability'] <= max_prob]
            if len(pool) == 0:
                print(f"\n⚠️  No {priority} samples under final_prob {max_prob:.3f}")
                continue
        
        # Check if we've reached max samples
        if len(selected_ids) >= max_samples:
            print(f"\n⚠️  Reached MAX_SAMPLES limit ({max_samples}), stopping selection")
            break
        
        # Select from this tier (up to remaining capacity)
        remaining = max_samples - len(selected_ids)
        if len(pool) > remaining:
            print(f"\n⚠️  {priority.capitalize()}: {len(pool)} samples available, "
                  f"but only selecting {remaining} due to MAX_SAMPLES limit")
            tier_samples = pool.head(remaining)
        else:
            tier_samples = pool
        
        selected_ids.extend(tier_samples['ID'].values)
        selection_breakdown[priority] = len(tier_samples)
    
    return np.array(selected_ids), selection_breakdown


def main():
    """Main execution pipeline"""
    
    # Load configuration
    config = load_config('config.yaml')
    
    # Suppress warnings if configured
    if config['processing']['suppress_warnings']:
        suppress_warnings()
    
    print("="*70)
    print("  TabPFN TIRE QUALITY PREDICTION PIPELINE")
    print("="*70)
    
    # Step 1: Load data
    train_df, test_df, train_ids, test_ids, y = load_data(
        config['paths']['train_data'],
        config['paths']['test_data']
    )
    
    # Preprocess
    train_df, test_df = preprocess_data(train_df, test_df)
    y = pd.Series(y, index=train_df.index)
    
    # Step 2: Feature Engineering
    X, X_test_final = engineer_features(train_df, test_df, y, config)
    
    # Step 3: Feature Selection
    final_features = split_bias_free_feature_selection(
        X, y, 
        n_final=config['features']['n_features'],
        run_selection=config['features'].get('run_selection', True),
        predefined_features=config['features'].get('predefined_features', None)
    )
    
    X_selected = X[final_features]
    X_test_selected = X_test_final[final_features]
    
    # Apply Power Transform
    print("   -> Applying PowerTransformer on full dataset...")
    pt = PowerTransformer(method='yeo-johnson')
    X_full_pt = pd.DataFrame(
        pt.fit_transform(X_selected), 
        columns=final_features, 
        index=X.index
    )
    X_test_pt = pd.DataFrame(
        pt.transform(X_test_selected), 
        columns=final_features, 
        index=X_test_final.index
    )
    
    # Step 4: Train TabPFN Ensemble
    print("\n[STEP 4] TRAINING TabPFN ENSEMBLE")
    print("-"*70)
    print(f"   Passing ALL {len(X_full_pt)} samples to ensemble")
    print(f"   Each of {config['model']['n_models']} models will create its own train/val split")
    
    (mean_probs, unanimous_good, strong_consensus, majority_good, 
     all_models, all_val_aucs, best_model_idx, prob_std) = train_ensemble_tabpfn(
        X_full_pt, y, X_test_pt, 
        n_estimators=config['model']['n_estimators'], 
        n_models=config['model']['n_models']
    )
    
    # Get best model predictions
    best_model = all_models[best_model_idx]
    best_model_probs = best_model.predict_proba(X_test_pt)[:, 1]
    
    print("\n[STEP 4.5] FINAL MODEL (Full Training Data)")
    print("="*70)
    
    from model_training import train_final_model
    final_model, final_probs, final_preds, final_threshold = train_final_model(
        X_full_pt, y, X_test_pt, 
        n_estimators=config['model']['n_estimators']
    )
    
    # Final model says Good (prediction = 0)
    final_good = (final_preds == 0)
    
    # Step 5: Selection Strategy
    print("\n[STEP 5] SELECTION STRATEGY - FLEXIBLE CONSENSUS")
    print("="*70)
    
    result_df = pd.DataFrame({
        'ID': test_ids,
        'final_probability': mean_probs,
        'best_model_probability': best_model_probs,
        'final_model_probability': final_probs,  # NEW: Final model 확률
        'final_model_good': final_good,  # NEW: Final model 예측
        'prob_std': prob_std,  # NEW: 확률 표준편차 추가
        'unanimous_good': unanimous_good,
        'strong_consensus': strong_consensus,
        'majority_good': majority_good
    })
    
    # Select samples
    selected_ids, selection_breakdown = select_samples(result_df, config)
    
    # ⭐ NEW: Apply final model filter if enabled
    if config['selection'].get('use_final_model_filter', False):
        print("\n⚡ Applying FINAL MODEL FILTER (intersection)")
        before = len(selected_ids)
        
        # Keep only samples where final model ALSO says Good
        final_model_good_ids = result_df[result_df['final_model_good']]['ID'].values
        selected_ids = np.array([sid for sid in selected_ids if sid in final_model_good_ids])
        
        after = len(selected_ids)
        removed = before - after
        
        print(f"   Before: {before} samples")
        print(f"   Final model Good: {len(final_model_good_ids)} samples")
        print(f"   Intersection: {after} samples")
        print(f"   Removed: {removed} samples ({removed/before*100:.1f}%)")
        
        if after == 0:
            print(f"   ⚠️  No samples pass final model filter!")
            print(f"   Suggestion: Disable final model filter")
    
    # Update decision column
    result_df['decision'] = False
    result_df.loc[result_df['ID'].isin(selected_ids), 'decision'] = True
    
    # Print summary
    print_selection_summary(result_df, selected_ids, selection_breakdown, config)
    
    # Sort by final probability
    result_df = result_df.sort_values('final_probability', ascending=True)
    
    # Create submission
    submission = create_submission(
        result_df, 
        config['paths']['sample_submission'],
        config['paths']['output_file'],
        selected_ids
    )
    
    # Print final summary
    print_final_summary(
        config, 
        config['model']['n_models'],
        len(selected_ids),
        config['features']['n_features'],
        config['selection']['consensus_priority']
    )


if __name__ == '__main__':
    main()