"""
Utility functions module
"""
import numpy as np
import pandas as pd
import yaml


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_base_id(full_id):
    """Extract base ID from full ID string"""
    return full_id.rsplit('_', 1)[0]


def create_submission(result_df, sample_submission_path, output_path, selected_ids):
    """
    Create submission file
    
    Args:
        result_df: Results dataframe with probabilities and decisions
        sample_submission_path: Path to sample submission file
        output_path: Path to save submission
        selected_ids: Array of selected sample IDs
        
    Returns:
        submission dataframe
    """
    try:
        submission = pd.read_csv(sample_submission_path)
        
        submission['base_id'] = submission['ID'].apply(get_base_id)
        
        prob_dict = result_df.set_index('ID')['final_probability'].to_dict()
        dec_dict = result_df.set_index('ID')['decision'].to_dict()
        
        submission['probability'] = submission['base_id'].map(prob_dict).fillna(1.0)
        submission['decision'] = submission['base_id'].map(dec_dict).fillna(False).astype(bool)
        
        submission = submission.drop(columns=['base_id'])
        submission.to_csv(output_path, index=False)
        
        print(f"\n✓ Saved '{output_path}'")
        print(f"  Final selection: {submission['decision'].sum()} samples")
        
        return submission
        
    except Exception as e:
        # Fallback: save result_df directly
        result_df.to_csv('submission_source.csv', index=False)
        print(f"\n✓ Saved 'submission_source.csv'")
        print(f"  Error: {e}")
        return None


def print_selection_summary(result_df, selected_ids, selection_breakdown, config):
    """
    Print summary of selection strategy and results
    
    Args:
        result_df: Results dataframe
        selected_ids: Selected sample IDs
        selection_breakdown: Dictionary of selection counts by level
        config: Configuration dictionary
    """
    selected_df = result_df[result_df['ID'].isin(selected_ids)]
    
    print(f"\n[Final Selection]")
    print(f"  Total Selected: {len(selected_ids)}")
    print(f"\n  Composition:")
    for level, count in selection_breakdown.items():
        pct = (count / len(selected_ids)) * 100 if len(selected_ids) > 0 else 0
        print(f"    {level.capitalize():12s}: {count:3d} samples ({pct:5.1f}%)")
    
    print(f"\n  Probability Statistics (final_probability):")
    print(f"    Mean: {selected_df['final_probability'].mean():.4f}")
    print(f"    Max:  {selected_df['final_probability'].max():.4f}")
    print(f"    Min:  {selected_df['final_probability'].min():.4f}")
    
    # Risk analysis
    high_risk = selected_df[selected_df['final_probability'] > 0.3]
    if len(high_risk) > 0:
        print(f"\n  ⚠️  High Risk (final_prob > 0.3): {len(high_risk)} samples")
    else:
        print(f"\n  ✅ All samples < 0.3 final_probability")


def print_final_summary(config, n_models, n_selected, n_features, consensus_priority):
    """Print final configuration summary"""
    print("\n" + "="*70)
    print("  COMPLETED")
    print("="*70)
    print(f"\nFinal Configuration:")
    print(f"  Models:       {n_models}")
    print(f"  Selected:     {n_selected} samples")
    print(f"  Features:     {n_features}")
    print(f"  Strategy:     {' + '.join(consensus_priority)}")
    print("="*70)


def suppress_warnings():
    """Suppress various warnings for cleaner output"""
    import warnings
    import os
    
    warnings.filterwarnings('ignore')
    os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"