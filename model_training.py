"""
Model training module for TabPFN ensemble
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion


def find_optimal_threshold(y_val, val_probs, target_ng_recall=0.40):
    """
    Find optimal threshold to maximize Good Precision
    while maintaining minimum NG Recall
    
    Args:
        y_val: Validation labels
        val_probs: Validation probabilities
        target_ng_recall: Minimum NG recall to maintain
        
    Returns:
        best_threshold
    """
    thresholds = np.linspace(0.3, 0.7, 100)
    best_precision = 0
    best_threshold = 0.5
    
    for thresh in thresholds:
        preds = (val_probs >= thresh).astype(int)
        cm = confusion_matrix(y_val, preds)
        
        if cm.shape != (2, 2):
            continue
        
        # Good Precision
        good_precision = cm[0, 0] / (cm[0, 0] + cm[1, 0]) if (cm[0, 0] + cm[1, 0]) > 0 else 0
        
        # NG Recall
        ng_recall = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
        
        # Maximize precision while maintaining minimum recall
        if ng_recall >= target_ng_recall and good_precision > best_precision:
            best_precision = good_precision
            best_threshold = thresh
    
    # If target NG recall not achievable, maximize precision only
    if best_precision == 0:
        print("   ⚠️  Optimizing for maximum Good Precision only...")
        for thresh in thresholds:
            preds = (val_probs >= thresh).astype(int)
            cm = confusion_matrix(y_val, preds)
            
            if cm.shape != (2, 2):
                continue
            
            good_precision = cm[0, 0] / (cm[0, 0] + cm[1, 0]) if (cm[0, 0] + cm[1, 0]) > 0 else 0
            
            if good_precision > best_precision:
                best_precision = good_precision
                best_threshold = thresh
    
    return best_threshold


def train_ensemble_tabpfn(X, y, X_test, n_estimators=1, n_models=10):
    """
    Train ensemble of TabPFN models with different random splits
    
    Args:
        X: Training features
        y: Training labels
        X_test: Test features
        n_estimators: Number of TabPFN estimators per model
        n_models: Number of models in ensemble
        
    Returns:
        mean_probs: Mean probabilities across models
        unanimous_good: Boolean array for unanimous good predictions
        strong_consensus: Boolean array for strong consensus
        majority_good: Boolean array for majority good predictions
        all_models: List of trained models
        all_val_aucs: List of validation AUCs
        best_model_idx: Index of best model
    """
    print("\n" + "="*70)
    print(f"  ENSEMBLE TabPFN TRAINING ({n_models} models)")
    print("="*70)
    
    # Generate random states
    base_states = [42, 1, 100, 7, 23, 88, 15, 50, 77, 3, 
                   11, 33, 55, 66, 99, 123, 200, 17, 8, 25]
    
    if n_models <= len(base_states):
        random_states = base_states[:n_models]
    else:
        random_states = base_states + list(range(300, 300 + n_models - len(base_states)))
    
    print(f"Random states: {random_states}")
    
    all_test_probs = []
    all_thresholds = []
    all_models = []
    all_val_aucs = []
    
    for i, rs in enumerate(random_states):
        print(f"\n[Model {i+1}/{n_models}] Random State: {rs}")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=rs, stratify=y
        )
        
        # Train TabPFN
        model = TabPFNClassifier.create_default_for_version(
            ModelVersion.V2,
            device='auto',
            n_estimators=n_estimators,
            balance_probabilities=True
        )
        
        model.fit(X_train, y_train)
        
        # Find optimal threshold on validation set
        val_probs = model.predict_proba(X_val)[:, 1]
        threshold = find_optimal_threshold(y_val, val_probs, target_ng_recall=0.40)
        
        # Evaluate on validation set
        val_preds = (val_probs >= threshold).astype(int)
        cm = confusion_matrix(y_val, val_preds)
        
        good_precision = cm[0, 0] / (cm[0, 0] + cm[1, 0]) if (cm[0, 0] + cm[1, 0]) > 0 else 0
        ng_recall = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
        val_auc = roc_auc_score(y_val, val_probs)
        
        all_val_aucs.append(val_auc)
        
        print(f"   Threshold: {threshold:.4f} | Good Prec: {good_precision:.4f} | "
              f"NG Recall: {ng_recall:.4f} | Val AUC: {val_auc:.4f}")
        
        # Predict on test set
        test_probs = model.predict_proba(X_test)[:, 1]
        
        all_test_probs.append(test_probs)
        all_thresholds.append(threshold)
        all_models.append(model)
    
    # Ensemble voting
    print("\n" + "="*70)
    print("  ENSEMBLE VOTING STRATEGY")
    print("="*70)
    
    # Convert probabilities to predictions (Good=0, NG=1)
    all_predictions = []
    for probs, thresh in zip(all_test_probs, all_thresholds):
        preds = (probs >= thresh).astype(int)
        all_predictions.append(preds)
    
    all_predictions = np.array(all_predictions)  # [n_models, n_samples]
    
    # Count NG votes
    vote_sum = np.sum(all_predictions, axis=0)
    
    # Define consensus levels (dynamically based on n_models)
    unanimous_good = (vote_sum == 0)  # All models say Good
    
    # Strong: 85-90% models say Good
    strong_threshold = max(2, int(n_models * 0.15))  # Allow max 15%
    strong_consensus = (vote_sum <= strong_threshold)
    
    # Majority: 60%+ models say Good
    majority_threshold = int(n_models * 0.4)  # Max 40% NG
    majority_good = (vote_sum <= majority_threshold)
    
    print(f"\nVoting Results:")
    print(f"  All {n_models} models say Good (unanimous):     {np.sum(unanimous_good):3d} samples")
    print(f"  {n_models-strong_threshold}+ models say Good (strong):         "
          f"{np.sum(strong_consensus):3d} samples")
    print(f"  {n_models-majority_threshold}+ models say Good (majority):       "
          f"{np.sum(majority_good):3d} samples")
    
    # Mean probability for ordering
    mean_probs = np.mean(all_test_probs, axis=0)
    
    # Select best model
    best_model_idx = np.argmax(all_val_aucs)
    best_val_auc = all_val_aucs[best_model_idx]
    print(f"\n🏆 Best Model: #{best_model_idx+1} "
          f"(Random State {random_states[best_model_idx]}) "
          f"with Val AUC: {best_val_auc:.4f}")
    
    return (mean_probs, unanimous_good, strong_consensus, majority_good, 
            all_models, all_val_aucs, best_model_idx)