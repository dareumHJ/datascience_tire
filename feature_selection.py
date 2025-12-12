"""
Feature selection module
"""
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr
from collections import Counter
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion


def phase1_model_free_filtering(X, y, top_k=40):
    """
    Model-free methods for initial filtering (no split bias)
    
    Uses three methods:
    1. Mutual Information
    2. ANOVA F-test
    3. Spearman Correlation
    
    Returns: Selected features based on voting
    """
    print("\n" + "="*70)
    print("PHASE 1: Model-Free Pre-Filtering")
    print("="*70)
    print("✅ No split bias - using full data!\n")
    
    n_features = X.shape[1]
    feature_names = X.columns.tolist()
    
    # Method 1: Mutual Information
    print("[1/3] Mutual Information...")
    mi_scores = mutual_info_classif(X, y, random_state=42, n_jobs=-1)
    mi_rank = np.argsort(mi_scores)[::-1]
    mi_top = [feature_names[i] for i in mi_rank[:top_k]]
    print(f"  Top 5: {mi_top[:5]}")
    
    # Method 2: ANOVA F-test
    print("\n[2/3] ANOVA F-test...")
    f_scores, _ = f_classif(X, y)
    f_rank = np.argsort(f_scores)[::-1]
    f_top = [feature_names[i] for i in f_rank[:top_k]]
    print(f"  Top 5: {f_top[:5]}")
    
    # Method 3: Spearman Correlation
    print("\n[3/3] Spearman Correlation...")
    corr_scores = []
    for col in feature_names:
        corr, _ = spearmanr(X[col], y)
        corr_scores.append(abs(corr))
    
    corr_scores = np.array(corr_scores)
    corr_rank = np.argsort(corr_scores)[::-1]
    corr_top = [feature_names[i] for i in corr_rank[:top_k]]
    print(f"  Top 5: {corr_top[:5]}")
    
    # Voting
    print("\n[Voting]")
    all_candidates = set(mi_top + f_top + corr_top)
    
    vote_scores = {}
    for feat in all_candidates:
        score = 0
        methods = []
        
        if feat in mi_top:
            rank = mi_top.index(feat)
            score += (top_k - rank)
            methods.append("MI")
        
        if feat in f_top:
            rank = f_top.index(feat)
            score += (top_k - rank)
            methods.append("F")
        
        if feat in corr_top:
            rank = corr_top.index(feat)
            score += (top_k - rank)
            methods.append("Corr")
        
        vote_scores[feat] = {
            'score': score,
            'n_methods': len(methods),
            'methods': methods
        }
    
    # Sort by score and number of methods
    sorted_features = sorted(
        vote_scores.items(), 
        key=lambda x: (x[1]['score'], x[1]['n_methods']), 
        reverse=True
    )
    
    # Select top features
    selected_features = [feat for feat, info in sorted_features[:top_k]]
    
    print(f"\n[Top 20 Features by Model-Free Voting]")
    print(f"{'Rank':<6}{'Feature':<35}{'Score':<8}{'Methods'}")
    print("-"*70)
    for i, (feat, info) in enumerate(sorted_features[:20]):
        methods_str = ','.join(info['methods'])
        print(f"{i+1:<6}{feat:<35}{info['score']:<8}{methods_str}")
    
    print(f"\n✅ Selected {len(selected_features)} features (no split bias!)")
    
    return selected_features


def phase2_stability_selection(X, y, candidate_features, n_iterations=20, threshold=0.6):
    """
    Stability selection using multiple random splits
    
    Args:
        X: Feature dataframe
        y: Target labels
        candidate_features: Features from phase 1
        n_iterations: Number of random splits
        threshold: Minimum frequency to be considered stable
        
    Returns:
        stable_features, feature_counts
    """
    print("\n" + "="*70)
    print("PHASE 2: Stability Selection")
    print("="*70)
    print(f"Testing {len(candidate_features)} features with {n_iterations} iterations\n")
    
    X_subset = X[candidate_features]
    feature_selection_counts = Counter()
    
    for iteration in range(n_iterations):
        print(f"[Iteration {iteration+1}/{n_iterations}]", end=" ", flush=True)
        
        seed = 100 + iteration
        X_train, X_val, y_train, y_val = train_test_split(
            X_subset, y, test_size=0.2, random_state=seed, stratify=y
        )
        
        # Train TabPFN
        tabpfn = TabPFNClassifier.create_default_for_version(
            ModelVersion.V2,
            device='auto',
            n_estimators=32,
        )
        tabpfn.fit(X_train, y_train)
        
        # Permutation importance
        result = permutation_importance(
            tabpfn, X_val, y_val,
            n_repeats=3,
            random_state=seed,
            scoring='roc_auc'
        )
        
        # Top 20 features
        top_indices = np.argsort(result.importances_mean)[::-1][:20]
        top_features = [candidate_features[i] for i in top_indices]
        
        for feat in top_features:
            feature_selection_counts[feat] += 1
        
        print(f"Top 5: {top_features[:5]}")
    
    # Filter by stability threshold
    threshold_count = int(n_iterations * threshold)
    stable_features = [
        feat for feat, count in feature_selection_counts.items() 
        if count >= threshold_count
    ]
    
    print(f"\n[Stability Results]")
    print(f"{'Feature':<35}{'Selected':<10}{'Frequency'}")
    print("-"*70)
    for feat, count in feature_selection_counts.most_common(30):
        freq = count / n_iterations
        stable = "✅" if count >= threshold_count else ""
        print(f"{feat:<35}{stable:<10}{count}/{n_iterations} ({freq:.1%})")
    
    print(f"\n✅ Selected {len(stable_features)} stable features")
    print(f"   (appeared in {threshold:.0%}+ of iterations)")
    
    return stable_features, feature_selection_counts


def phase3_final_validation(X, y, stable_features, feature_counts, test_sizes=[8, 10, 12, 15]):
    """
    Find optimal number of features using cross-validation
    
    Args:
        X: Feature dataframe
        y: Target labels
        stable_features: Stable features from phase 2
        feature_counts: Feature selection counts
        test_sizes: List of feature counts to test
        
    Returns:
        best_features
    """
    print("\n" + "="*70)
    print("PHASE 3: Final Validation")
    print("="*70)
    
    # Sort by stability
    sorted_by_stability = sorted(
        feature_counts.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    stable_feature_order = [
        feat for feat, count in sorted_by_stability 
        if feat in stable_features
    ]
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    
    print(f"\nTesting different sizes: {test_sizes}\n")
    
    for n_feat in test_sizes:
        if n_feat > len(stable_feature_order):
            print(f"[Skipping {n_feat}] Not enough stable features")
            continue
        
        print(f"[Testing {n_feat} features]")
        selected_feats = stable_feature_order[:n_feat]
        X_subset = X[selected_feats]
        
        aucs = []
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_subset, y)):
            X_train = X_subset.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_val = X_subset.iloc[val_idx]
            y_val = y.iloc[val_idx]
            
            tabpfn = TabPFNClassifier.create_default_for_version(
                ModelVersion.V2,
                device='auto',
                n_estimators=32,
            )
            tabpfn.fit(X_train, y_train)
            
            probs = tabpfn.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, probs)
            aucs.append(auc)
        
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        
        results[n_feat] = {
            'mean_auc': mean_auc,
            'std_auc': std_auc,
            'features': selected_feats
        }
        
        print(f"  Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
        print(f"  Top 5 features: {selected_feats[:5]}")
    
    # Find best size
    print(f"\n[Summary]")
    print(f"{'N Features':<12}{'Mean AUC':<15}{'Std AUC'}")
    print("-"*40)
    for n_feat in sorted(results.keys()):
        info = results[n_feat]
        print(f"{n_feat:<12}{info['mean_auc']:<15.4f}{info['std_auc']:.4f}")
    
    best_n = max(results.keys(), key=lambda k: results[k]['mean_auc'])
    best_features = results[best_n]['features']
    
    print(f"\n✅ Optimal size: {best_n} features")
    print(f"   Mean AUC: {results[best_n]['mean_auc']:.4f}")
    print(f"   Std AUC: {results[best_n]['std_auc']:.4f}")
    
    return best_features


def split_bias_free_feature_selection(X, y, n_final=80, run_selection=True, predefined_features=None):
    """
    Complete 3-phase feature selection pipeline
    
    Args:
        X: Feature dataframe
        y: Target labels
        n_final: Target number of features
        run_selection: If False, use predefined_features instead
        predefined_features: List of feature names to use when run_selection=False
        
    Returns:
        final_features
    """
    print("\n[STEP 3] FEATURE SELECTION")
    print("-"*70)
    
    # Skip feature selection if run_selection=False
    if not run_selection:
        if predefined_features is None:
            raise ValueError("predefined_features must be provided when run_selection=False")
        
        # Validate features exist
        missing_features = [f for f in predefined_features if f not in X.columns]
        if missing_features:
            print(f"⚠️  Missing features: {missing_features}")
            print(f"   Using available features only")
            final_features = [f for f in predefined_features if f in X.columns]
        else:
            final_features = predefined_features
        
        print(f"   ⚡ SKIPPING feature selection (using predefined features)")
        print(f"   Using {len(final_features)} predefined features:")
        for i, feat in enumerate(final_features, 1):
            print(f"      {i:2d}. {feat}")
        
        return final_features
    
    # Run full 3-phase selection
    print(f"   Using all {len(X)} samples for feature selection")
    
    # Phase 1: Model-free filtering
    candidate_features = phase1_model_free_filtering(X, y, top_k=40)
    
    # Phase 2: Stability selection
    stable_features, feature_counts = phase2_stability_selection(
        X, y, candidate_features, n_iterations=20, threshold=0.6
    )
    
    # Phase 3: Final validation
    test_sizes = [8, 10, 12, 15, 20]
    final_features = phase3_final_validation(
        X, y, stable_features, feature_counts, test_sizes=test_sizes
    )
    
    # If we need more features, add from candidate list
    if len(final_features) < n_final:
        print(f"\n⚠️  Only {len(final_features)} features selected, padding to {n_final}...")
        
        # Add remaining candidates by stability
        sorted_candidates = sorted(
            feature_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        for feat, count in sorted_candidates:
            if feat not in final_features:
                final_features.append(feat)
                if len(final_features) >= n_final:
                    break
    
    print(f"\n>>> Using {len(final_features)} features")
    
    return final_features