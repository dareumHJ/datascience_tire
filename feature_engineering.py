"""
Feature engineering module for tire pressure data
"""
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks


def extract_tire_features(df):
    """
    물리 기반 타이어 feature 추출
    
    압력 분포 (p0~p255):
    - 타이어를 세워서 앞에서 본 상태
    - x축으로 눌렀을 때
    - 타이어 line을 따라 각 (xi, yi) 좌표의 압력
    
    G1~G4:
    - FEM 시뮬레이션 변화량 통계
    """
    p_cols = [f'p{i}' for i in range(256)]
    x_cols = [f'x{i}' for i in range(256)]
    y_cols = [f'y{i}' for i in range(256)]
    
    if not all(c in df.columns for c in p_cols):
        return pd.DataFrame()
    
    P = df[p_cols].values
    X = df[x_cols].values if all(c in df.columns for c in x_cols) else np.zeros_like(P)
    Y = df[y_cols].values if all(c in df.columns for c in y_cols) else np.zeros_like(P)
    
    feats = pd.DataFrame(index=df.index)
    
    # =========================================================================
    # [A] 기본 압력 통계
    # =========================================================================
    feats['p_mean'] = np.mean(P, axis=1)
    feats['p_std'] = np.std(P, axis=1)
    feats['p_max'] = np.max(P, axis=1)
    feats['p_min'] = np.min(P, axis=1)
    feats['p_median'] = np.median(P, axis=1)
    feats['p_sum'] = np.sum(P, axis=1)
    feats['p_range'] = feats['p_max'] - feats['p_min']
    feats['p_skew'] = skew(P, axis=1)
    feats['p_kurtosis'] = kurtosis(P, axis=1)
    feats['p_cv'] = feats['p_std'] / (feats['p_mean'] + 1e-9)
    feats['p_q25'] = np.percentile(P, 25, axis=1)
    feats['p_q75'] = np.percentile(P, 75, axis=1)
    feats['p_iqr'] = feats['p_q75'] - feats['p_q25']
    
    # =========================================================================
    # [B] Line 따라 변화 (1D sequence analysis)
    # =========================================================================
    P_diff = np.diff(P, axis=1)
    feats['seq_grad_mean'] = np.mean(np.abs(P_diff), axis=1)
    feats['seq_grad_max'] = np.max(np.abs(P_diff), axis=1)
    feats['seq_grad_std'] = np.std(P_diff, axis=1)
    
    threshold_change = np.percentile(np.abs(P_diff), 90, axis=1, keepdims=True)
    feats['sudden_change_count'] = np.sum(np.abs(P_diff) > threshold_change, axis=1)
    
    P_diff2 = np.diff(P_diff, axis=1)
    feats['curvature_mean'] = np.mean(np.abs(P_diff2), axis=1)
    feats['curvature_max'] = np.max(np.abs(P_diff2), axis=1)
    
    # =========================================================================
    # [C] Peak 분석 (압력 집중 지점)
    # =========================================================================
    peak_counts = []
    peak_prominences = []
    peak_distances = []
    
    for i in range(len(df)):
        peaks, properties = find_peaks(P[i], prominence=feats['p_std'].iloc[i]*0.5)
        peak_counts.append(len(peaks))
        
        if len(peaks) > 0:
            peak_prominences.append(np.mean(properties['prominences']))
            if len(peaks) > 1:
                peak_distances.append(np.mean(np.diff(peaks)))
            else:
                peak_distances.append(0)
        else:
            peak_prominences.append(0)
            peak_distances.append(0)
    
    feats['peak_count'] = peak_counts
    feats['peak_prominence'] = peak_prominences
    feats['peak_spacing'] = peak_distances
    
    max_positions = np.argmax(P, axis=1)
    feats['max_position_centered'] = np.abs(max_positions - 127.5) / 127.5
    
    # =========================================================================
    # [D] Center of Pressure (무게중심)
    # =========================================================================
    total_p = np.sum(P, axis=1) + 1e-9
    cop_x = np.sum(X * P, axis=1) / total_p
    cop_y = np.sum(Y * P, axis=1) / total_p
    
    center_x = np.mean(X, axis=1)
    center_y = np.mean(Y, axis=1)
    
    feats['cop_offset_x'] = cop_x - center_x
    feats['cop_offset_y'] = cop_y - center_y
    feats['cop_offset_dist'] = np.sqrt(feats['cop_offset_x']**2 + feats['cop_offset_y']**2)
    
    # =========================================================================
    # [E] Geometry (타이어 접지 형상)
    # =========================================================================
    width = np.max(X, axis=1) - np.min(X, axis=1) + 1e-9
    height = np.max(Y, axis=1) - np.min(Y, axis=1) + 1e-9
    feats['geom_width'] = width
    feats['geom_height'] = height
    feats['geom_aspect'] = width / height
    feats['geom_area'] = width * height
    feats['pressure_density'] = feats['p_sum'] / feats['geom_area']
    feats['cop_offset_norm'] = feats['cop_offset_dist'] / width
    
    # =========================================================================
    # [F] 좌우 비대칭 (물리적 중요!)
    # =========================================================================
    p_left = np.mean(P[:, :128], axis=1)
    p_right = np.mean(P[:, 128:], axis=1)
    feats['pressure_asymmetry'] = np.abs(p_left - p_right)
    feats['pressure_asymmetry_ratio'] = feats['pressure_asymmetry'] / (feats['p_mean'] + 1e-9)
    
    # 집중도
    feats['pressure_concentration'] = feats['p_max'] / (feats['p_mean'] + 1e-9)
    
    # =========================================================================
    # [G] Shape vectors (X1~X5, Y1~Y5) - 타이어 형태 벡터
    # =========================================================================
    X_vecs = [f'X{i}' for i in range(1, 6)]
    Y_vecs = [f'Y{i}' for i in range(1, 6)]
    
    if all(c in df.columns for c in X_vecs):
        X_vals = df[X_vecs].values
        Y_vals = df[Y_vecs].values
        
        feats['shape_magnitude'] = np.sqrt(np.sum(X_vals**2 + Y_vals**2, axis=1))
        feats['shape_x_bias'] = np.sum(np.abs(X_vals), axis=1)
        feats['shape_y_bias'] = np.sum(np.abs(Y_vals), axis=1)
        feats['shape_asymmetry'] = feats['shape_x_bias'] / (feats['shape_y_bias'] + 1e-9)
    
    # =========================================================================
    # [H] FEM 시뮬레이션 변화량 통계 (G1~G4)
    # =========================================================================
    G_cols = [f'G{i}' for i in range(1, 5)]
    if all(c in df.columns for c in G_cols):
        for col in G_cols:
            feats[col] = df[col]
        
        G_vals = df[G_cols].values
        feats['G_mean'] = np.mean(G_vals, axis=1)
        feats['G_std'] = np.std(G_vals, axis=1)
        feats['G_max'] = np.max(G_vals, axis=1)
        feats['G_min'] = np.min(G_vals, axis=1)
        feats['G_range'] = feats['G_max'] - feats['G_min']
        
        # G 변화율 (시계열로 가정)
        feats['G_diff_1'] = df['G2'] - df['G1']
        feats['G_diff_2'] = df['G3'] - df['G2']
        feats['G_diff_3'] = df['G4'] - df['G3']
        feats['G_diff_mean'] = np.mean([feats['G_diff_1'], feats['G_diff_2'], feats['G_diff_3']], axis=0)
        feats['G_diff_std'] = np.std([feats['G_diff_1'], feats['G_diff_2'], feats['G_diff_3']], axis=0)
        
        # G 트렌드
        feats['G_trend'] = (df['G4'] - df['G1']) / 3  # 평균 변화율
    
    # =========================================================================
    # [I] 타이어 스펙 기반 feature
    # =========================================================================
    if 'Width' in df.columns:
        width_val = df['Width'].values
        aspect_val = df['Aspect'].values if 'Aspect' in df.columns else np.ones(len(df))
        inch_val = df['Inch'].values if 'Inch' in df.columns else np.ones(len(df))
        
        feats['pressure_per_width'] = feats['p_sum'] / (width_val + 1e-9)
        feats['pressure_per_aspect'] = feats['p_sum'] / (aspect_val + 1e-9)
        feats['cop_offset_per_width'] = feats['cop_offset_dist'] / (width_val + 1e-9)
        feats['pressure_variability_per_width'] = feats['p_std'] / (width_val + 1e-9)
        feats['gradient_per_width'] = feats['seq_grad_mean'] / (width_val + 1e-9)
        
        # 타이어 크기 지표
        feats['tire_volume_proxy'] = width_val * aspect_val * inch_val
    
    # =========================================================================
    # [J] 공정 파라미터 interactions (중요한 것만)
    # =========================================================================
    proc_cols = [c for c in df.columns if 'Proc_Param' in c and c != 'Proc_Param6']
    if len(proc_cols) >= 2:
        # 상위 3개만 (너무 많으면 noise)
        for col in proc_cols[:3]:
            if df[col].dtype in ['int64', 'float64']:
                feats[f'{col}_x_pressure_dens'] = df[col] * feats['pressure_density']
                feats[f'{col}_x_cop_offset'] = df[col] * feats['cop_offset_dist']
    
    return feats


# =============================================================================
# PCA Features
# =============================================================================

def add_pca_features(train_df, test_df, n_components=10):
    """PCA features from pressure data"""
    p_cols = [f'p{i}' for i in range(256)]
    
    pca = PCA(n_components=n_components, random_state=42)
    scaler = StandardScaler()
    
    P_train = train_df[p_cols].values
    P_test = test_df[p_cols].values
    
    P_train_sc = scaler.fit_transform(P_train)
    P_test_sc = scaler.transform(P_test)
    
    pca.fit(P_train_sc)
    
    train_pca = pca.transform(P_train_sc)
    test_pca = pca.transform(P_test_sc)
    
    train_pca_df = pd.DataFrame(
        train_pca, 
        columns=[f'pca_p_{i}' for i in range(n_components)], 
        index=train_df.index
    )
    test_pca_df = pd.DataFrame(
        test_pca, 
        columns=[f'pca_p_{i}' for i in range(n_components)], 
        index=test_df.index
    )
    
    return train_pca_df, test_pca_df


# =============================================================================
# Anomaly Features
# =============================================================================

def add_anomaly_features(X_train, X_test, y_train):
    """Anomaly detection using Isolation Forest"""
    # Good 샘플만으로 학습
    X_good = X_train[y_train == 0]
    
    iso = IsolationForest(
        contamination=0.1,
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    iso.fit(X_good)
    
    # Score 계산
    train_scores = iso.score_samples(X_train)
    test_scores = iso.score_samples(X_test)
    
    train_anom = pd.DataFrame({
        'isolation_score': train_scores
    }, index=X_train.index)
    
    test_anom = pd.DataFrame({
        'isolation_score': test_scores
    }, index=X_test.index)
    
    return train_anom, test_anom

# =============================================================================
# Encode Categorical Features
# =============================================================================

def encode_categorical_features(train_df, test_df):
    """
    Categorical을 숫자로 변환 (One-Hot 대신!)
    
    - Mass_Pilot: True=1, False=0
    - Proc_Param6: P6_0=0, P6_1=1, P6_2=2 (ordinal)
    - Plant: 완전 배제!
    """
    # Mass_Pilot: boolean → int
    train_df['Mass_Pilot'] = train_df['Mass_Pilot'].astype(int)
    test_df['Mass_Pilot'] = test_df['Mass_Pilot'].astype(int)
    
    # Proc_Param6: Ordinal encoding
    param6_map = {'P6_0': 0, 'P6_1': 1, 'P6_2': 2}
    train_df['Proc_Param6'] = train_df['Proc_Param6'].map(param6_map)
    test_df['Proc_Param6'] = test_df['Proc_Param6'].map(param6_map)
    
    # Plant: 완전 제거! (bias 제거)
    if 'Plant' in train_df.columns:
        train_df = train_df.drop('Plant', axis=1)
    if 'Plant' in test_df.columns:
        test_df = test_df.drop('Plant', axis=1)
    
    return train_df, test_df


# =============================================================================
# Main Feature Engineering Pipeline
# =============================================================================

def engineer_features(train_df, test_df, y, config):
    """
    Main feature engineering pipeline
    
    Args:
        train_df: Training dataframe
        test_df: Test dataframe
        y: Training labels
        config: Configuration dictionary
        
    Returns:
        X_train, X_test with engineered features
    """
    print("\n[STEP 2] FEATURE ENGINEERING")
    print("-" * 70)
    
    train_df, test_df = encode_categorical_features(train_df.copy(), test_df.copy())
    
    train_feats = extract_tire_features(train_df)
    test_feats = extract_tire_features(test_df)
    
    print(f"   Physics features: {len(train_feats.columns)}")
    
    print("   -> Adding PCA features...")
    train_pca, test_pca = add_pca_features(train_df, test_df, n_components=10)
    
    train_feats = pd.concat([train_feats, train_pca], axis=1)
    test_feats = pd.concat([test_feats, test_pca], axis=1)
    
    # Meta features (Proc_Param, Mass_Pilot, Width, Aspect, Inch)
    meta_cols = ['Mass_Pilot', 'Width', 'Aspect', 'Inch', 'Proc_Param6']
    proc_cols = [c for c in train_df.columns if 'Proc_Param' in c and c != 'Proc_Param6']
    meta_cols.extend(proc_cols)
    
    meta_cols = [c for c in meta_cols if c in train_df.columns]
    
    X = pd.concat([train_df[meta_cols], train_feats], axis=1)
    X_test = pd.concat([test_df[meta_cols], test_feats], axis=1)
    
    # Fill NaN
    X = X.fillna(0)
    X_test = X_test.fillna(0)
    
    print(f"   Total features before anomaly: {X.shape[1]}")
    
    train_anom, test_anom = add_anomaly_features(X, X_test, y)
    
    X = pd.concat([X, train_anom], axis=1)
    X_test = pd.concat([X_test, test_anom], axis=1)
    
    print(f"   ✅ Total features: {X.shape[1]}")
    
    return X, X_test