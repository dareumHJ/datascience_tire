# TabPFN Tire Quality Prediction - Refactored

타이어 품질 예측을 위한 TabPFN 기반 머신러닝 파이프라인 (리팩토링 버전)

## 📁 파일 구조

```
project/
├── config.yaml              # 모든 설정값 (하이퍼파라미터, 경로 등)
├── main.py                  # 메인 실행 파일
├── data_loader.py           # 데이터 로딩 및 전처리
├── feature_engineering.py   # 피처 엔지니어링 (압력, PCA, 이상치 탐지 등)
├── feature_selection.py     # 3단계 피처 선택 (Model-free → Stability → Validation)
├── model_training.py        # TabPFN 앙상블 학습
├── utils.py                 # 유틸리티 함수들
├── README.md               # 이 파일
├── train.csv               # 학습 데이터 (ID 없음, 타겟: Class)
├── test.csv                # 테스트 데이터 (ID 있음)
└── sample_submission.csv   # 제출 양식
```

## 📊 데이터 형식

### train.csv (720 samples, 799 columns)
- **타겟 컬럼**: `Class` (문자열: "Good"=양품, "NG"=불량)
  - 자동으로 숫자로 변환됨: Good→0, NG→1
  - 분포: Good=613, NG=107
- **ID 컬럼**: 없음 (자동 생성)
- **피처**: Mass_Pilot, Width, Aspect, Inch, Plant, Proc_Param1-11, X1-X5, Y1-Y5, G1-G4, p0-p255, x0-x255, y0-y255

### test.csv (466 samples, 799 columns)
- **ID 컬럼**: ID_0, ID_1, ... (있음)
- **피처**: train.csv와 동일 (Class 제외)

### sample_submission.csv
- **형식**: ID, probability, decision
- **ID**: ID_0_L, ID_0_R, ID_1_L, ID_1_R, ... (좌우 타이어)

## 🚀 사용 방법

### 1. 기본 실행
```bash
python main.py
```

### 2. 설정 변경
`config.yaml` 파일을 수정하여 파라미터 조정:

```yaml
# 모델 개수 변경
model:
  n_models: 10  # 앙상블 모델 개수

# 피처 개수 변경
features:
  n_features: 80  # 최종 선택할 피처 개수

# 선택 전략 변경
selection:
  max_samples: 200  # 최대 선택 샘플 수
  consensus_priority:  # 우선순위 변경 가능
    - unanimous
    - strong
```

## 📊 주요 기능

### 1. Feature Engineering (`feature_engineering.py`)
- **압력 기반 피처**: 비대칭성, 집중도, 그래디언트
- **타이어 통계 피처**: 평균, 표준편차, 왜도, 첨도 등 60+ 피처
- **PCA 피처**: 압력 데이터의 주성분 분석
- **이상치 탐지**: Isolation Forest 기반 이상치 스코어

### 2. Feature Selection (`feature_selection.py`)
**3단계 선택 전략 (Split Bias 최소화):**

1. **Phase 1 - Model-Free Filtering**
   - Mutual Information
   - ANOVA F-test
   - Spearman Correlation
   - 투표 기반 상위 40개 선택

2. **Phase 2 - Stability Selection**
   - 20번의 다른 random split
   - TabPFN permutation importance
   - 안정적으로 선택되는 피처만 유지

3. **Phase 3 - Final Validation**
   - 5-Fold Cross Validation
   - 최적 피처 개수 결정

### 3. Model Training (`model_training.py`)
- **앙상블 전략**: 여러 random split으로 학습한 TabPFN 모델들
- **Consensus Voting**: unanimous / strong / majority 레벨
- **Best Model Selection**: Validation AUC 기준

### 4. Selection Strategy
```python
# Consensus 레벨별 선택
unanimous:  100% 모델이 Good 판정
strong:     85-90% 모델이 Good 판정
majority:   60%+ 모델이 Good 판정
```

## ⚙️ 설정 파라미터 (config.yaml)

### 경로 설정
```yaml
paths:
  train_data: "train.csv"
  test_data: "test.csv"
  output_file: "submission_tabpfn_optimized.csv"
```

### 모델 파라미터
```yaml
model:
  n_models: 10          # 앙상블 모델 개수 (추천: 5-15)
  n_estimators: 1       # TabPFN estimators (보통 1로 고정)
  model_version: "v12"  # v11 또는 v12
  val_size: 0.2         # Validation 비율
```

### 피처 설정
```yaml
features:
  n_features: 80        # 최종 선택 피처 개수 (추천: 60-100)
  pca_components: 10    # PCA 컴포넌트 개수
  
  anomaly:
    contamination: 0.1   # Isolation Forest 오염률
    n_estimators: 100    # Isolation Forest 트리 개수
```

### 선택 전략
```yaml
selection:
  max_samples: 200           # 최대 선택 샘플 (200 고정)
  max_probability: 1.0       # 확률 필터 (1.0 = 필터 없음)
  
  consensus_priority:        # 우선순위 (순서 중요!)
    - unanimous              # 먼저 unanimous 선택
    - strong                 # 그 다음 strong 선택
```

## 🔧 주요 모듈 설명

### data_loader.py
- `load_data()`: CSV 파일 로딩
- `preprocess_data()`: 결측치 처리 및 컬럼 정렬

### feature_engineering.py
- `add_pressure_features()`: 기본 압력 피처
- `extract_tire_features()`: 60+ 타이어 물리 피처
- `add_pca_features()`: PCA 변환
- `add_anomaly_features()`: 이상치 탐지 피처
- `engineer_features()`: 전체 파이프라인

### feature_selection.py
- `phase1_model_free_filtering()`: Model-free 초기 필터링
- `phase2_stability_selection()`: 안정성 기반 선택
- `phase3_final_validation()`: CV 기반 최종 검증
- `split_bias_free_feature_selection()`: 전체 3단계 실행

### model_training.py
- `find_optimal_threshold()`: Good Precision 최대화 threshold
- `train_ensemble_tabpfn()`: 앙상블 학습 및 투표

### utils.py
- `load_config()`: YAML 설정 로딩
- `create_submission()`: 제출 파일 생성
- `print_selection_summary()`: 결과 요약 출력

## 📈 실행 결과

실행 시 다음과 같은 정보가 출력됩니다:

1. **데이터 로딩**: 샘플 수, 타겟 분포
2. **피처 엔지니어링**: 생성된 피처 수
3. **피처 선택**: 3단계 과정 및 최종 선택 피처
4. **모델 학습**: 각 모델의 성능 (AUC, Precision, Recall)
5. **Consensus 결과**: 각 레벨별 샘플 수
6. **최종 선택**: 선택된 샘플 통계 및 구성

## 🎯 권장 실험 설정

### 빠른 테스트
```yaml
model:
  n_models: 3
features:
  n_features: 50
```

### 기본 설정 (추천)
```yaml
model:
  n_models: 10
features:
  n_features: 80
```

### 고성능 설정
```yaml
model:
  n_models: 15
features:
  n_features: 100
```

## 📝 원본 코드와의 차이점

### 개선 사항:
1. ✅ **모듈화**: 단일 파일 → 7개 모듈로 분리
2. ✅ **설정 외부화**: 하드코딩된 값 → YAML 설정 파일
3. ✅ **재사용성**: 함수별 독립 실행 가능
4. ✅ **유지보수**: 각 모듈이 명확한 역할 담당
5. ✅ **확장성**: 새로운 피처/모델 추가 용이

### 유지된 기능:
- 모든 피처 엔지니어링 로직
- 3단계 피처 선택 전략
- TabPFN 앙상블 방식
- Consensus 기반 선택 전략

## 🔍 디버깅 및 실험

### 특정 단계만 실행하기
```python
from feature_engineering import extract_tire_features
from utils import load_config

# 피처만 추출해서 확인
config = load_config('config.yaml')
features = extract_tire_features(train_df)
print(features.head())
```

### 설정값 테스트
```python
# 여러 n_features 값 테스트
for n_feat in [50, 80, 100]:
    # config.yaml의 n_features 변경 후
    # python main.py 실행
```

## 📞 문의 및 개선사항

코드 개선 제안이나 버그 리포트는 이슈로 등록해주세요!