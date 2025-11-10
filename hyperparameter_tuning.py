"""
RandomForest 하이퍼파라미터 튜닝
여러 파라미터 조합으로 학습/예측을 수행하고 정확도를 비교합니다.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from itertools import product
from datetime import datetime

from load_dataframe import load_by_data_fields
from random_forest_pipeline import enforce_schema_types, prepare_features_and_targets, calculate_rule_based_score, hybrid_predict


def train_predict_with_params(
    X: pd.DataFrame,
    y: pd.Series,
    df_original: pd.DataFrame,
    n_estimators: int,
    max_depth: int,
    min_samples_split: int,
    min_samples_leaf: int,
    max_features,
    use_hybrid: bool = True,
    weight_ml: float = 0.7,
    weight_rule: float = 0.3,
    random_state: int = 42
):
    """특정 파라미터로 학습 후 예측 정확도 반환"""
    # 층화 분할
    stratify_arg = None
    unique_vals, counts = np.unique(y, return_counts=True)
    if (counts.min() if len(counts) > 0 else 0) >= 2:
        stratify_arg = y

    X_tr, X_pred, y_tr, y_true = train_test_split(
        X, y, train_size=0.2, test_size=0.8, random_state=random_state, stratify=stratify_arg
    )
    
    train_idx = X_tr.index
    predict_idx = X_pred.index

    # RandomForest 학습
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth if max_depth > 0 else None,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        n_jobs=-1,
        random_state=random_state
    )
    clf.fit(X_tr, y_tr)
    
    # ML 예측
    y_pred_ml = clf.predict(X_pred)
    
    # 하이브리드 예측
    if use_hybrid and df_original is not None:
        rule_score_all = calculate_rule_based_score(df_original)
        rule_score_pred = rule_score_all.loc[predict_idx]
        y_pred_final = hybrid_predict(y_pred_ml, rule_score_pred, weight_ml, weight_rule)
    else:
        y_pred_final = y_pred_ml
    
    # 정확도 계산
    acc_ml = accuracy_score(y_true, y_pred_ml)
    acc_hybrid = accuracy_score(y_true, y_pred_final)
    
    return acc_ml, acc_hybrid


def run_hyperparameter_tuning(
    csv_path: str = "기업신용평가정보_합성데이터.csv",
    output_csv: str = "hyperparameter_results.csv",
    use_hybrid: bool = True
):
    """
    여러 하이퍼파라미터 조합으로 학습/예측하고 정확도를 비교합니다.
    
    Args:
        csv_path: 데이터 파일 경로
        output_csv: 결과 저장 CSV 파일명
        use_hybrid: 하이브리드 방식 사용 여부
    """
    print("=" * 60)
    print("하이퍼파라미터 튜닝 시작")
    print("=" * 60)
    
    # 데이터 로드
    print("\n데이터 로드 중...")
    df = load_by_data_fields(csv_path)
    df = enforce_schema_types(df)
    print(f"데이터 형태: {df.shape}")
    
    X, y1, y2, t1_name, t2_name = prepare_features_and_targets(df)
    print(f"특성 개수: {X.shape[1]}")
    print(f"타겟1: {t1_name}")
    print(f"타겟2: {t2_name}")
    
    # 파라미터 그리드 정의
    param_grid = {
        'n_estimators': [50, 100, 150, 200, 250, 300, 350, 400, 500],
        'max_depth': [3, 5, 7, 10, 12, 15, 20, 25, 30, 0],  # 0 = None (무제한)
        'min_samples_split': [2, 3, 5, 7, 10, 15, 20],
        'min_samples_leaf': [1, 2, 4, 6, 8, 10],
        'max_features': ['sqrt', 'log2', None]  # None = 모든 특성
    }
    
    print("\n파라미터 그리드:")
    for key, values in param_grid.items():
        print(f"  {key}: {values}")
    
    # 전체 조합 수 계산
    total_combinations = 1
    for values in param_grid.values():
        total_combinations *= len(values)
    print(f"\n총 조합 수: {total_combinations}")
    
    # 결과 저장 리스트
    results = []
    
    # 모든 파라미터 조합 시도
    param_combinations = list(product(
        param_grid['n_estimators'],
        param_grid['max_depth'],
        param_grid['min_samples_split'],
        param_grid['min_samples_leaf'],
        param_grid['max_features']
    ))
    
    print("\n" + "=" * 60)
    print("파라미터 튜닝 진행 중...")
    print("=" * 60)
    
    for idx, (n_est, max_d, min_split, min_leaf, max_feat) in enumerate(param_combinations, 1):
        print(f"\n[{idx}/{total_combinations}] 조합: n_estimators={n_est}, max_depth={max_d if max_d > 0 else 'None'}, min_samples_split={min_split}, min_samples_leaf={min_leaf}, max_features={max_feat}")
        
        try:
            # 타겟1 학습/예측
            acc_t1_ml, acc_t1_hybrid = train_predict_with_params(
                X, y1, df, n_est, max_d, min_split, min_leaf, max_feat, use_hybrid
            )
            
            # 타겟2 학습/예측
            acc_t2_ml, acc_t2_hybrid = train_predict_with_params(
                X, y2, df, n_est, max_d, min_split, min_leaf, max_feat, use_hybrid
            )
            
            print(f"  타겟1 정확도 (ML: {acc_t1_ml:.4f}, 하이브리드: {acc_t1_hybrid:.4f})")
            print(f"  타겟2 정확도 (ML: {acc_t2_ml:.4f}, 하이브리드: {acc_t2_hybrid:.4f})")
            
            # 결과 저장
            results.append({
                'n_estimators': n_est,
                'max_depth': max_d if max_d > 0 else 'None',
                'min_samples_split': min_split,
                'min_samples_leaf': min_leaf,
                'max_features': str(max_feat),
                '타겟1_ML_정확도': acc_t1_ml,
                '타겟1_하이브리드_정확도': acc_t1_hybrid,
                '타겟2_ML_정확도': acc_t2_ml,
                '타겟2_하이브리드_정확도': acc_t2_hybrid,
                '타겟1_개선율': acc_t1_hybrid - acc_t1_ml,
                '타겟2_개선율': acc_t2_hybrid - acc_t2_ml
            })
            
        except Exception as e:
            print(f"  오류 발생: {e}")
            results.append({
                'n_estimators': n_est,
                'max_depth': max_d if max_d > 0 else 'None',
                'min_samples_split': min_split,
                'min_samples_leaf': min_leaf,
                'max_features': str(max_feat),
                '타겟1_ML_정확도': None,
                '타겟1_하이브리드_정확도': None,
                '타겟2_ML_정확도': None,
                '타겟2_하이브리드_정확도': None,
                '타겟1_개선율': None,
                '타겟2_개선율': None,
                '오류': str(e)
            })
    
    # 결과를 DataFrame으로 변환
    results_df = pd.DataFrame(results)
    
    # 정확도 기준으로 정렬 (타겟1 하이브리드 정확도 내림차순)
    results_df = results_df.sort_values('타겟1_하이브리드_정확도', ascending=False, na_position='last')
    
    # 저장
    results_df.to_csv(output_csv, index=False, encoding='utf-8')
    print("\n" + "=" * 60)
    print(f"결과 저장 완료: {output_csv}")
    print("=" * 60)
    
    # 상위 10개 출력
    print("\n상위 10개 파라미터 조합 (타겟1 하이브리드 정확도 기준):")
    print(results_df.head(10).to_string(index=False))
    
    # 최적 파라미터 출력
    if len(results_df) > 0 and results_df.iloc[0]['타겟1_하이브리드_정확도'] is not None:
        best = results_df.iloc[0]
        print("\n" + "=" * 60)
        print("최적 파라미터 조합")
        print("=" * 60)
        print(f"n_estimators: {best['n_estimators']}")
        print(f"max_depth: {best['max_depth']}")
        print(f"min_samples_split: {best['min_samples_split']}")
        print(f"min_samples_leaf: {best['min_samples_leaf']}")
        print(f"max_features: {best['max_features']}")
        print(f"\n타겟1 하이브리드 정확도: {best['타겟1_하이브리드_정확도']:.4f}")
        print(f"타겟2 하이브리드 정확도: {best['타겟2_하이브리드_정확도']:.4f}")
    
    return results_df


if __name__ == "__main__":
    results_df = run_hyperparameter_tuning(
        csv_path="기업신용평가정보_합성데이터.csv",
        output_csv="hyperparameter_results.csv",
        use_hybrid=True
    )

