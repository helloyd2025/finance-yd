"""
랜덤 포레스트 파이프라인
 - 데이터 로드: load_by_data_fields 사용 (따옴표 기준 컬럼 파싱)
 - 특성: 숫자형 컬럼 (마지막 2개 타겟 컬럼 제외)
 - 타겟: 마지막 2개 컬럼
 - 학습/검증 분리 후 각 타겟에 대해 RandomForestClassifier 학습/평가
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score

from load_dataframe import load_by_data_fields
from sklearn.linear_model import LogisticRegression

CUML_AVAILABLE = False
try:
    import cudf  # type: ignore
    from cuml.ensemble import RandomForestClassifier as cuRF  # type: ignore
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False

GRADE_FEATURE_COLUMN = "기업신용평가등급(구간화)"


def calculate_rule_based_score(df: pd.DataFrame) -> pd.Series:
    """
    규칙 기반 신용평가 점수 계산 (1~10 등급)
    주요 재무 지표를 기반으로 점수를 산정합니다.
    """
    score = pd.Series(5.0, index=df.index)  # 기본 점수 5 (중간 등급)
    
    # 1. 부채비율 (낮을수록 좋음, 300% 이상이면 등급 하향)
    if '재무비율_부채비율' in df.columns:
        debt_ratio = pd.to_numeric(df['재무비율_부채비율'], errors='coerce')
        score += np.where(debt_ratio < 100, 1.0, 0)  # 100% 미만: +1점
        score += np.where((debt_ratio >= 300), -2.0, 0)  # 300% 이상: -2점
        score += np.where((debt_ratio >= 500), -1.0, 0)  # 500% 이상: 추가 -1점
    
    # 2. ROE (높을수록 좋음, 10% 이상이면 등급 상향)
    if '재무비율_자기자본이익률(ROE)' in df.columns:
        roe = pd.to_numeric(df['재무비율_자기자본이익률(ROE)'], errors='coerce')
        score += np.where(roe >= 10, 1.5, 0)  # 10% 이상: +1.5점
        score += np.where(roe >= 20, 0.5, 0)  # 20% 이상: 추가 +0.5점
        score += np.where(roe < 0, -1.0, 0)  # 음수: -1점
    
    # 3. 유동비율 (100~200% 적정 범위)
    if '재무비율_유동비율' in df.columns:
        current_ratio = pd.to_numeric(df['재무비율_유동비율'], errors='coerce')
        score += np.where((current_ratio >= 100) & (current_ratio <= 200), 0.5, 0)  # 적정 범위: +0.5점
        score += np.where(current_ratio < 50, -1.0, 0)  # 50% 미만: -1점
    
    # 4. 영업이익률 (높을수록 좋음)
    if '재무비율_영업이익율' in df.columns:
        operating_margin = pd.to_numeric(df['재무비율_영업이익율'], errors='coerce')
        score += np.where(operating_margin >= 5, 1.0, 0)  # 5% 이상: +1점
        score += np.where(operating_margin < 0, -1.0, 0)  # 음수: -1점
    
    # 5. 당기순이익 (양수면 좋음)
    if '당기순이익' in df.columns:
        net_income = pd.to_numeric(df['당기순이익'], errors='coerce')
        score += np.where(net_income > 0, 0.5, 0)  # 양수: +0.5점
        score += np.where(net_income < -1000000000, -1.0, 0)  # 큰 손실: -1점
    
    # 6. 공공정보 건수 (없을수록 좋음)
    if '신용도판단정보공공정보건수(CIS)(5년내발생)(해제포함)' in df.columns:
        public_info = pd.to_numeric(df['신용도판단정보공공정보건수(CIS)(5년내발생)(해제포함)'], errors='coerce')
        score += np.where(public_info == 0, 0.5, 0)  # 건수 0: +0.5점
        score += np.where(public_info >= 5, -1.0, 0)  # 5건 이상: -1점
    
    # 7. EBITDA/금융비용 (높을수록 좋음, 이자보상능력)
    if 'EBITDA/금융비용' in df.columns:
        ebitda_interest = pd.to_numeric(df['EBITDA/금융비용'], errors='coerce')
        score += np.where(ebitda_interest >= 2, 0.5, 0)  # 2 이상: +0.5점
        score += np.where(ebitda_interest < 1, -0.5, 0)  # 1 미만: -0.5점
    
    # 점수를 1~10 범위로 제한 및 반올림
    score = np.clip(score, 1, 10)
    score = np.round(score).astype(int)
    
    return score


def hybrid_predict(
    ml_pred: np.ndarray,
    rule_score: pd.Series,
    weight_ml: float = 0.7,
    weight_rule: float = 0.3
) -> np.ndarray:
    """
    머신러닝 예측값과 규칙 기반 점수를 결합
    
    Args:
        ml_pred: ML 예측값 (1~10 등급)
        rule_score: 규칙 기반 점수 (1~10 등급)
        weight_ml: ML 가중치 (기본 0.7)
        weight_rule: 규칙 가중치 (기본 0.3)
    
    Returns:
        결합된 예측값 (1~10 등급)
    """
    # 가중 평균 후 반올림
    combined = (ml_pred * weight_ml + rule_score.values * weight_rule)
    combined = np.round(np.clip(combined, 1, 10)).astype(int)
    return combined


STRING_COLUMNS = [
    "기준년월", "가명식별자", "업종(중분류)", "외감구분", "설립일자", "주소지시군구",
    "상장일자", "상장폐지일자",
]

# 숫자형으로 강제할 대표 컬럼들 (일부만 발췌, 나머지는 자동 numeric 변환에서 커버)
NUMERIC_PRIORITY_COLUMNS = [
    "종업원수", "유동자산", "비유동자산", "당좌자산", "재고자산", "유형자산", "재공품", "현금",
    "현금등가물", "상품유가증권", "현금성자산", "매출채권", "매출채권(전기)", "매출채권처분손실(당기)",
    "무형자산", "투자자산", "자산총계", "자산총계(전기)", "유동부채", "단기차입금", "차입금",
    "매입채무", "비유동부채", "부채총계", "자기자본(납입자본금)", "자본잉여금", "납입자본",
    "이익잉여금", "자본조정", "기타포괄손익누계액", "유보금", "자본총계", "전기자본총계",
    "매출액", "전기매출액", "매출원가", "매출총이익", "판매비와관리비", "법인세비용차감전순이익",
    "전기법인세차감전순이익", "법인세", "계속사업이익", "중단산업손익", "금융비용", "영업손익",
    "전기영업이익", "영업외수익", "영업외비용", "법인세차감전순이익", "당기순이익",
    "당기순이익(전기)", "현금흐름", "영업활동현금흐름", "투자활동현금흐름", "재무활동현금흐름",
    "부채상환계수", "영업이익이자보상배율", "이자비용", "사채이자(당기)", "이자보상배율",
    "적립금비율", "EBIT", "EBITDA", "청산가치율", "청산가치", "순운전자본", "순차입금",
    "재무비율_총자산증가율", "재무비율_부채비율", "재무비율_자기자본비율", "재무비율_유동비율",
    "재무비율_차입금의존도", "재무비율_매출액증가율", "재무비율_영업이익율", "재무비율_당기순이익율",
    "재무비율_매출원가율", "재무비율_판관비율", "재무비율_자기자본이익률(ROE)", "재무비율_매출채권회전율",
    "재무비율_재고자산회전율", "재무비율_매입채무회전율", "재무비율_총자산회전율", "재무비율_총자산순이익률",
    "재무비율_유동자산증가율", "재무비율_유형자산증가율", "단기차입금의존도", "당좌비율", "순차입금비율",
    "순운전자본회전율", "총자본회전율", "자기자본순이익율", "매출총이익율", "EBITDA마진율",
    "영업이익증가율", "당기순이익증가율", "EBITDA증가율", "OCF/매출액비용", "부채상환계수.1",
    "차입금/EBITDA", "EBITDA/금융비용", "소유건축물건수", "소유건축물실거래가합계",
    "신용도판단공공정보건수(CIS)(5년내발생)(해제포함)", "신용도판단정보공공정보건수(CIS)(미해제)",
    "공공정보(국세,지방세,관세체납)건수(CIS)(미해제)", "공공정보(국세,지방세,관세체납)건수(CIS)(5년내발생)",
    "공공정보(국세,지방세,관세체납,고용산재체납)건수(CIS)(미해제)", "공공정보(국세,지방세,관세체납,고용산재체납)건수(CIS)(5년내발생)",
    "신용도판단정보공공정보최근발생일자로부터경과일수(CIS)(해제,삭제)",
    "신용도판단정보공공정보최근해제일자로부터경과일수(CIS)(해제,삭제)",
]

if GRADE_FEATURE_COLUMN not in NUMERIC_PRIORITY_COLUMNS:
    NUMERIC_PRIORITY_COLUMNS.append(GRADE_FEATURE_COLUMN)


def enforce_schema_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 문자열 강제
    for col in STRING_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # 숫자형 강제 (우선순위 컬럼)
    for col in NUMERIC_PRIORITY_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 전역적으로 숫자 해석 가능한 컬럼은 숫자로 변환 시도 (타겟 2개 제외)
    last2 = list(df.columns[-2:])
    for col in df.columns:
        if col in last2:
            continue
        # 이미 숫자면 패스, 아니면 변환 시도
        if not np.issubdtype(df[col].dtype, np.number):
            try:
                converted = pd.to_numeric(df[col], errors='raise')
                if np.issubdtype(converted.dtype, np.number):
                    df[col] = converted
            except Exception:
                # 변환 불가 시 원본 유지
                pass

    # 특수 sentinel 값을 NaN 처리 (999999999) - 경과일수 2개 컬럼
    for sentinel_col in [
        "신용도판단정보공공정보최근발생일자로부터경과일수(CIS)(해제,삭제)",
        "신용도판단정보공공정보최근해제일자로부터경과일수(CIS)(해제,삭제)",
    ]:
        if sentinel_col in df.columns:
            df[sentinel_col] = pd.to_numeric(df[sentinel_col], errors='coerce')
            df.loc[df[sentinel_col] == 999999999, sentinel_col] = np.nan

    return df


def prepare_features_and_targets(df: pd.DataFrame):
    if df.shape[1] < 3:
        raise ValueError("컬럼 수가 부족합니다. 최소 3개 필요 (특성 + 타겟2)")

    # 마지막 2개 컬럼을 타겟으로 지정
    target_col_1 = df.columns[-2]
    target_col_2 = df.columns[-1]

    # 숫자형 특성만 사용하고, 타겟 컬럼 제외
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    for c in [target_col_1, target_col_2]:
        if c in numeric_df.columns:
            numeric_df = numeric_df.drop(columns=[c])

    X = numeric_df
    y1 = pd.to_numeric(df[target_col_1], errors='coerce')
    y2 = pd.to_numeric(df[target_col_2], errors='coerce')

    # 타겟 결측 제거 (두 타겟 모두 존재하는 행만 사용)
    valid_mask = (~y1.isna()) & (~y2.isna())
    X = X.loc[valid_mask]
    y1 = y1.loc[valid_mask].astype(int)
    y2 = y2.loc[valid_mask].astype(int)

    return X, y1, y2, target_col_1, target_col_2


def train_and_evaluate_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    df_original: pd.DataFrame = None,
    use_hybrid: bool = True,
    weight_ml: float = 0.7,
    weight_rule: float = 0.3,
    random_state: int = 42,
    use_cuml: bool = False,
):
    """
    RandomForest 학습 및 평가 (하이브리드 방식 지원)
    
    Args:
        X: 특성 데이터
        y: 타겟 데이터
        df_original: 원본 DataFrame (규칙 기반 점수 계산용)
        use_hybrid: 하이브리드 방식 사용 여부
        weight_ml: ML 가중치
        weight_rule: 규칙 기반 가중치
        random_state: 랜덤 시드
    """
    # 층화 분할 가능한지 확인 (모든 클래스 2개 이상)
    stratify_arg = None
    unique_vals, counts = np.unique(y, return_counts=True)
    if (counts.min() if len(counts) > 0 else 0) >= 2:
        stratify_arg = y

    use_gpu = use_cuml and CUML_AVAILABLE
    if use_cuml and not CUML_AVAILABLE:
        print("⚠️ cuML을 사용할 수 없어 scikit-learn CPU 모드로 대체합니다.")

    if use_gpu:
        X_train_pd, X_test_pd, y_train_pd, y_test_pd = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=stratify_arg
        )
        train_idx = X_train_pd.index
        test_idx = X_test_pd.index

        clf = cuRF(
            n_estimators=300,
            max_depth=None,
            random_state=random_state
        )
        X_train_gpu = cudf.from_pandas(X_train_pd)
        X_test_gpu = cudf.from_pandas(X_test_pd)
        y_train_gpu = cudf.Series(y_train_pd.values)

        clf.fit(X_train_gpu, y_train_gpu)

        y_pred_train_ml = clf.predict(X_train_gpu).to_pandas().to_numpy()
        y_pred_test_ml = clf.predict(X_test_gpu).to_pandas().to_numpy()

        X_train = X_train_pd
        X_test = X_test_pd
        y_train = y_train_pd
        y_test = y_test_pd
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=stratify_arg
        )
        
        train_idx = X_train.index
        test_idx = X_test.index

        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            n_jobs=-1,
            random_state=random_state
        )
        clf.fit(X_train, y_train)

        # ML 예측
        y_pred_train_ml = clf.predict(X_train)
        y_pred_test_ml = clf.predict(X_test)
    
    # 하이브리드 예측 (규칙 기반 점수와 결합)
    y_pred_train_final = y_pred_train_ml.copy()
    y_pred_test_final = y_pred_test_ml.copy()
    rule_score_train = None
    rule_score_test = None
    
    if use_hybrid and df_original is not None:
        # 규칙 기반 점수 계산
        rule_score_all = calculate_rule_based_score(df_original)
        rule_score_train = rule_score_all.loc[train_idx]
        rule_score_test = rule_score_all.loc[test_idx]
        
        # 하이브리드 예측
        y_pred_train_final = hybrid_predict(y_pred_train_ml, rule_score_train, weight_ml, weight_rule)
        y_pred_test_final = hybrid_predict(y_pred_test_ml, rule_score_test, weight_ml, weight_rule)

    # 성능 (하이브리드 결과 기준)
    acc_train_ml = accuracy_score(y_train, y_pred_train_ml)
    acc_test_ml = accuracy_score(y_test, y_pred_test_ml)
    acc_train_hybrid = accuracy_score(y_train, y_pred_train_final)
    acc_test_hybrid = accuracy_score(y_test, y_pred_test_final)
    f1_test_ml = f1_score(y_test, y_pred_test_ml, average="weighted", zero_division=0)
    recall_test_ml = recall_score(y_test, y_pred_test_ml, average="weighted", zero_division=0)
    f1_test_hybrid = f1_score(y_test, y_pred_test_final, average="weighted", zero_division=0)
    recall_test_hybrid = recall_score(y_test, y_pred_test_final, average="weighted", zero_division=0)

    return {
        "model": clf,
        "splits": (X_train, X_test, y_train, y_test),
        "pred_train_ml": y_pred_train_ml,
        "pred_test_ml": y_pred_test_ml,
        "pred_train_hybrid": y_pred_train_final,
        "pred_test_hybrid": y_pred_test_final,
        "rule_score_train": rule_score_train,
        "rule_score_test": rule_score_test,
        "acc_train_ml": acc_train_ml,
        "acc_test_ml": acc_test_ml,
        "acc_train_hybrid": acc_train_hybrid,
        "acc_test_hybrid": acc_test_hybrid,
        "f1_test_ml": f1_test_ml,
        "recall_test_ml": recall_test_ml,
        "f1_test_hybrid": f1_test_hybrid,
        "recall_test_hybrid": recall_test_hybrid,
        "report_test_ml": classification_report(y_test, y_pred_test_ml, zero_division=0),
        "report_test_hybrid": classification_report(y_test, y_pred_test_final, zero_division=0)
    }


def train20_predict80(
    X: pd.DataFrame,
    y: pd.Series,
    df_original: pd.DataFrame = None,
    use_hybrid: bool = False,
    weight_ml: float = 0.7,
    weight_rule: float = 0.3,
    random_state: int = 42,
    use_cuml: bool = False,
):
    """
    20% 학습, 80% 예측 (하이브리드 방식 지원)
    """
    # 20% 학습, 80% 예측용 분할
    stratify_arg = None
    unique_vals, counts = np.unique(y, return_counts=True)
    if (counts.min() if len(counts) > 0 else 0) >= 2:
        stratify_arg = y

    use_gpu = use_cuml and CUML_AVAILABLE
    if use_cuml and not CUML_AVAILABLE:
        print("⚠️ cuML을 사용할 수 없어 scikit-learn CPU 모드로 대체합니다.")

    if use_gpu:
        X_tr_pd, X_pred_pd, y_tr_pd, y_true_pred_pd = train_test_split(
            X, y, train_size=0.2, test_size=0.8, random_state=random_state, stratify=stratify_arg
        )
        train_idx = X_tr_pd.index
        predict_idx = X_pred_pd.index

        clf = cuRF(
            n_estimators=400,
            max_depth=None,
            random_state=random_state
        )
        X_tr_gpu = cudf.from_pandas(X_tr_pd)
        X_pred_gpu = cudf.from_pandas(X_pred_pd)
        y_tr_gpu = cudf.Series(y_tr_pd.values)

        clf.fit(X_tr_gpu, y_tr_gpu)
        y_pred_ml = clf.predict(X_pred_gpu).to_pandas().to_numpy()

        X_tr = X_tr_pd
        X_pred = X_pred_pd
        y_tr = y_tr_pd
        y_true_pred = y_true_pred_pd
    else:
        X_tr, X_pred, y_tr, y_true_pred = train_test_split(
            X, y, train_size=0.2, test_size=0.8, random_state=random_state, stratify=stratify_arg
        )
        
        train_idx = X_tr.index
        predict_idx = X_pred.index

        clf = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            n_jobs=-1,
            random_state=random_state
        )
        clf.fit(X_tr, y_tr)
        
        # ML 예측
        y_pred_ml = clf.predict(X_pred)
    y_pred_final = y_pred_ml
    
    # 하이브리드 예측
    if use_hybrid and df_original is not None:
        rule_score_all = calculate_rule_based_score(df_original)
        rule_score_pred = rule_score_all.loc[predict_idx]
        y_pred_final = hybrid_predict(y_pred_ml, rule_score_pred, weight_ml, weight_rule)

    return {
        "model": clf,
        "train_indices": train_idx,
        "predict_indices": predict_idx,
        "y_true_predict": y_true_pred,
        "y_pred_ml": y_pred_ml,
        "y_pred_hybrid": y_pred_final,
    }


def run_pipeline(csv_path: str = "기업신용평가정보_합성데이터.csv", use_hybrid: bool = True, use_cuml: bool = False):
    print("=" * 60)
    print("데이터 로드")
    print("=" * 60)
    df = load_by_data_fields(csv_path)
    df = enforce_schema_types(df)
    print(f"데이터 형태: {df.shape}")

    print("\n" + "=" * 60)
    print("특성/타겟 분리")
    print("=" * 60)
    X, y1, y2, t1_name, t2_name = prepare_features_and_targets(df)
    print(f"특성 개수: {X.shape[1]}")
    print(f"타겟1: {t1_name}, 타겟2: {t2_name}")
    
    if use_hybrid:
        print("\n→ 하이브리드 방식 사용 (규칙 기반 점수 + 머신러닝 예측)")

    print("\n" + "=" * 60)
    print(f"RandomForest 학습/평가 - 타겟1 ({t1_name})")
    print("=" * 60)
    res1 = train_and_evaluate_random_forest(X, y1, df_original=df, use_hybrid=use_hybrid, use_cuml=use_cuml)
    
    if use_hybrid:
        print(f"[ML만] 학습 정확도: {res1['acc_train_ml']:.4f}, 검증 정확도: {res1['acc_test_ml']:.4f}")
        print(f"[하이브리드] 학습 정확도: {res1['acc_train_hybrid']:.4f}, 검증 정확도: {res1['acc_test_hybrid']:.4f}")
        print(f"[ML만] F1-score: {res1['f1_test_ml']:.4f}, Recall: {res1['recall_test_ml']:.4f}")
        print(f"[하이브리드] F1-score: {res1['f1_test_hybrid']:.4f}, Recall: {res1['recall_test_hybrid']:.4f}")
        print("\n[하이브리드] 검증 리포트:\n" + res1["report_test_hybrid"])
    else:
        print(f"학습 정확도: {res1['acc_train_ml']:.4f}")
        print(f"검증 정확도: {res1['acc_test_ml']:.4f}")
        print(f"F1-score: {res1['f1_test_ml']:.4f}, Recall: {res1['recall_test_ml']:.4f}")
        print("검증 리포트:\n" + res1["report_test_ml"]) 

    print("\n샘플 비교(검증 상위 10개):")
    X_train, X_test, y1_train, y1_test = res1["splits"]
    pred_key = 'pred_test_hybrid' if use_hybrid else 'pred_test_ml'
    for i in range(min(10, len(y1_test))):
        pred_val = res1[pred_key][i]
        actual_val = y1_test.iloc[i]
        rule_val = res1.get('rule_score_test', None)
        rule_str = f", 규칙점수={rule_val.iloc[i]}" if rule_val is not None else ""
        print(f"  [{i}] 예측={pred_val}  실제={actual_val}{rule_str}")

    print("\n" + "=" * 60)
    print(f"RandomForest 학습/평가 - 타겟2 ({t2_name})")
    print("=" * 60)
    res2 = train_and_evaluate_random_forest(X, y2, df_original=df, use_hybrid=use_hybrid, use_cuml=use_cuml)
    
    if use_hybrid:
        print(f"[ML만] 학습 정확도: {res2['acc_train_ml']:.4f}, 검증 정확도: {res2['acc_test_ml']:.4f}")
        print(f"[하이브리드] 학습 정확도: {res2['acc_train_hybrid']:.4f}, 검증 정확도: {res2['acc_test_hybrid']:.4f}")
        print(f"[ML만] F1-score: {res2['f1_test_ml']:.4f}, Recall: {res2['recall_test_ml']:.4f}")
        print(f"[하이브리드] F1-score: {res2['f1_test_hybrid']:.4f}, Recall: {res2['recall_test_hybrid']:.4f}")
        print("\n[하이브리드] 검증 리포트:\n" + res2["report_test_hybrid"])
    else:
        print(f"학습 정확도: {res2['acc_train_ml']:.4f}")
        print(f"검증 정확도: {res2['acc_test_ml']:.4f}")
        print(f"F1-score: {res2['f1_test_ml']:.4f}, Recall: {res2['recall_test_ml']:.4f}")
        print("검증 리포트:\n" + res2["report_test_ml"]) 

    print("\n샘플 비교(검증 상위 10개):")
    X_train2, X_test2, y2_train, y2_test = res2["splits"]
    pred_key = 'pred_test_hybrid' if use_hybrid else 'pred_test_ml'
    for i in range(min(10, len(y2_test))):
        pred_val = res2[pred_key][i]
        actual_val = y2_test.iloc[i]
        print(f"  [{i}] 예측={pred_val}  실제={actual_val}")

    return {
        "target1_name": t1_name,
        "target2_name": t2_name,
        "result_target1": res1,
        "result_target2": res2,
    }


def run_pipeline_predict80(
    csv_path: str = "기업신용평가정보_합성데이터.csv", 
    save_csv: bool = True, 
    out_path: str = "predictions_80.csv",
    use_hybrid: bool = False,
    use_cuml: bool = False
):
    print("=" * 60)
    print("데이터 로드")
    print("=" * 60)
    df = load_by_data_fields(csv_path)
    df = enforce_schema_types(df)
    print(f"데이터 형태: {df.shape}")
    
    if use_hybrid:
        print("\n→ 하이브리드 방식 사용 (규칙 기반 점수 + 머신러닝 예측)")

    X, y1, y2, t1_name, t2_name = prepare_features_and_targets(df)

    print("\n" + "=" * 60)
    print(f"20% 학습 / 80% 예측 분할 및 예측 - 타겟1 ({t1_name})")
    print("=" * 60)
    r1 = train20_predict80(X, y1, df_original=df, use_hybrid=use_hybrid, use_cuml=use_cuml)
    print(f"학습 샘플: {len(r1['train_indices'])}, 예측 샘플: {len(r1['predict_indices'])}")

    print("\n" + "=" * 60)
    print(f"20% 학습 / 80% 예측 분할 및 예측 - 타겟2 ({t2_name})")
    print("=" * 60)
    r2 = train20_predict80(X, y2, df_original=df, use_hybrid=use_hybrid, use_cuml=use_cuml)
    print(f"학습 샘플: {len(r2['train_indices'])}, 예측 샘플: {len(r2['predict_indices'])}")

    # 예측 결과를 하나의 DataFrame으로 병합 (공통 예측 인덱스 교집합 기준)
    # 주의: predict_indices는 원본 DataFrame의 인덱스입니다.
    # 원본 CSV 행 번호로 변환하려면 +2를 해야 합니다 (행 0: BOM?, 행 1: 헤더).
    idx_common = pd.Index(r1["predict_indices"]).intersection(pd.Index(r2["predict_indices"]))
    pred_df = pd.DataFrame(index=idx_common)
    
    pred_key = 'y_pred_hybrid' if use_hybrid else 'y_pred_ml'
    # 예측값 먼저 추가
    pred_df[f"pred_{t1_name}"] = pd.Series(r1[pred_key], index=r1["predict_indices"]).reindex(idx_common)
    pred_df[f"pred_{t2_name}"] = pd.Series(r2[pred_key], index=r2["predict_indices"]).reindex(idx_common)
    
    if use_hybrid:
        pred_df[f"pred_{t1_name}_ml_only"] = pd.Series(r1["y_pred_ml"], index=r1["predict_indices"]).reindex(idx_common)
        pred_df[f"pred_{t2_name}_ml_only"] = pd.Series(r2["y_pred_ml"], index=r2["predict_indices"]).reindex(idx_common)
    
    # 실제값은 맨 뒤에 추가
    pred_df[f"true_{t1_name}"] = pd.Series(r1["y_true_predict"].values, index=r1["predict_indices"]).reindex(idx_common)
    pred_df[f"true_{t2_name}"] = pd.Series(r2["y_true_predict"].values, index=r2["predict_indices"]).reindex(idx_common)

    # 원본 CSV 행 번호는 헤더+1을 고려하여 +2 보정
    pred_df.index = pred_df.index + 2
    pred_df.index.name = "원본CSV행번호"

    # 예측 정확도 계산 및 출력
    print("\n" + "=" * 60)
    print("예측 정확도")
    print("=" * 60)
    
    # 타겟1 정확도
    true_t1 = pred_df[f"true_{t1_name}"].values
    pred_t1 = pred_df[f"pred_{t1_name}"].values
    acc_t1 = accuracy_score(true_t1, pred_t1)
    print(f"\n타겟1 ({t1_name}):")
    main_label = "하이브리드" if use_hybrid else "ML"
    print(f"  {main_label} 예측 정확도: {acc_t1:.4f}")
    f1_t1 = f1_score(true_t1, pred_t1, average="weighted", zero_division=0)
    recall_t1 = recall_score(true_t1, pred_t1, average="weighted", zero_division=0)
    print(f"  {main_label} F1-score: {f1_t1:.4f}, Recall: {recall_t1:.4f}")
    
    if use_hybrid :
        pred_t1_ml = pred_df[f"pred_{t1_name}_ml_only"].values
        acc_t1_ml = accuracy_score(true_t1, pred_t1_ml)
        print(f"  ML만 예측 정확도: {acc_t1_ml:.4f}")
        f1_t1_ml = f1_score(true_t1, pred_t1_ml, average="weighted", zero_division=0)
        recall_t1_ml = recall_score(true_t1, pred_t1_ml, average="weighted", zero_division=0)
        print(f"  ML만 F1-score: {f1_t1_ml:.4f}, Recall: {recall_t1_ml:.4f}")
    
    # 타겟2 정확도
    true_t2 = pred_df[f"true_{t2_name}"].values
    pred_t2 = pred_df[f"pred_{t2_name}"].values
    acc_t2 = accuracy_score(true_t2, pred_t2)
    print(f"\n타겟2 ({t2_name}):")
    print(f"  {main_label} 예측 정확도: {acc_t2:.4f}")
    f1_t2 = f1_score(true_t2, pred_t2, average="weighted", zero_division=0)
    recall_t2 = recall_score(true_t2, pred_t2, average="weighted", zero_division=0)
    print(f"  {main_label} F1-score: {f1_t2:.4f}, Recall: {recall_t2:.4f}")
    
    if use_hybrid:
        pred_t2_ml = pred_df[f"pred_{t2_name}_ml_only"].values
        acc_t2_ml = accuracy_score(true_t2, pred_t2_ml)
        print(f"  ML만 예측 정확도: {acc_t2_ml:.4f}")
        f1_t2_ml = f1_score(true_t2, pred_t2_ml, average="weighted", zero_division=0)
        recall_t2_ml = recall_score(true_t2, pred_t2_ml, average="weighted", zero_division=0)
        print(f"  ML만 F1-score: {f1_t2_ml:.4f}, Recall: {recall_t2_ml:.4f}")

    if save_csv:
        pred_df.to_csv(out_path, encoding="utf-8", index=True)
        print(f"\n예측 결과 저장: {out_path} (행수: {len(pred_df)})")

    print("\n샘플 출력(상위 10행):")
    print(pred_df.head(10))

    return {
        "target1_name": t1_name,
        "target2_name": t2_name,
        "predictions": pred_df,
    }

if __name__ == "__main__":
    # run_pipeline()
    run_pipeline_predict80()


