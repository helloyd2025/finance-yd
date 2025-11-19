"""
모델 초기화 및 외부 데이터 예측을 담당하는 헬퍼 모듈.

사용 시나리오
--------------
1) 초기 실행:
    python model_manager.py init \
        --original-csv 기업신용평가정보_합성데이터.csv \
        --model-path artifacts/rf_credit_models.joblib \
        --predictions-path predictions_80.csv \
        --hybrid

   - 내부적으로 `run_pipeline_predict80`를 실행해 학습/예측 리포트를 남깁니다.
   - 전체 원본 데이터를 이용해 RandomForest 모델 2개(타겟1/타겟2)를 다시 학습하고 저장합니다.

2) 외부 데이터 예측:
    python model_manager.py predict \
        --input-csv new_data.csv \
        --model-path artifacts/rf_credit_models.joblib \
        --output-csv external_predictions.csv

필요에 따라 함수 단위로 import 하여 사용할 수도 있습니다.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from random_forest_pipeline import (
    calculate_rule_based_score,
    enforce_schema_types,
    hybrid_predict,
    prepare_features_and_targets,
    run_pipeline_predict80,
)
from load_dataframe import load_by_data_fields

DEFAULT_ORIGINAL_CSV = "기업신용평가정보_합성데이터.csv"
DEFAULT_MODEL_PATH = Path("artifacts/rf_credit_models.joblib")
DEFAULT_PREDICTION_PATH = Path("predictions_80.csv")


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _fit_full_models(
    df: pd.DataFrame,
    random_state: int = 42,
) -> Tuple[RandomForestClassifier, RandomForestClassifier, str, str, pd.DataFrame]:
    """
    전체 데이터를 활용해 두 개의 RandomForestClassifier를 학습합니다.

    Returns
    -------
    (model_t1, model_t2, target1_name, target2_name, feature_frame)
    """
    X, y1, y2, t1_name, t2_name = prepare_features_and_targets(df)

    params = dict(
        n_estimators=400,
        max_depth=None,
        n_jobs=-1,
        random_state=random_state,
    )

    model_t1 = RandomForestClassifier(**params)
    model_t2 = RandomForestClassifier(**params)

    model_t1.fit(X, y1)
    model_t2.fit(X, y2)
    return model_t1, model_t2, t1_name, t2_name, X


def initialize_models(
    original_csv: str = DEFAULT_ORIGINAL_CSV,
    model_path: Path = DEFAULT_MODEL_PATH,
    predictions_path: Path = DEFAULT_PREDICTION_PATH,
    use_hybrid: bool = True,
    use_cuml: bool = False,
    random_state: int = 42,
) -> Dict[str, object]:
    """
    원본 CSV를 이용해 모델을 학습하고 저장합니다.
    동시에 `run_pipeline_predict80` 결과를 재활용해 학습/예측 리포트를 생성합니다.

    Returns
    -------
    metadata(dict): 저장된 메타데이터
    """
    # 1) 리포트 및 내부 검증 (20/80) 수행
    run_pipeline_predict80(
        csv_path=original_csv,
        save_csv=True,
        out_path=str(predictions_path),
        use_hybrid=use_hybrid,
        use_cuml=use_cuml,
    )

    # 2) 전체 데이터로 최종 모델 학습
    df = load_by_data_fields(original_csv)
    df = enforce_schema_types(df)

    model_t1, model_t2, t1_name, t2_name, feature_frame = _fit_full_models(
        df,
        random_state=random_state,
    )

    metadata = {
        "target1_name": t1_name,
        "target2_name": t2_name,
        "feature_columns": list(feature_frame.columns),
        "use_hybrid": use_hybrid,
        "random_state": random_state,
    }

    payload = {
        "model_target1": model_t1,
        "model_target2": model_t2,
        "metadata": metadata,
    }

    _ensure_parent_dir(Path(model_path))
    joblib.dump(payload, model_path)
    print(f"[INFO] 모델 저장 완료: {model_path}")
    return metadata


def _load_models(model_path: Path) -> Dict[str, object]:
    if not Path(model_path).exists():
        raise FileNotFoundError(f"모델 파일이 존재하지 않습니다: {model_path}")

    payload = joblib.load(model_path)
    required_keys = {"model_target1", "model_target2", "metadata"}
    if not required_keys.issubset(payload.keys()):
        raise ValueError(f"모델 파일에 필요한 키가 없습니다: {required_keys}")
    return payload


def _prepare_external_features(
    df: pd.DataFrame,
    feature_columns: list,
) -> pd.DataFrame:
    df_numeric = df.select_dtypes(include=[np.number]).copy()

    # 필요한 컬럼만 유지하고, 누락된 컬럼은 0으로 채움
    for col in feature_columns:
        if col not in df_numeric.columns:
            df_numeric[col] = 0.0

    # 불필요한 컬럼 제거 및 순서 정렬
    df_numeric = df_numeric[feature_columns]
    return df_numeric


def predict_external(
    input_csv: str,
    model_path: Path = DEFAULT_MODEL_PATH,
    output_csv: Path | None = None,
) -> pd.DataFrame:
    """
    저장된 모델을 이용해 외부 CSV의 예측값을 계산합니다.

    output_csv가 주어지면 결과를 CSV로 저장합니다.
    """
    payload = _load_models(model_path)
    model_t1: RandomForestClassifier = payload["model_target1"]
    model_t2: RandomForestClassifier = payload["model_target2"]
    metadata: Dict[str, object] = payload["metadata"]

    feature_columns = metadata["feature_columns"]
    use_hybrid = metadata.get("use_hybrid", False)
    t1_name = metadata["target1_name"]
    t2_name = metadata["target2_name"]

    df_raw = load_by_data_fields(input_csv)
    df_raw = enforce_schema_types(df_raw)

    feature_frame = _prepare_external_features(df_raw, feature_columns)

    pred_t1_ml = model_t1.predict(feature_frame)
    pred_t2_ml = model_t2.predict(feature_frame)

    pred_df = pd.DataFrame(
        {
            f"pred_{t1_name}_ml": pred_t1_ml,
            f"pred_{t2_name}_ml": pred_t2_ml,
        },
        index=df_raw.index,
    )

    if use_hybrid:
        rule_scores = calculate_rule_based_score(df_raw)
        pred_t1_hybrid = hybrid_predict(pred_t1_ml, rule_scores)
        pred_t2_hybrid = hybrid_predict(pred_t2_ml, rule_scores)
        pred_df[f"pred_{t1_name}"] = pred_t1_hybrid
        pred_df[f"pred_{t2_name}"] = pred_t2_hybrid
    else:
        pred_df[f"pred_{t1_name}"] = pred_df[f"pred_{t1_name}_ml"]
        pred_df[f"pred_{t2_name}"] = pred_df[f"pred_{t2_name}_ml"]

    columns_to_drop = {f"pred_{t1_name}", f"pred_{t2_name}"}
    pred_df = pred_df.drop(columns=columns_to_drop, errors="ignore")

    if output_csv:
        _ensure_parent_dir(Path(output_csv))
        pred_df.to_csv(output_csv, encoding="utf-8", index=False)
        print(f"[INFO] 예측 결과 저장: {output_csv}")

    return pred_df


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RandomForest 모델 관리 유틸리티")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="모델 초기 학습 및 저장")
    init_parser.add_argument("--original-csv", default=DEFAULT_ORIGINAL_CSV)
    init_parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    init_parser.add_argument("--predictions-path", default=str(DEFAULT_PREDICTION_PATH))
    init_parser.add_argument("--no-hybrid", action="store_true", help="하이브리드 모드 비활성화")
    init_parser.add_argument("--use-cuml", action="store_true", help="cuML 사용")

    predict_parser = subparsers.add_parser("predict", help="외부 CSV 예측 수행")
    predict_parser.add_argument("--input-csv", required=True)
    predict_parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    predict_parser.add_argument("--output-csv", help="예측 결과 저장 경로")

    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.command == "init":
        initialize_models(
            original_csv=args.original_csv,
            model_path=Path(args.model_path),
            predictions_path=Path(args.predictions_path),
            use_hybrid=not args.no_hybrid,
            use_cuml=args.use_cuml,
        )
    elif args.command == "predict":
        predict_external(
            input_csv=args.input_csv,
            model_path=Path(args.model_path),
            output_csv=Path(args.output_csv) if args.output_csv else None,
        )
    else:
        parser.error(f"지원하지 않는 명령입니다: {args.command}")


    df_pred = predict_external(
      input_csv="기업신용평가정보_합성데이터.csv",
      model_path="artifacts/rf_credit_models.joblib",
      output_csv="external_predictions.csv",  # 저장이 필요 없으면 생략
  )
    print(df_pred.head())
    


if __name__ == "__main__":
    main()

