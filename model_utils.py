# -*- coding: utf-8 -*-
"""
模型工具模块
复用 ipso_bp_slope_stability_fixed.py 中的特征工程(create_features)与
集成模型类(OptimizedEnsemble)，提供：
  - 训练好的模型加载 (models/ipso_bp_ensemble_model.pkl)
  - 缺失时的快速训练（使用项目数据或内置参考数据集）
  - 单样本预测接口
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

# 复用训练脚本中的特征工程与集成模型类（该脚本被 __main__ 保护，导入安全）
from ipso_bp_slope_stability_fixed import create_features, OptimizedEnsemble

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "ipso_bp_ensemble_model.pkl")
META_PATH = os.path.join(MODEL_DIR, "model_meta.json")
DATA_PATH = os.path.join(BASE_DIR, "边坡稳定性数据（修正版3）.xlsx")

# 六个输入参数的列名（与训练管线保持一致）
FEATURE_COLUMNS = [
    "容重 Y(kg/m3)",
    "粘聚力 C(kPa)",
    "内摩擦角 φ(°)",
    "坡角 β(°)",
    "坡高 H(m)",
    "孔隙水压力比 r.",
]

# 输入参数校验范围（min, max）
PARAM_RANGES = {
    "容重 Y(kg/m3)": (10.0, 30.0),
    "粘聚力 C(kPa)": (0.0, 200.0),
    "内摩擦角 φ(°)": (0.0, 60.0),
    "坡角 β(°)": (5.0, 90.0),
    "坡高 H(m)": (1.0, 500.0),
    "孔隙水压力比 r.": (0.0, 0.5),
}

# ---------------------------------------------------------------------------
# 内置参考数据集（经典边坡稳定性案例，6参数 + 稳定性标签: 1=稳定, 0=不稳定）
# 当项目目录缺少真实数据文件时作为训练兜底，保证应用开箱即用。
# 如需使用真实数据，请将 "边坡稳定性数据（修正版3）.xlsx" 放到项目根目录后重新训练。
# ---------------------------------------------------------------------------
REFERENCE_DATA = [
    # (容重γ, 粘聚力C, 内摩擦角φ, 坡角β, 坡高H, 孔隙水压力比ru, 稳定性)
    (22.4, 10.0, 35.0, 45.0, 8.0, 0.40, 1),
    (26.0, 5.0, 30.0, 45.0, 60.0, 0.40, 0),
    (22.0, 5.0, 35.0, 40.0, 10.0, 0.40, 1),
    (21.0, 15.0, 30.0, 40.0, 20.0, 0.40, 1),
    (18.0, 15.0, 25.0, 35.0, 15.0, 0.40, 1),
    (20.0, 12.0, 22.0, 45.0, 20.0, 0.35, 0),
    (20.0, 20.0, 36.0, 45.0, 20.0, 0.30, 1),
    (22.0, 5.0, 28.0, 45.0, 20.0, 0.45, 0),
    (20.0, 25.0, 30.0, 40.0, 25.0, 0.25, 1),
    (25.0, 5.0, 35.0, 50.0, 20.0, 0.40, 0),
    (25.0, 5.0, 35.0, 40.0, 15.0, 0.40, 1),
    (26.0, 5.0, 30.0, 35.0, 30.0, 0.35, 0),
    (27.0, 25.0, 32.0, 42.0, 20.0, 0.30, 1),
    (27.0, 22.0, 33.0, 42.0, 30.0, 0.25, 1),
    (27.0, 20.0, 32.0, 45.0, 30.0, 0.40, 0),
    (22.0, 10.0, 25.0, 45.0, 25.0, 0.40, 0),
    (24.0, 15.0, 26.0, 45.0, 20.0, 0.35, 0),
    (23.0, 15.0, 28.0, 40.0, 20.0, 0.30, 1),
    (21.0, 15.0, 30.0, 32.0, 25.0, 0.25, 1),
    (27.0, 12.0, 35.0, 42.0, 60.0, 0.30, 1),
    (27.0, 15.0, 35.0, 42.0, 60.0, 0.30, 1),
    (27.0, 15.0, 35.0, 42.0, 80.0, 0.30, 1),
    (27.0, 15.0, 35.0, 42.0, 100.0, 0.35, 0),
    (27.0, 15.0, 35.0, 46.0, 80.0, 0.35, 0),
    (27.0, 18.0, 35.0, 46.0, 80.0, 0.30, 1),
    (27.0, 20.0, 35.0, 50.0, 100.0, 0.35, 1),
    (27.0, 20.0, 35.0, 55.0, 100.0, 0.35, 0),
    (27.0, 22.0, 35.0, 55.0, 100.0, 0.30, 1),
    (28.0, 25.0, 35.0, 55.0, 120.0, 0.30, 1),
    (28.0, 25.0, 35.0, 60.0, 120.0, 0.35, 0),
    (28.0, 30.0, 35.0, 60.0, 120.0, 0.30, 1),
    (28.0, 30.0, 38.0, 60.0, 150.0, 0.30, 1),
    (28.0, 30.0, 38.0, 65.0, 150.0, 0.35, 0),
    (28.0, 35.0, 38.0, 65.0, 150.0, 0.25, 1),
    (28.0, 35.0, 40.0, 65.0, 180.0, 0.25, 1),
    (28.0, 40.0, 40.0, 70.0, 180.0, 0.25, 1),
    (28.0, 40.0, 40.0, 70.0, 200.0, 0.30, 1),
    (28.0, 40.0, 40.0, 75.0, 200.0, 0.35, 0),
    (28.0, 45.0, 40.0, 75.0, 200.0, 0.25, 1),
    (28.0, 45.0, 42.0, 75.0, 250.0, 0.25, 1),
    (28.0, 50.0, 42.0, 80.0, 250.0, 0.25, 1),
    (18.5, 15.0, 0.0, 30.0, 6.0, 0.30, 0),
    (18.5, 45.0, 20.0, 30.0, 6.0, 0.25, 1),
    (18.5, 11.0, 12.0, 30.0, 6.0, 0.35, 0),
    (18.5, 40.0, 28.0, 35.0, 8.0, 0.25, 1),
    (19.0, 10.0, 10.0, 35.0, 10.0, 0.40, 0),
    (19.0, 30.0, 25.0, 35.0, 10.0, 0.30, 1),
    (19.5, 12.0, 18.0, 40.0, 15.0, 0.35, 0),
    (19.5, 35.0, 28.0, 40.0, 15.0, 0.25, 1),
    (20.0, 8.0, 15.0, 45.0, 20.0, 0.40, 0),
    (20.0, 40.0, 30.0, 45.0, 20.0, 0.25, 1),
    (21.0, 20.0, 20.0, 50.0, 30.0, 0.35, 0),
    (21.0, 45.0, 32.0, 50.0, 30.0, 0.25, 1),
    (22.0, 18.0, 18.0, 55.0, 40.0, 0.40, 0),
    (22.0, 50.0, 35.0, 55.0, 40.0, 0.25, 1),
    (23.0, 15.0, 16.0, 60.0, 50.0, 0.45, 0),
    (23.0, 55.0, 35.0, 60.0, 50.0, 0.25, 1),
    (24.0, 10.0, 14.0, 65.0, 60.0, 0.45, 0),
    (24.0, 60.0, 38.0, 65.0, 60.0, 0.25, 1),
]


def build_reference_dataset():
    """将内置参考数据构造为训练脚本兼容的 DataFrame"""
    df = pd.DataFrame(
        REFERENCE_DATA,
        columns=FEATURE_COLUMNS + ["边坡状态"],
    )
    return df[FEATURE_COLUMNS], df["边坡状态"]


def load_project_dataset():
    """
    优先加载项目真实数据 '边坡稳定性数据（修正版3）.xlsx'，
    不存在则返回 None, None（由调用方决定是否使用参考数据集）。
    """
    if not os.path.exists(DATA_PATH):
        return None, None
    df = pd.read_excel(DATA_PATH)
    df = df.rename(columns={
        "unit weight Y(kn/m3)": "容重 Y(kg/m3)",
        "cohesion C(kPa)": "粘聚力 C(kPa)",
        "internal friction angle φ(°)": "内摩擦角 φ(°)",
        " slope angleβ(°)": "坡角 β(°)",
        "slope height H(m)": "坡高 H(m)",
        "pore water pressure ratio ru": "孔隙水压力比 r.",
        "stability": "边坡状态",
    })
    missing = [c for c in FEATURE_COLUMNS + ["边坡状态"] if c not in df.columns]
    if missing:
        raise ValueError(f"数据文件缺少列: {missing}")
    return df[FEATURE_COLUMNS], df["边坡状态"]


def train_and_save(data_source="auto", verbose=True):
    """
    使用与 ipso_bp_slope_stability_fixed.main() 完全一致的管线训练集成模型：
      特征工程(45维) -> 训练/验证划分 -> SMOTE -> 集成训练 -> 阈值优化 -> 保存
    data_source: 'auto' 优先真实数据，缺失时用参考数据集
                 'reference' 强制使用参考数据集
    """
    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import SMOTE

    if data_source == "reference":
        X, y = build_reference_dataset()
        source_name = "内置参考数据集"
    else:
        X, y = load_project_dataset()
        source_name = "边坡稳定性数据（修正版3）.xlsx"
        if X is None:
            X, y = build_reference_dataset()
            source_name = "内置参考数据集"

    # 特征工程（与训练脚本一致，45维增强特征）
    X_enhanced = create_features(X)

    # 训练/验证划分（与训练脚本一致）
    X_train_full, X_val, y_train_full, y_val = train_test_split(
        X_enhanced, y, test_size=0.2, random_state=42, stratify=y
    )

    # SMOTE 数据增强（正负样本均 > k_neighbors 时启用）
    if min((y_train_full == 0).sum(), (y_train_full == 1).sum()) > 5:
        smote = SMOTE(random_state=42, k_neighbors=5, sampling_strategy="auto")
        X_train_res, y_train_res = smote.fit_resample(X_train_full, y_train_full)
    else:
        X_train_res, y_train_res = X_train_full, y_train_full

    # 集成模型训练（使用训练脚本 FAST 模式预设最优参数）
    model = OptimizedEnsemble(input_size=X_enhanced.shape[1])
    model.best_params = {
        "xgb": {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.08,
                "min_child_weight": 2, "subsample": 0.85, "colsample_bytree": 0.85,
                "reg_alpha": 0.05, "reg_lambda": 0.5, "scale_pos_weight": 1.1},
        "lgb": {"n_estimators": 200, "max_depth": 7, "learning_rate": 0.06,
                "num_leaves": 35, "subsample": 0.85, "colsample_bytree": 0.85,
                "reg_alpha": 0.05, "reg_lambda": 0.5, "min_split_gain": 0.1,
                "verbose": -1},
        "cat": {"iterations": 250, "depth": 7, "learning_rate": 0.06,
                "l2_leaf_reg": 2.0, "subsample": 0.85,
                "auto_class_weights": "Balanced", "verbose": 0},
        "rf": {"n_estimators": 180, "max_depth": 10, "min_samples_split": 3,
               "min_samples_leaf": 1, "max_features": 0.75, "class_weight": "balanced"},
        "et": {"n_estimators": 180, "max_depth": 11, "min_samples_split": 2,
               "min_samples_leaf": 1, "max_features": 0.8, "class_weight": "balanced"},
        "gb": {"n_estimators": 150, "max_depth": 6, "learning_rate": 0.07,
               "min_samples_split": 3, "min_samples_leaf": 1, "subsample": 0.9,
               "max_features": 0.8},
    }
    model.fit(X_train_res, y_train_res, X_val, y_val, verbose=verbose)

    # 阈值优化（与训练脚本一致，兜底 0.5）
    try:
        threshold = float(model.find_optimal_threshold(X_val, y_val))
        threshold = min(max(threshold, 0.3), 0.7)
    except Exception:
        threshold = 0.5

    # 验证集指标
    from sklearn.metrics import accuracy_score, f1_score
    y_pred = model.predict(X_val, threshold)
    metrics = {
        "val_accuracy": float(accuracy_score(y_val, y_pred)),
        "val_f1": float(f1_score(y_val, y_pred, zero_division=0)),
        "n_train": int(len(X_train_res)),
        "n_val": int(len(X_val)),
        "n_features": int(X_enhanced.shape[1]),
    }

    # 保存模型与元信息
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    meta = {
        "threshold": threshold,
        "weights": {k: float(v) for k, v in model.weights.items()},
        "data_source": source_name,
        "feature_columns": FEATURE_COLUMNS,
        **metrics,
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"[OK] 模型已保存: {MODEL_PATH}")
        print(f"[OK] 阈值: {threshold:.4f}, 权重: {meta['weights']}")
        print(f"[OK] 验证集 Acc={metrics['val_accuracy']:.4f}, F1={metrics['val_f1']:.4f}")

    return model, meta


def load_model():
    """
    加载训练好的模型；若不存在则自动快速训练（优先真实数据）。
    返回 (model, meta_dict)
    """
    if os.path.exists(MODEL_PATH) and os.path.exists(META_PATH):
        model = joblib.load(MODEL_PATH)
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return model, meta
    return train_and_save(data_source="auto", verbose=False)


def predict_single(model, meta, params):
    """
    对单个边坡样本预测。
    params: dict，键为 FEATURE_COLUMNS 中的列名
    返回 dict: {label, label_text, proba_stable, proba_unstable, threshold}
    """
    X = pd.DataFrame([params], columns=FEATURE_COLUMNS)
    X_enhanced = create_features(X)
    proba = float(model.predict_proba(X_enhanced)[0])
    threshold = float(meta.get("threshold", 0.5))
    label = 1 if proba >= threshold else 0
    return {
        "label": label,
        "label_text": "稳定" if label == 1 else "不稳定",
        "proba_stable": proba,
        "proba_unstable": 1.0 - proba,
        "threshold": threshold,
    }
