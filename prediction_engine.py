# -*- coding: utf-8 -*-
"""
prediction_engine.py - 边坡稳定性预测引擎（纯预测模块）

从 ipso_bp_slope_stability_fixed.py 中提取的特征工程(create_features)与
集成模型类(OptimizedEnsemble)，**不含 imblearn 依赖**，用于 Streamlit Cloud 部署。

imblearn 仅在训练时（SMOTE 采样）使用，预测时不需要，
因此此模块不导入 imblearn，避免 Streamlit Cloud Python 3.14 环境下
imbalanced-learn 与 scikit-learn 的兼容性问题。
"""

import numpy as np
import pandas as pd
import warnings

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              ExtraTreesClassifier)
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings('ignore')

optuna.logging.set_verbosity(optuna.logging.WARNING)


def create_features(X):
    """45维物理增强特征工程"""
    X_new = X.copy()

    X_new['C_phi'] = X['粘聚力 C(kPa)'] * X['内摩擦角 φ(°)']
    X_new['Y_H'] = X['容重 Y(kg/m3)'] * X['坡高 H(m)']
    X_new['beta_H'] = X['坡角 β(°)'] / (X['坡高 H(m)'] + 0.1)
    X_new['C_Y'] = X['粘聚力 C(kPa)'] / (X['容重 Y(kg/m3)'] + 0.1)
    X_new['phi_beta'] = X['内摩擦角 φ(°)'] / (X['坡角 β(°)'] + 0.1)
    X_new['r_C'] = X['孔隙水压力比 r.'] * X['粘聚力 C(kPa)']
    X_new['H_phi'] = X['坡高 H(m)'] / (X['内摩擦角 φ(°)'] + 0.1)
    X_new['Y_beta'] = X['容重 Y(kg/m3)'] * X['坡角 β(°)']

    X_new['C_phi_beta'] = X_new['C_phi'] / (X['坡角 β(°)'] + 0.1)
    X_new['Y_H_beta'] = X_new['Y_H'] / (X['坡角 β(°)'] + 0.1)
    X_new['stability_index'] = (X['粘聚力 C(kPa)'] * X['内摩擦角 φ(°)']) / (X['坡高 H(m)'] * X['坡角 β(°)'] + 0.1)
    X_new['factor_H'] = X['坡高 H(m)'] * X['孔隙水压力比 r.']
    X_new['C_r_Y'] = X['粘聚力 C(kPa)'] / (X['容重 Y(kg/m3)'] * (X['孔隙水压力比 r.'] + 0.01) + 0.1)

    X_new['tan_phi'] = np.tan(np.radians(X['内摩擦角 φ(°)']))
    X_new['tan_beta'] = np.tan(np.radians(X['坡角 β(°)']))
    X_new['phi_beta_ratio'] = X_new['tan_phi'] / (X_new['tan_beta'] + 0.01)

    X_new['C_H_Y'] = X['粘聚力 C(kPa)'] / (X['坡高 H(m)'] * X['容重 Y(kg/m3)'] + 0.1)
    X_new['r_beta'] = X['孔隙水压力比 r.'] * X['坡角 β(°)']
    X_new['r_H'] = X['孔隙水压力比 r.'] * X['坡高 H(m)']

    X_new['log_H'] = np.log1p(X['坡高 H(m)'])
    X_new['sqrt_C'] = np.sqrt(X['粘聚力 C(kPa)'])
    X_new['sqrt_phi'] = np.sqrt(X['内摩擦角 φ(°)'])

    X_new['C2'] = X['粘聚力 C(kPa)'] ** 2
    X_new['phi2'] = X['内摩擦角 φ(°)'] ** 2
    X_new['H2'] = X['坡高 H(m)'] ** 2
    X_new['beta2'] = X['坡角 β(°)'] ** 2

    X_new['C_sqrt_phi'] = X['粘聚力 C(kPa)'] * np.sqrt(X['内摩擦角 φ(°)'])
    X_new['Y_sqrt_H'] = X['容重 Y(kg/m3)'] * np.sqrt(X['坡高 H(m)'])

    X_new['sin_beta'] = np.sin(np.radians(X['坡角 β(°)']))
    X_new['cos_beta'] = np.cos(np.radians(X['坡角 β(°)']))
    X_new['sin_phi'] = np.sin(np.radians(X['内摩擦角 φ(°)']))
    X_new['cos_phi'] = np.cos(np.radians(X['内摩擦角 φ(°)']))

    X_new['safety_factor_approx'] = (X['粘聚力 C(kPa)'] + X['容重 Y(kg/m3)'] * X['坡高 H(m)'] * np.tan(np.radians(X['内摩擦角 φ(°)']))) / (X['容重 Y(kg/m3)'] * X['坡高 H(m)'] * np.sin(np.radians(X['坡角 β(°)'])) + 0.1)

    X_new['C_cubed'] = X['粘聚力 C(kPa)'] ** 1.5
    X_new['phi_cubed'] = X['内摩擦角 φ(°)'] ** 1.5
    X_new['H_cubed'] = X['坡高 H(m)'] ** 1.5

    X_new['C_phi_H'] = X_new['C_phi'] / (X['坡高 H(m)'] + 0.1)
    X_new['Y_phi'] = X['容重 Y(kg/m3)'] * X['内摩擦角 φ(°)']
    X_new['C_beta'] = X['粘聚力 C(kPa)'] / (X['坡角 β(°)'] + 0.1)

    return X_new


class OptimizedEnsemble:
    """优化集成模型（六模型加权集成）"""

    def __init__(self, input_size):
        self.input_size = input_size
        self.scaler = RobustScaler()
        self.models = {}
        self.weights = {}
        self.best_params = {}

    def bayesian_optimize(self, X_train, y_train, X_val, y_val,
                          n_trials=50, verbose=True):
        if verbose:
            print("\n" + "=" * 60)
            print("【贝叶斯超参数优化】")
            print("=" * 60)
            print(f"  搜索次数: {n_trials}")
            print(f"  优化目标: 验证集综合得分 (AUC*0.4 + Acc*0.35 + F1*0.25)")

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        y_train_arr = y_train.values if hasattr(y_train, 'values') else y_train
        y_val_arr = y_val.values if hasattr(y_val, 'values') else y_val

        def objective_xgb(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 400, 800),
                'max_depth': trial.suggest_int('max_depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
                'subsample': trial.suggest_float('subsample', 0.7, 0.9),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.65, 0.85),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 6),
                'gamma': trial.suggest_float('gamma', 0.01, 0.15, log=True),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 0.2, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 5.0, log=True),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 1.8),
                'random_state': 42,
                'use_label_encoder': False,
                'eval_metric': 'logloss'
            }
            model = xgb.XGBClassifier(**params)
            model.fit(X_train_scaled, y_train_arr, verbose=False)
            val_proba = model.predict_proba(X_val_scaled)[:, 1]
            val_pred = (val_proba >= 0.5).astype(int)
            try:
                auc_score = roc_auc_score(y_val_arr, val_proba)
            except Exception:
                auc_score = 0.5
            acc = accuracy_score(y_val_arr, val_pred)
            f1 = f1_score(y_val_arr, val_pred, zero_division=0)
            return 0.4 * auc_score + 0.35 * acc + 0.25 * f1

        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        study.optimize(objective_xgb, n_trials=n_trials, show_progress_bar=False)

        if verbose:
            print(f"\nXGBoost 最佳参数: {study.best_params}")
            print(f"XGBoost 最佳得分: {study.best_value:.4f}")

        self.best_params['xgb'] = study.best_params
        return self

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
        X_scaled = self.scaler.fit_transform(X_train)
        y_arr = y_train.values if hasattr(y_train, 'values') else y_train

        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            y_val_arr = y_val.values if hasattr(y_val, 'values') else y_val
        else:
            X_val_scaled = None
            y_val_arr = None

        use_bayesian = len(self.best_params) > 0

        if use_bayesian and verbose:
            print("训练基模型 (使用贝叶斯优化参数)...")
        elif verbose:
            print("训练基模型 (默认参数)...")

        def get_xgb_params():
            base_params = {
                'random_state': 42, 'use_label_encoder': False,
                'eval_metric': 'logloss'
            }
            if 'xgb' in self.best_params:
                base_params.update(self.best_params['xgb'])
            else:
                base_params.update({
                    'n_estimators': 600, 'max_depth': 7, 'learning_rate': 0.025,
                    'subsample': 0.80, 'colsample_bytree': 0.78, 'min_child_weight': 2,
                    'gamma': 0.05, 'reg_alpha': 0.03, 'reg_lambda': 2.0,
                    'scale_pos_weight': 1.2
                })
            return xgb.XGBClassifier(**base_params)

        def get_lgb_params():
            base_params = {'class_weight': 'balanced', 'random_state': 42, 'verbose': -1}
            if 'lgb' in self.best_params:
                base_params.update(self.best_params['lgb'])
            else:
                base_params.update({
                    'n_estimators': 600, 'max_depth': 8, 'learning_rate': 0.025,
                    'subsample': 0.80, 'colsample_bytree': 0.78, 'min_child_samples': 4,
                    'reg_alpha': 0.03, 'reg_lambda': 2.0
                })
            return lgb.LGBMClassifier(**base_params)

        def get_cat_params():
            base_params = {'auto_class_weights': 'Balanced', 'random_state': 42, 'verbose': 0}
            if 'cat' in self.best_params:
                base_params.update(self.best_params['cat'])
            else:
                base_params.update({
                    'iterations': 600, 'depth': 7, 'learning_rate': 0.025,
                    'l2_leaf_reg': 3.0
                })
            return CatBoostClassifier(**base_params)

        def get_rf_params():
            base_params = {'class_weight': 'balanced_subsample', 'random_state': 42, 'n_jobs': -1}
            if 'rf' in self.best_params:
                base_params.update(self.best_params['rf'])
            else:
                base_params.update({
                    'n_estimators': 600, 'max_depth': 14, 'min_samples_split': 3,
                    'min_samples_leaf': 2, 'max_features': 'sqrt'
                })
            return RandomForestClassifier(**base_params)

        def get_et_params():
            base_params = {'class_weight': 'balanced_subsample', 'random_state': 42, 'n_jobs': -1}
            if 'et' in self.best_params:
                base_params.update(self.best_params['et'])
            else:
                base_params.update({
                    'n_estimators': 500, 'max_depth': 11, 'min_samples_split': 2,
                    'min_samples_leaf': 1, 'max_features': 'sqrt'
                })
            return ExtraTreesClassifier(**base_params)

        def get_gb_params():
            base_params = {'random_state': 42}
            if 'gb' in self.best_params:
                base_params.update(self.best_params['gb'])
            else:
                base_params.update({
                    'n_estimators': 500, 'max_depth': 6, 'learning_rate': 0.025,
                    'subsample': 0.82, 'min_samples_split': 3, 'min_samples_leaf': 2,
                    'max_features': 'sqrt'
                })
            return GradientBoostingClassifier(**base_params)

        model_creators = {
            'xgb': get_xgb_params,
            'lgb': get_lgb_params,
            'cat': get_cat_params,
            'rf': get_rf_params,
            'et': get_et_params,
            'gb': get_gb_params
        }

        val_scores = {}
        val_aucs = {}

        for name, creator in model_creators.items():
            if verbose:
                print(f"  训练 {name}...", end=' ')
            model = creator()
            model.fit(X_scaled, y_arr)
            self.models[name] = model

            if X_val is not None:
                val_pred = model.predict(X_val_scaled)
                val_acc = accuracy_score(y_val_arr, val_pred)
                val_proba = model.predict_proba(X_val_scaled)[:, 1]

                try:
                    val_auc = roc_auc_score(y_val_arr, val_proba)
                except Exception:
                    val_auc = val_acc

                val_f1 = f1_score(y_val_arr, val_pred, zero_division=0)
                combined_score = 0.4 * val_auc + 0.35 * val_acc + 0.25 * val_f1

                val_scores[name] = combined_score
                val_aucs[name] = val_auc
                if verbose:
                    print(f"综合分={combined_score:.4f} (AUC={val_auc:.4f}, Acc={val_acc:.4f})")
            else:
                val_scores[name] = 1.0
                val_aucs[name] = 1.0
                if verbose:
                    print("完成")

        sorted_models = sorted(val_scores.items(), key=lambda x: -x[1])
        top_n = min(6, len(sorted_models))
        top_models = sorted_models[:top_n]

        raw_scores = np.array([score for _, score in top_models])
        exp_weights = np.exp(raw_scores * 5)
        normalized_weights = exp_weights / exp_weights.sum()

        for (name, _), w in zip(top_models, normalized_weights):
            self.weights[name] = w

        if verbose:
            print(f"\n选择Top {top_n}模型 (指数加权):")
            for name, weight in sorted(self.weights.items(), key=lambda x: -x[1]):
                print(f"  {name}: 权重={weight:.4f} (综合分={val_scores.get(name, 0):.4f})")

        return self

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        probas = []
        weights = []
        for name, weight in self.weights.items():
            model = self.models[name]
            proba = model.predict_proba(X_scaled)[:, 1]
            probas.append(proba.astype(float))
            weights.append(weight)
        weighted_proba = np.zeros(len(X), dtype=float)
        for proba, weight in zip(probas, weights):
            weighted_proba += proba * weight
        return weighted_proba

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def find_optimal_threshold(self, X_val, y_val):
        y_proba = self.predict_proba(X_val)
        best_threshold = 0.5
        best_score = -float('inf')

        for threshold in np.arange(0.28, 0.72, 0.002):
            y_pred = (y_proba >= threshold).astype(int)
            acc = accuracy_score(y_val, y_pred)
            prec = precision_score(y_val, y_pred, zero_division=0)
            rec = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)

            if acc < 0.72 or prec < 0.72 or rec < 0.72:
                score = -1000
            elif rec >= 0.88 and f1 >= 0.85:
                score = 600000 + rec * 5000 + acc * 3000 + f1 * 2500 + prec * 1500
                if acc >= 0.85:
                    score += 100000
                if abs(prec - rec) <= 0.10:
                    score = 160000 + acc * 1000

            if score > best_score:
                best_score = score
                best_threshold = threshold

        return best_threshold
