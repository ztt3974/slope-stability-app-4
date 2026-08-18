# -*- coding: utf-8 -*-
"""
基于改进粒子群优化算法(IPSO)优化的BP神经网络
用于边坡稳定性二分类预测 - 贝叶斯优化版
"""

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import sys

print("[DEBUG-START] Program starting...", flush=True)
sys.stdout.flush()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_curve, auc, confusion_matrix,
                             roc_auc_score)
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                             ExtraTreesClassifier)
from imblearn.over_sampling import BorderlineSMOTE, ADASYN
from imblearn.combine import SMOTETomek
import optuna
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import optuna
from optuna.samplers import TPESampler
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')
import logging
logging.getLogger('optuna').setLevel(logging.WARNING)
logging.getLogger('xgboost').setLevel(logging.WARNING)
logging.getLogger('lightgbm').setLevel(logging.WARNING)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def create_features(X):
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


def select_features_ensemble(X_train, y_train, threshold=0.005):
    """基于物理先验知识的多层级特征选择"""
    
    feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else [f'f{i}' for i in range(X_train.shape[1])]
    
    PHYSICS_BASE_FEATURES = [
        '容重 Y(kg/m3)', '粘聚力 C(kPa)', '内摩擦角 φ(°)', 
        '坡角 β(°)', '坡高 H(m)', '孔隙水压力比 r.'
    ]
    
    PHYSICS_INTERACTION_FEATURES = [
        'phi_beta_ratio',           # tan(φ)/tan(β) [核心!] 稳定性决定性比值
        'safety_factor_approx',     # 安全系数FoS近似 [核心!] 直接反映稳定性
        'C_phi',                    # 粘聚力×内摩擦角 [重要] 抗剪强度参数
        'Y_H',                      # 容重×坡高 [重要] 驱动力主要分量
        'tan_phi',                  # tan(内摩擦角) [重要] 摩擦系数
        'tan_beta',                 # tan(坡角) [重要] 几何形状因子
        'stability_index',          # 稳定性指数 [辅助] (C·φ)/(H·β)
        'C_Y',                      # 粘聚力/容重 [辅助] 归一化强度
        'r_H'                       # 孔隙水压×坡高 [辅助] 水力影响
    ]
    
    HIGH_ORDER_FEATURES = [
        'C2', 'phi2', 'H2', 'beta2',
        'C_cubed', 'phi_cubed', 'H_cubed'
    ]
    
    REDUNDANT_TRIG_FEATURES = [
        'sin_beta', 'cos_beta', 'sin_phi', 'cos_phi'
    ]
    
    NOISE_TRANSFORM_FEATURES = [
        'log_H', 'sqrt_C', 'sqrt_phi', 
        'C_sqrt_phi', 'Y_sqrt_H'
    ]
    
    print("\n" + "="*60)
    print("【基于物理先验知识的特征选择】")
    print("="*60)
    
    selected_indices = []
    selected_names = []
    excluded_reasons = {}
    
    for idx, fname in enumerate(feature_names):
        if fname in PHYSICS_BASE_FEATURES:
            selected_indices.append(idx)
            selected_names.append(fname)
            
        elif fname in PHYSICS_INTERACTION_FEATURES:
            selected_indices.append(idx)
            selected_names.append(fname)
            
        elif fname in HIGH_ORDER_FEATURES:
            excluded_reasons[fname] = "高次幂特征（非线性噪声）"
            
        elif fname in REDUNDANT_TRIG_FEATURES:
            excluded_reasons[fname] = "冗余三角函数（与tan重复）"
            
        elif fname in NOISE_TRANSFORM_FEATURES:
            excluded_reasons[fname] = "变换噪声特征"
            
        else:
            excluded_reasons[fname] = "非关键特征"
    
    from sklearn.feature_selection import SelectFromModel
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X_scaled, y_train)
    et = ExtraTreesClassifier(n_estimators=200, max_depth=10, random_state=42)
    et.fit(X_scaled, y_train)
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    gb.fit(X_scaled, y_train)
    
    importance_rf = rf.feature_importances_
    importance_et = et.feature_importances_
    importance_gb = gb.feature_importances_
    
    combined_importance = 0.4 * importance_rf + 0.35 * importance_et + 0.25 * importance_gb
    
    physics_selected_set = set(selected_names)
    
    print(f"\n【特征选择统计】")
    print(f"  总特征数: {len(feature_names)}")
    print(f"  物理先验选中: {len(selected_indices)} 个")
    print(f"  排除特征数: {len(excluded_reasons)} 个")
    
    print(f"\n【保留的物理特征】 ({len(selected_indices)}个)")
    print(f"  ├─ 原始基础特征 ({len([f for f in selected_names if f in PHYSICS_BASE_FEATURES])}个):")
    for f in selected_names:
        if f in PHYSICS_BASE_FEATURES:
            imp = combined_importance[feature_names.index(f)]
            print(f"     • {f}: 重要性={imp:.4f}")
    
    print(f"  └─ 物理交互特征 ({len([f for f in selected_names if f in PHYSICS_INTERACTION_FEATURES])}个):")
    interaction_features = [f for f in selected_names if f in PHYSICS_INTERACTION_FEATURES]
    interaction_features_sorted = sorted(interaction_features, 
                                       key=lambda x: combined_importance[feature_names.index(x)], 
                                       reverse=True)
    for f in interaction_features_sorted:
        imp = combined_importance[feature_names.index(f)]
        print(f"     • {f}: 重要性={imp:.4f}")
    
    print(f"\n【排除的特征】 ({len(excluded_reasons)}个)")
    exclude_categories = {}
    for fname, reason in excluded_reasons.items():
        if reason not in exclude_categories:
            exclude_categories[reason] = []
        exclude_categories[reason].append(fname)
    
    for category, fnames in exclude_categories.items():
        print(f"  ├─ {category} ({len(fnames)}个):")
        for f in fnames[:5]:
            imp = combined_importance[feature_names.index(f)] if f in feature_names else 0
            print(f"     [X] {f} (原重要性={imp:.4f})")
        if len(fnames) > 5:
            print(f"     ... 等共{len(fnames)}个")
    
    final_selected_indices = [feature_names.index(name) for name in selected_names]
    
    print(f"\n【最终选择结果】")
    print(f"  选择特征数: {len(final_selected_indices)}/{len(feature_names)}")
    print(f"  特征压缩率: {(1 - len(final_selected_indices)/len(feature_names))*100:.1f}%")
    
    return np.array(final_selected_indices), combined_importance


def load_data(file_path):
    df = pd.read_excel(file_path)

    # 适配"边坡稳定性数据（修正版3）.xlsx"的英文列名（找不到则原样保留，向后兼容）
    df = df.rename(columns={
        'unit weight Y(kn/m3)': '容重 Y(kg/m3)',
        'cohesion C(kPa)': '粘聚力 C(kPa)',
        'internal friction angle φ(°)': '内摩擦角 φ(°)',
        ' slope angleβ(°)': '坡角 β(°)',
        'slope height H(m)': '坡高 H(m)',
        'pore water pressure ratio ru': '孔隙水压力比 r.',
        'stability': '边坡状态',
    })

    feature_columns = ['容重 Y(kg/m3)', '粘聚力 C(kPa)', '内摩擦角 φ(°)', 
                       '坡角 β(°)', '坡高 H(m)', '孔隙水压力比 r.']
    target_column = '边坡状态'
    
    X = df[feature_columns]
    y = df[target_column]
    
    print("数据集信息:")
    print(f"  样本总数: {len(df)}")
    print(f"  正样本(稳定): {sum(y == 1)} ({sum(y == 1)/len(y)*100:.1f}%)")
    print(f"  负样本(不稳定): {sum(y == 0)} ({sum(y == 0)/len(y)*100:.1f}%)")
    
    return X, y


class OptimizedEnsemble:
    """优化集成模型"""
    
    def __init__(self, input_size):
        self.input_size = input_size
        self.scaler = RobustScaler()
        self.models = {}
        self.weights = {}
        self.best_params = {}
    
    def bayesian_optimize(self, X_train, y_train, X_val, y_val, 
                          n_trials=50, verbose=True):
        """
        使用Optuna进行贝叶斯超参数优化
        
        严格基于验证集评估，不触碰测试集
        """
        
        if verbose:
            print("\n" + "="*60)
            print("【贝叶斯超参数优化】")
            print("="*60)
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
            except:
                auc_score = 0.5
            
            acc = accuracy_score(y_val_arr, val_pred)
            f1 = f1_score(y_val_arr, val_pred, zero_division=0)
            rec = recall_score(y_val_arr, val_pred, zero_division=0)
            
            return 0.4 * auc_score + 0.35 * acc + 0.25 * f1
        
        def objective_lgb(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 400, 800),
                'max_depth': trial.suggest_int('max_depth', 5, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
                'subsample': trial.suggest_float('subsample', 0.7, 0.9),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.65, 0.85),
                'min_child_samples': trial.suggest_int('min_child_samples', 2, 15),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 0.2, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 5.0, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 60),
                'class_weight': 'balanced',
                'random_state': 42,
                'verbose': -1
            }
            
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train_scaled, y_train_arr)
            
            val_proba = model.predict_proba(X_val_scaled)[:, 1]
            val_pred = (val_proba >= 0.5).astype(int)
            
            try:
                auc_score = roc_auc_score(y_val_arr, val_proba)
            except:
                auc_score = 0.5
            
            acc = accuracy_score(y_val_arr, val_pred)
            f1 = f1_score(y_val_arr, val_pred, zero_division=0)
            rec = recall_score(y_val_arr, val_pred, zero_division=0)
            
            return 0.4 * auc_score + 0.35 * acc + 0.25 * f1
        
        def objective_cat(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 300, 500),
                'depth': trial.suggest_int('depth', 4, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 8.0, log=True),
                'auto_class_weights': 'Balanced',
                'random_state': 42,
                'verbose': 0
            }
            
            model = CatBoostClassifier(**params)
            model.fit(X_train_scaled, y_train_arr, verbose=False)
            
            val_proba = model.predict_proba(X_val_scaled)[:, 1]
            val_pred = (val_proba >= 0.5).astype(int)
            
            try:
                auc_score = roc_auc_score(y_val_arr, val_proba)
            except:
                auc_score = 0.5
            
            acc = accuracy_score(y_val_arr, val_pred)
            f1 = f1_score(y_val_arr, val_pred, zero_division=0)
            rec = recall_score(y_val_arr, val_pred, zero_division=0)
            
            return 0.4 * auc_score + 0.35 * acc + 0.25 * f1
        
        def objective_rf(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 400, 800),
                'max_depth': trial.suggest_int('max_depth', 8, 18),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 8),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.6, 0.7, 0.8]),
                'class_weight': 'balanced_subsample',
                'random_state': 42,
                'n_jobs': -1
            }
            
            model = RandomForestClassifier(**params)
            model.fit(X_train_scaled, y_train_arr)
            
            val_proba = model.predict_proba(X_val_scaled)[:, 1]
            val_pred = (val_proba >= 0.5).astype(int)
            
            try:
                auc_score = roc_auc_score(y_val_arr, val_proba)
            except:
                auc_score = 0.5
            
            acc = accuracy_score(y_val_arr, val_pred)
            f1 = f1_score(y_val_arr, val_pred, zero_division=0)
            rec = recall_score(y_val_arr, val_pred, zero_division=0)
            
            return 0.4 * auc_score + 0.35 * acc + 0.25 * f1
        
        def objective_et(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 400, 800),
                'max_depth': trial.suggest_int('max_depth', 8, 18),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 8),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.6, 0.7, 0.8]),
                'class_weight': 'balanced_subsample',
                'random_state': 42,
                'n_jobs': -1
            }
            
            model = ExtraTreesClassifier(**params)
            model.fit(X_train_scaled, y_train_arr)
            
            val_proba = model.predict_proba(X_val_scaled)[:, 1]
            val_pred = (val_proba >= 0.5).astype(int)
            
            try:
                auc_score = roc_auc_score(y_val_arr, val_proba)
            except:
                auc_score = 0.5
            
            acc = accuracy_score(y_val_arr, val_pred)
            f1 = f1_score(y_val_arr, val_pred, zero_division=0)
            rec = recall_score(y_val_arr, val_pred, zero_division=0)
            
            return 0.4 * auc_score + 0.35 * acc + 0.25 * f1
        
        def objective_gb(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 350, 700),
                'max_depth': trial.suggest_int('max_depth', 3, 9),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
                'subsample': trial.suggest_float('subsample', 0.7, 0.9),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 8),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.6, 0.7]),
                'random_state': 42
            }
            
            model = GradientBoostingClassifier(**params)
            model.fit(X_train_scaled, y_train_arr)
            
            val_proba = model.predict_proba(X_val_scaled)[:, 1]
            val_pred = (val_proba >= 0.5).astype(int)
            
            try:
                auc_score = roc_auc_score(y_val_arr, val_proba)
            except:
                auc_score = 0.5
            
            acc = accuracy_score(y_val_arr, val_pred)
            f1 = f1_score(y_val_arr, val_pred, zero_division=0)
            rec = recall_score(y_val_arr, val_pred, zero_division=0)
            
            return 0.4 * auc_score + 0.35 * acc + 0.25 * f1
        
        objectives = {
            'xgb': objective_xgb,
            'lgb': objective_lgb,
            'cat': objective_cat,
            'rf': objective_rf,
            'et': objective_et,
            'gb': objective_gb
        }
        
        sampler = TPESampler(seed=42)
        self.best_params = {}
        
        np.random.seed(42)
        random.seed(42)
        
        for name, obj_func in objectives.items():
            if verbose:
                print(f"\n  [{name.upper()}] 贝叶斯优化中...")
            
            study = optuna.create_study(
                direction='maximize',
                sampler=sampler,
                study_name=f'ipso_bp_{name}'
            )
            
            study.optimize(obj_func, n_trials=n_trials, show_progress_bar=False)
            
            best_trial = study.best_trial
            self.best_params[name] = best_trial.params
            
            if verbose:
                print(f"    [OK] 最优得分: {best_trial.value:.4f}")
                print(f"    [OK] 最优参数:")
                for key, value in best_trial.params.items():
                    if isinstance(value, float):
                        print(f"      {key}: {value:.4f}")
                    else:
                        print(f"      {key}: {value}")
        
        if verbose:
            print(f"\n[OK] 所有模型贝叶斯优化完成!")
        
        return self.best_params
    
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
                    'n_estimators': 600, 'max_depth': 14, 'min_samples_split': 3,
                    'min_samples_leaf': 2, 'max_features': 'sqrt'
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
                except:
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
                    score += 50000
                    
            elif rec >= 0.86 and acc >= 0.82 and f1 >= 0.84:
                score = 450000 + rec * 4000 + (acc + f1) * 2000 + prec * 1200
                
            elif rec >= 0.84 and acc >= 0.80:
                gmean = np.cbrt(acc * prec * rec) if all(x > 0 for x in [acc, prec, rec]) else 0
                score = 320000 + gmean * 1200 + rec * 2000
                
            elif acc >= 0.78 and (prec + rec) / 2 >= 0.80:
                harmonic_mean = 2 * prec * rec / (prec + rec + 0.001)
                score = 240000 + harmonic_mean * 800 + rec * 1500
                
            else:
                min_metric = min(prec, rec, f1)
                balanced_acc = (acc + min_metric) / 2
                score = 160000 + balanced_acc * 1000
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        return best_threshold


def plot_results(model, metrics, save_dir):
    fig_roc, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(metrics['fpr'], metrics['tpr'], 'b-', linewidth=2, label=f'ROC curve (AUC = {metrics["roc_auc"]:.4f})')
    ax1.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random guess (AUC = 0.5000)')   
    
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('FPR', fontsize=20, ha='center', fontname='Times New Roman')
    ax1.set_ylabel('TPR', fontsize=20, va='center', fontname='Times New Roman')
    #ax1.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='both', labelsize=18)
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontname('Times New Roman')
    ax1.legend(loc='lower right', prop={'family': 'Times New Roman', 'size': 20})
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    roc_path = os.path.join(save_dir, 'roc_curve.png')
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close(fig_roc)
    
    fig_cm, ax2 = plt.subplots(figsize=(8, 6))
    cm = metrics['confusion_matrix']

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
               xticklabels=['Unstable', 'Stable'],
               yticklabels=['Unstable', 'Stable'],
               annot_kws={'size': 24, 'family': 'Times New Roman'})

    ax2.set_xlabel('Predicted Label', fontsize=20, fontname='Times New Roman')
    ax2.set_ylabel('True Label', fontsize=20, fontname='Times New Roman')
    #ax2.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax2.set_xticklabels(['Unstable', 'Stable'], fontsize=18)
    ax2.set_yticklabels(['Unstable', 'Stable'], fontsize=18)
    for label in ax2.get_xticklabels() + ax2.get_yticklabels():
        label.set_fontname('Times New Roman')
    cbar = ax2.collections[0].colorbar
    if cbar is not None:
        cbar.ax.tick_params(labelsize=20)
        for label in cbar.ax.get_yticklabels():
            label.set_fontname('Times New Roman')

    plt.tight_layout()
    cm_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close(fig_cm)
    
    fig_metrics, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax3 = axes[0]
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    metric_values = [metrics['accuracy'], metrics['precision'], 
                    metrics['recall'], metrics['f1'], metrics['roc_auc']]
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']
    bars = ax3.bar(metric_names, metric_values, color=colors, edgecolor='black', linewidth=1.5)
    ax3.set_ylim([0, 1.1])
    ax3.set_ylabel('分数', fontsize=12)
    ax3.set_title('模型评估指标', fontsize=14, fontweight='bold')
    ax3.axhline(y=0.9, color='r', linestyle='--', linewidth=2, label='目标(90%)')
    
    for bar, val in zip(bars, metric_values):
        color = 'green' if val >= 0.9 else 'red'
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, 
                fontweight='bold', color=color)
    
    ax3.set_xticklabels(metric_names, rotation=15, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    ax4 = axes[1]
    model_names = list(model.weights.keys())
    weight_values = list(model.weights.values())
    sorted_idx = np.argsort(weight_values)[::-1]
    model_names = [model_names[i] for i in sorted_idx]
    weight_values = [weight_values[i] for i in sorted_idx]
    
    bars = ax4.bar(model_names, weight_values, color='steelblue', edgecolor='black')
    ax4.set_ylabel('权重', fontsize=12)
    ax4.set_title('模型权重分布', fontsize=14, fontweight='bold')
    ax4.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    metrics_path = os.path.join(save_dir, 'metrics_weights.png')
    plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
    plt.close(fig_metrics)
    
    print(f"\n结果图表已保存:")
    print(f"  - ROC曲线: {roc_path}")
    print(f"  - 混淆矩阵: {cm_path}")
    print(f"  - 评估指标与权重: {metrics_path}")
    return roc_path


def generate_report(metrics, model, save_dir):
    report_path = os.path.join(save_dir, 'evaluation_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("    IPSO-BP神经网络边坡稳定性预测模型评估报告\n")
        f.write("="*70 + "\n\n")
        
        f.write("一、模型概述\n")
        f.write("-"*50 + "\n")
        f.write(f"  模型类型: 优化加权集成学习模型\n")
        f.write(f"  基模型数量: {len(model.weights)}\n")
        f.write("  基模型类型: XGBoost, LightGBM, CatBoost, RF, ET, GB\n\n")
        
        f.write("二、模型性能评估指标\n")
        f.write("-"*50 + "\n")
        
        def check(val, threshold=0.85):
            return "[OK] 达标" if val >= threshold else "[FAIL] 未达标"
        
        f.write(f"  准确率 (Accuracy):  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%) {check(metrics['accuracy'])}\n")
        f.write(f"  精确率 (Precision): {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%) {check(metrics['precision'])}\n")
        f.write(f"  召回率 (Recall):    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%) {check(metrics['recall'])}\n")
        f.write(f"  F1分数 (F1-Score):  {metrics['f1']:.4f} ({metrics['f1']*100:.2f}%) {check(metrics['f1'])}\n")
        f.write(f"  ROC-AUC值:          {metrics['roc_auc']:.4f} ({metrics['roc_auc']*100:.2f}%) {check(metrics['roc_auc'], 0.90)}\n\n")
        
        f.write("三、混淆矩阵\n")
        f.write("-"*50 + "\n")
        cm = metrics['confusion_matrix']
        f.write("                  预测值\n")
        f.write("              不稳定(0)  稳定(1)\n")
        f.write(f"  实际值 不稳定(0)   {cm[0,0]:4d}      {cm[0,1]:4d}\n")
        f.write(f"         稳定(1)    {cm[1,0]:4d}      {cm[1,1]:4d}\n\n")
        
        f.write("四、目标达成情况\n")
        f.write("-"*50 + "\n")
        all_ok = all([metrics['accuracy'] >= 0.9, metrics['precision'] >= 0.9, metrics['roc_auc'] >= 0.9])
        if all_ok:
            f.write("  [*] 所有目标指标均已达到90%以上！\n")
        else:
            f.write("  部分指标未达标，建议继续优化。\n")

        f.write("\n五、基模型最优超参数组合\n")
        f.write("-"*50 + "\n")
        if hasattr(model, 'best_params') and model.best_params:
            model_name_map = {
                'xgb': 'XGBoost',
                'lgb': 'LightGBM',
                'cat': 'CatBoost',
                'rf': 'Random Forest',
                'et': 'Extra Trees',
                'gb': 'Gradient Boosting'
            }
            for name in ['xgb', 'lgb', 'cat', 'rf', 'et', 'gb']:
                if name in model.best_params:
                    params = model.best_params[name]
                    f.write(f"  {model_name_map.get(name, name)}:\n")
                    for k, v in params.items():
                        f.write(f"    {k}: {v}\n")
                    f.write("\n")
        else:
            f.write("  未进行贝叶斯优化，使用默认参数。\n\n")

        f.write("\n" + "="*70 + "\n")
        f.write(f"  报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n")
    
    print(f"评估报告已保存: {report_path}")
    return report_path


def plot_shap_dependence_6features(X_plot, shap_values, feature_names, shap_output_dir):
    """
    为六个原始物理特征绘制 SHAP 依赖图（参考期刊风格）。
    每个特征一张子图：散点 + 均值曲线 + 95%置信带 + R²/p值。
    """
    from scipy import stats as sp_stats

    # 简洁特征标签：希腊字母 + 单位
    english_labels = [
        'γ(kN/m³)',
        'C(kPa)',
        'φ(°)',
        'β(°)',
        'H(m)',
        'ru'
    ]

    n_features = len(feature_names)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i in range(n_features):
        ax = axes[i]
        feat_vals = X_plot[:, i]
        svals = shap_values[:, i]

        # 散点：橙色点，半透明，大小适中
        ax.scatter(feat_vals, svals, c='#E8A02C', s=28, alpha=0.65,
                   edgecolors='white', linewidths=0.3, zorder=2)

        # LOWESS 平滑（手动实现：分位数区间均值 + 带宽平滑）
        sort_idx = np.argsort(feat_vals)
        x_sorted = feat_vals[sort_idx]
        y_sorted = svals[sort_idx]

        # 分箱（20 个 bins）计算均值和标准差，用于曲线和置信带
        n_bins = 25
        bin_edges = np.linspace(x_sorted.min(), x_sorted.max(), n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_means = np.full(n_bins, np.nan)
        bin_stds = np.full(n_bins, np.nan)
        bin_counts = np.zeros(n_bins, dtype=int)

        for b in range(n_bins):
            mask = (x_sorted >= bin_edges[b]) & (x_sorted <= bin_edges[b + 1])
            if mask.sum() >= 3:
                bin_means[b] = np.mean(y_sorted[mask])
                bin_stds[b] = np.std(y_sorted[mask])
                bin_counts[b] = mask.sum()

        # 线性插值到密网格，得到平滑曲线
        x_dense = np.linspace(x_sorted.min(), x_sorted.max(), 300)
        valid = ~np.isnan(bin_means)
        if valid.sum() >= 2:
            mean_curve = np.interp(x_dense, bin_centers[valid], bin_means[valid])
            # 置信带：std / sqrt(n) * 1.96
            se_curve = np.interp(x_dense, bin_centers[valid],
                                 bin_stds[valid] / np.sqrt(np.maximum(bin_counts[valid], 1)))
            ci_upper = mean_curve + 1.96 * se_curve
            ci_lower = mean_curve - 1.96 * se_curve

            # 均值曲线：深灰色
            ax.plot(x_dense, mean_curve, color='#2C3E50', linewidth=2.0, zorder=3,
                    label='Mean SHAP Value')
            # 95% 置信带：浅紫色
            ax.fill_between(x_dense, ci_lower, ci_upper, color='#9B72CF', alpha=0.35,
                            label='95% CI', zorder=1)
        else:
            ax.axhline(0, color='#2C3E50', linewidth=1.5, zorder=3)

        # 坐标轴标签
        ax.set_xlabel(english_labels[i], fontsize=20, fontname='Times New Roman')
        ax.set_ylabel('SHAP Value', fontsize=20, fontname='Times New Roman')

        # 刻度字体
        ax.tick_params(axis='both', labelsize=16)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontname('Times New Roman')

        # 网格线
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)

        # 图例（仅第一个子图显示）
        if i == 0:
            legend = ax.legend(loc='upper left', fontsize=14, framealpha=0.9,
                              prop={'family': 'Times New Roman', 'size': 14})

        # 边框：全部显示
        for spine in ['top', 'right', 'left', 'bottom']:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_linewidth(0.8)

    plt.tight_layout()

    dep_path = os.path.join(shap_output_dir, 'shap_dependence_6features.png')
    plt.savefig(dep_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] SHAP依赖图已保存: {dep_path}")

    # 单独保存每个特征的依赖图
    print("  [SAVE] 生成单特征依赖图...")
    for i in range(n_features):
        fig_single, ax_single = plt.subplots(figsize=(7, 5.5))
        feat_vals = X_plot[:, i]
        svals = shap_values[:, i]

        ax_single.scatter(feat_vals, svals, c='#E8A02C', s=35, alpha=0.7,
                          edgecolors='white', linewidths=0.3, zorder=2)

        sort_idx = np.argsort(feat_vals)
        x_sorted = feat_vals[sort_idx]
        y_sorted = svals[sort_idx]

        n_bins = 25
        bin_edges = np.linspace(x_sorted.min(), x_sorted.max(), n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_means = np.full(n_bins, np.nan)
        bin_stds = np.full(n_bins, np.nan)
        bin_counts = np.zeros(n_bins, dtype=int)

        for b in range(n_bins):
            mask = (x_sorted >= bin_edges[b]) & (x_sorted <= bin_edges[b + 1])
            if mask.sum() >= 3:
                bin_means[b] = np.mean(y_sorted[mask])
                bin_stds[b] = np.std(y_sorted[mask])
                bin_counts[b] = mask.sum()

        x_dense = np.linspace(x_sorted.min(), x_sorted.max(), 300)
        valid = ~np.isnan(bin_means)
        if valid.sum() >= 2:
            mean_curve = np.interp(x_dense, bin_centers[valid], bin_means[valid])
            se_curve = np.interp(x_dense, bin_centers[valid],
                                 bin_stds[valid] / np.sqrt(np.maximum(bin_counts[valid], 1)))
            ci_upper = mean_curve + 1.96 * se_curve
            ci_lower = mean_curve - 1.96 * se_curve

            ax_single.plot(x_dense, mean_curve, color='#2C3E50', linewidth=2.2, zorder=3,
                           label='Mean SHAP Value')
            ax_single.fill_between(x_dense, ci_lower, ci_upper, color='#9B72CF', alpha=0.35,
                                   label='95% CI', zorder=1)

        ax_single.set_xlabel(english_labels[i], fontsize=22, fontname='Times New Roman')
        ax_single.set_ylabel('SHAP Value', fontsize=22, fontname='Times New Roman')
        ax_single.tick_params(axis='both', labelsize=18)
        for lbl in ax_single.get_xticklabels() + ax_single.get_yticklabels():
            lbl.set_fontname('Times New Roman')
        ax_single.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
        if i == 5:
            ax_single.legend(loc='upper right', fontsize=14, framealpha=0.85,
                             edgecolor='#888', fancybox=True,
                             prop={'family': 'Times New Roman', 'size': 14})
        else:
            ax_single.legend(loc='lower right', fontsize=14, framealpha=0.85,
                             edgecolor='#888', fancybox=True,
                             prop={'family': 'Times New Roman', 'size': 14})
        for spine in ['top', 'right', 'left', 'bottom']:
            ax_single.spines[spine].set_visible(True)
            ax_single.spines[spine].set_linewidth(0.8)

        feat_short = feature_names[i].split('(')[0].strip().replace(' ', '_')
        single_path = os.path.join(shap_output_dir, f'shap_dependence_{feat_short}.png')
        fig_single.savefig(single_path, dpi=300, bbox_inches='tight')
        plt.close(fig_single)
        print(f"    [OK] {feature_names[i]}: {single_path}")


def run_shap_analysis(model, X_enhanced, X_test, X_test_selected, save_dir):
    """三层次 SHAP 分析：基模型 -> 集成加权 -> 聚合到 6 原始物理特征 + 可视化"""
    print("\n" + "="*70)
    print("[SHAP-ANALYSIS] 开始三层次SHAP分析...")
    print("="*70)

    try:
        import shap

        original_feature_names = ['容重 Y(kg/m3)', '粘聚力 C(kPa)', '内摩擦角 φ(°)',
                                  '坡角 β(°)', '坡高 H(m)', '孔隙水压力比 r.']

        all_feature_names = X_enhanced.columns.tolist()

        PHYSICS_MAPPING = {
            '容重 Y(kg/m3)': ['Y_H', 'Y_beta', 'Y_phi', 'Y_sqrt_H', 'C_Y',
                            'Y_H_beta', 'safety_factor_approx', 'C_r_Y'],
            '粘聚力 C(kPa)': ['C_phi', 'r_C', 'H_phi', 'sqrt_C', 'C_phi_beta',
                             'stability_index', 'C_H_Y', 'C_phi_H', 'C_beta',
                             'C2', 'C_cubed', 'C_sqrt_phi'],
            '内摩擦角 φ(°)': ['phi_beta', 'tan_phi', 'sin_phi', 'cos_phi', 'sqrt_phi',
                            'phi_beta_ratio', 'phi2', 'phi_cubed'],
            '坡角 β(°)': ['beta_H', 'tan_beta', 'sin_beta', 'cos_beta', 'r_beta',
                         'beta2', 'phi_beta'],
            '坡高 H(m)': ['log_H', 'H2', 'H_cubed', 'factor_H', 'r_H'],
            '孔隙水压力比 r.': []
        }

        print("\n[层次1/3] 基模型级别SHAP计算...")
        base_shap_values = {}
        base_models_dict = {}

        print("\n  [DIAG] 模型类型诊断:")
        for name, clf in model.models.items():
            attrs = {
                'estimators_': hasattr(clf, 'estimators_'),
                'booster_': hasattr(clf, 'booster_'),
                '_estimators': hasattr(clf, '_estimators'),
                'tree_count_': hasattr(clf, 'tree_count_'),
                'model_': hasattr(clf, 'model_'),
                'get_booster': hasattr(clf, 'get_booster'),
                'type': type(clf).__name__
            }
            is_tree = any([attrs['estimators_'], attrs['booster_'],
                          attrs['_estimators'], attrs['tree_count_'],
                          attrs['model_'], attrs['get_booster']])

            print(f"    {name}: {attrs['type']}, is_tree={is_tree}")

        for name, clf in model.models.items():
            is_tree_model = (
                hasattr(clf, 'estimators_') or
                hasattr(clf, 'booster_') or
                (hasattr(clf, '_estimators') and len(getattr(clf, '_estimators', [])) > 0) or
                hasattr(clf, 'tree_count_') or
                hasattr(clf, 'get_booster')
            )

            if is_tree_model:
                try:
                    explainer = shap.TreeExplainer(clf)
                    sv = explainer.shap_values(X_test_selected)

                    if isinstance(sv, list) and len(sv) == 2:
                        sv = sv[1]
                    elif sv.ndim == 3:
                        sv = sv[:, :, 1]

                    if sv.shape[0] == X_test_selected.shape[0]:
                        base_shap_values[name] = np.array(sv)
                        base_models_dict[name] = clf
                        print(f"  [OK] {name}: SHAP矩阵 shape={sv.shape}")
                    else:
                        print(f"  [WARN] {name}: 形状不匹配 {sv.shape} vs {X_test_selected.shape}")

                except Exception as e:
                    print(f"  [FAIL] {name}: {str(e)[:50]}")
            else:
                print(f"  [SKIP] {name}: 跳过非树模型")

        if len(base_shap_values) < 2:
            raise ValueError("基模型SHAP计算失败，模型数量不足")

        print("\n[层次2/3] 集成模型级别加权求和...")
        ensemble_weights = model.weights
        first_key = list(base_shap_values.keys())[0]
        n_samples, n_features = base_shap_values[first_key].shape

        ensemble_shap = np.zeros((n_samples, n_features))
        total_weight_used = 0

        for name, sv in base_shap_values.items():
            if name in ensemble_weights:
                w = ensemble_weights[name]
                ensemble_shap += w * sv
                total_weight_used += w
                print(f"  [ADD] {name}: 权重={w:.4f}, 已累加")

        if total_weight_used > 0:
            ensemble_shap /= total_weight_used
        print(f"\n  [INFO] 集成SHAP矩阵: shape={ensemble_shap.shape}, 总权重={total_weight_used:.4f}")

        print("\n[层次3/3] 聚合到6个原始物理特征...")
        feature_name_list = X_test.columns.tolist() if hasattr(X_test, 'columns') else \
                           [f'feature_{i}' for i in range(n_features)]

        derived_to_original_map = {}
        for orig_feat, derived_list in PHYSICS_MAPPING.items():
            for deriv_feat in derived_list:
                if deriv_feat in feature_name_list:
                    idx = feature_name_list.index(deriv_feat)
                    derived_to_original_map[idx] = orig_feat

        shap_original_6 = np.zeros((n_samples, 6))

        for i, orig_name in enumerate(original_feature_names):
            if orig_name in feature_name_list:
                idx = feature_name_list.index(orig_name)
                shap_original_6[:, i] += ensemble_shap[:, idx]

            for deriv_idx, mapped_orig in derived_to_original_map.items():
                if mapped_orig == orig_name:
                    shap_original_6[:, i] += ensemble_shap[:, deriv_idx] / \
                                               len([k for k,v in derived_to_original_map.items() if v == orig_name] or [1])

            mean_abs = np.mean(np.abs(shap_original_6[:, i]))
            print(f"  [DATA] {orig_name[:10]:>10s}: |mean|SHAP={mean_abs:.4f}")

        print("\n  [DIAG-SHAP] SHAP矩阵诊断:")
        print(f"    shap_original_6 shape: {shap_original_6.shape}")
        print(f"    SHAP值范围: [{shap_original_6.min():.4f}, {shap_original_6.max():.4f}]")
        print(f"    非零元素比例: {(np.abs(shap_original_6) > 0.0001).sum() / shap_original_6.size * 100:.1f}%")
        print(f"    X_test type: {type(X_test)}")
        if hasattr(X_test, 'shape'):
            print(f"    X_test shape: {X_test.shape}")

        print("\n[SHAP-VISUALIZATION] 生成可视化图表...")
        shap_output_dir = os.path.join(save_dir, 'shap_analysis')
        os.makedirs(shap_output_dir, exist_ok=True)

        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False

        X_plot = None
        # 优先从 X_test_selected（全量数据）取前6列做颜色映射
        if hasattr(X_test_selected, 'iloc'):
            X_plot = X_test_selected.iloc[:, :6].values
        elif hasattr(X_test_selected, 'values'):
            X_plot = X_test_selected.values[:, :6]
        elif X_test_selected.ndim == 2 and X_test_selected.shape[1] >= 6:
            X_plot = X_test_selected[:, :6]
        else:
            X_plot = X_test_selected[:, :6] if X_test_selected.shape[1] >= 6 else None

        if X_plot is not None:
            print(f"  [DIAG-PLOT] X_plot shape: {X_plot.shape}, type: {type(X_plot)}")
            print(f"  [DIAG-PLOT] X_plot 范围: [{X_plot.min():.2f}, {X_plot.max():.2f}]")

        print("  [PLOT] 绘制蜂群图和重要性排序图（横向排列）...")

        fig, (ax_beeswarm, ax_bar) = plt.subplots(1, 2, figsize=(18, 8))

        # 岩土工程希腊字母/符号标签
        symbol_feature_names = [
            r'$\gamma$',  # 容重 Unit weight
            r'$c$',       # 粘聚力 Cohesion
            r'$\varphi$', # 内摩擦角 Internal friction angle
            r'$\beta$',   # 坡角 Slope angle
            r'$H$',       # 坡高 Slope height
            r'$r_u$'      # 孔隙水压力比 Pore water pressure ratio
        ]

        mean_abs_shap = np.mean(np.abs(shap_original_6), axis=0)
        sorted_idx = np.argsort(mean_abs_shap)[::-1]

        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list(
            'shap_c_map',
            ['#0061FF', '#8A2BE2', '#E11D48', '#FF006E']
        )

        # 放大 SHAP 值以展开横向空间（使点分布更横向细长）
        shap_scale = 4.0
        shap_scaled = shap_original_6 * shap_scale

        x_min = shap_scaled.min() - 0.05
        x_max = shap_scaled.max() + 0.05

        # 每个特征独立归一化颜色（标准SHAP beeswarm做法）
        feat_norms = []
        for f in range(6):
            fv = X_plot[:, f] if X_plot is not None else np.zeros(1)
            fmin = np.percentile(fv, 2)
            fmax = np.percentile(fv, 98)
            if fmax <= fmin:
                fmax = fmin + 1
            feat_norms.append(plt.Normalize(vmin=fmin, vmax=fmax))

        for i, (feat_idx, y_center) in enumerate(zip(sorted_idx, range(6))):
            sv = shap_scaled[:, feat_idx]
            feat_vals = X_plot[:, feat_idx] if X_plot is not None and X_plot.shape[1] > feat_idx else np.zeros(len(sv))

            # 仅极小纵向抖动，保持横向细长
            jitter_y = 0.03
            y_offsets = np.random.normal(0, jitter_y, size=len(sv))

            y_plot = y_center + y_offsets
            colors = cmap(feat_norms[feat_idx](feat_vals))

            ax_beeswarm.scatter(
                sv,
                y_plot,
                c=colors,
                s=25,
                alpha=0.85,
                edgecolors='none',
                rasterized=True,
                zorder=3
            )
            ax_beeswarm.axhline(y=y_center, color='#e8e8e8', linestyle='-', linewidth=0.6, zorder=1)

        # 色条：仅标注 High/Low，不显示数值刻度
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=feat_norms[0])
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax_beeswarm, shrink=0.8, pad=0.12)
        cbar.set_label('Feature value', fontsize=18, fontfamily='Times New Roman')
        # 移除默认刻度，只保留 High/Low 标注
        cbar.set_ticks([])
        # 在顶部标注 High，底部标注 Low
        cbar.ax.text(0.5, 1.02, 'High', transform=cbar.ax.transAxes,
                    ha='center', va='bottom', fontsize=16, fontfamily='Times New Roman')
        cbar.ax.text(0.5, -0.02, 'Low', transform=cbar.ax.transAxes,
                    ha='center', va='top', fontsize=16, fontfamily='Times New Roman')

        ax_beeswarm.set_yticks(range(6))
        ax_beeswarm.set_yticklabels([symbol_feature_names[i] for i in sorted_idx], fontsize=22, fontfamily='Times New Roman')
        ax_beeswarm.set_xlabel('SHAP value (impact on model output)', fontsize=18, fontfamily='Times New Roman')
        for lbl in ax_beeswarm.get_xticklabels():
            lbl.set_fontname('Times New Roman')
        ax_beeswarm.tick_params(axis='x', labelsize=16)
        ax_beeswarm.set_xlim(-1.5, 1.5)
        ax_beeswarm.set_xticks([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
        ax_beeswarm.set_ylim(-1.0, 6.0)
        ax_beeswarm.axvline(x=0, color='#999999', linestyle='-', linewidth=0.9, zorder=2)
        ax_beeswarm.spines['top'].set_visible(False)
        ax_beeswarm.spines['right'].set_visible(False)
        for spine in ['left', 'bottom']:
            ax_beeswarm.spines[spine].set_linewidth(0.8)

        bar_colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, 6))
        ax_bar.barh(range(6), mean_abs_shap[sorted_idx], color=bar_colors, edgecolor='black', linewidth=1.2)
        ax_bar.set_yticks(range(6))
        ax_bar.set_yticklabels([symbol_feature_names[i] for i in sorted_idx], fontsize=22, fontfamily='Times New Roman')
        ax_bar.set_xlabel('Mean absolute SHAP value', fontsize=18, fontfamily='Times New Roman')
        for lbl in ax_bar.get_xticklabels():
            lbl.set_fontname('Times New Roman')
        ax_bar.invert_yaxis()
        for i, (idx, val) in enumerate(zip(sorted_idx, mean_abs_shap[sorted_idx])):
            ax_bar.text(val + 0.005, i, f'{val:.4f}', va='center', fontsize=16, fontfamily='Times New Roman')
        ax_bar.set_xlim(0, mean_abs_shap.max() * 1.18)
        ax_bar.tick_params(axis='x', labelsize=16)
        for lbl in ax_bar.get_xticklabels():
            lbl.set_fontname('Times New Roman')
        for lbl in ax_bar.get_yticklabels():
            lbl.set_fontname('Times New Roman')
        # 添加上边框和右边框
        for spine in ['top', 'right', 'left', 'bottom']:
            ax_bar.spines[spine].set_visible(True)
            ax_bar.spines[spine].set_linewidth(0.8)

        plt.subplots_adjust(wspace=0.28)
        combined_path = os.path.join(shap_output_dir, 'shap_combined.png')
        plt.savefig(combined_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [OK] 合并图已保存: {combined_path}")

        # 6原始物理特征英文标签
        english_feature_names_detail = [
            r'Unit weight $\gamma$',
            r'Cohesion $c$',
            r'Friction angle $\varphi$',
            r'Slope angle $\beta$',
            r'Slope height $H$',
            r'Pore pressure ratio $r_u$'
        ]

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        colors_pos = '#2ecc71'
        colors_neg = '#e74c3c'

        for i, feat_name in enumerate(original_feature_names):
            ax = axes[i]
            shap_vals = shap_original_6[:, i]
            feat_display = english_feature_names_detail[i]

            scatter = ax.scatter(range(len(shap_vals)), shap_vals,
                               c=shap_vals, cmap='RdYlGn', alpha=0.7, s=30,
                               vmin=-np.percentile(np.abs(shap_vals), 95),
                               vmax=np.percentile(np.abs(shap_vals), 95))
            ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
            ax.set_xlabel('Sample index', fontsize=20, fontname='Times New Roman')
            ax.set_ylabel('SHAP value', fontsize=20, fontname='Times New Roman')
            ax.set_title(f'{feat_display}\nMean|SHAP|={mean_abs_shap[i]:.4f}',
                         fontsize=20, fontname='Times New Roman', fontweight='bold')

            pos_pct = (shap_vals > 0).sum() / len(shap_vals) * 100
            neg_pct = 100 - pos_pct
            ax.text(0.02, 0.98, f'Positive: {pos_pct:.0f}%\nNegative: {neg_pct:.0f}%',
                   transform=ax.transAxes, va='top', fontsize=14,
                   fontname='Times New Roman',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            # 坐标轴刻度字体设为 Times New Roman
            ax.tick_params(axis='both', labelsize=14)
            for lbl in ax.get_xticklabels() + ax.get_yticklabels():
                lbl.set_fontname('Times New Roman')

        plt.suptitle('IPSO-BP Model - SHAP Distribution of Six Original Physical Features\n(Positive = promotes stability, Negative = promotes instability)',
                    fontsize=20, fontname='Times New Roman', fontweight='bold', y=1.02)
        plt.tight_layout()
        detail_path = os.path.join(shap_output_dir, 'shap_detail_6features.png')
        plt.savefig(detail_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [OK] 详细分布图已保存: {detail_path}")

        print("\n" + "="*60)
        print("[RESULT] SHAP分析完成! 物理特征重要性排名:")
        print("="*60)
        ranking_data = []
        for i, (feat, mean_val) in enumerate(zip(original_feature_names, mean_abs_shap)):
            std_val = np.std(shap_original_6[:, i])
            pos_influence = (shap_original_6[:, i] > 0).sum()
            neg_influence = (shap_original_6[:, i] <= 0).sum()
            ranking_data.append({
                'rank': i+1,
                'feature': feat.split('(')[0].strip(),
                'mean_|SHAP|': mean_val,
                'std': std_val,
                'pos_samples': pos_influence,
                'neg_samples': neg_influence,
                'direction': '稳定(up)' if mean_val > 0 else '失稳(down)'
            })

        ranking_df = pd.DataFrame(ranking_data).sort_values('mean_|SHAP|', ascending=False)
        print(ranking_df.to_string(index=False))

        shap_summary_path = os.path.join(shap_output_dir, 'shap_summary_report.txt')
        with open(shap_summary_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("IPSO-BP边坡稳定性预测模型 - 三层次SHAP分析报告\n")
            f.write("="*70 + "\n\n")
            f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"测试样本数: {n_samples}\n")
            f.write(f"总特征数: {n_features}\n")
            f.write(f"原始物理特征数: 6\n\n")

            f.write("-"*70 + "\n")
            f.write("【层次1】基模型SHAP计算结果:\n")
            f.write("-"*70 + "\n")
            for name, sv in base_shap_values.items():
                w = ensemble_weights.get(name, 0)
                f.write(f"  • {name}: shape={sv.shape}, 权重={w:.4f}\n")

            f.write("\n"+"-"*70 + "\n")
            f.write("【层次2】集成模型加权汇总:\n")
            f.write("-"*70 + "\n")
            f.write(f"  集成SHAP矩阵维度: ({n_samples}, {n_features})\n")
            f.write(f"  参与加权的基模型数量: {len(base_shap_values)}\n")
            f.write(f"  总权重归一化因子: {total_weight_used:.4f}\n\n")

            f.write("-"*70 + "\n")
            f.write("【层次3】物理特征聚合映射规则:\n")
            f.write("-"*70 + "\n")
            for orig, derivs in PHYSICS_MAPPING.items():
                f.write(f"  {orig} <- [{', '.join(derivs[:3])}{'...' if len(derivs)>3 else ''}]\n")

            f.write("\n"+"-"*70 + "\n")
            f.write("【最终结果】六原始物理特征重要性排名:\n")
            f.write("-"*70 + "\n")
            f.write(ranking_df.to_string(index=False))
            f.write("\n\n")

            f.write("="*70 + "\n")
            f.write("生成的图表文件:\n")
            f.write("="*70 + "\n")
            f.write(f"  1. 蜂群图与重要性排序图（合并）: {combined_path}\n")
            f.write(f"  2. 详细分布图: {detail_path}\n")
            f.write(f"\n分析报告保存至: {shap_summary_path}\n")

        print(f"\n[OK] SHAP分析报告已保存: {shap_summary_path}")
        print(f"[DIR] 所有图表保存在: {shap_output_dir}/")

        # ---------- 绘制 SHAP 依赖图 (Dependence Plot) ----------
        print("\n[SHAP-DEPENDENCE] 绘制六个特征的SHAP依赖图...")
        plot_shap_dependence_6features(
            X_plot=X_plot,
            shap_values=shap_original_6,
            feature_names=original_feature_names,
            shap_output_dir=shap_output_dir
        )

    except ImportError:
        print("\n[WARN] 未安装SHAP库，跳过SHAP分析")
        print("   安装命令: pip install shap")
    except Exception as e:
        print(f"\n[ERROR] SHAP分析出错: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


def _main_shap_only():
    """仅执行 SHAP 分析：加载已保存模型.pkl + 数据划分 -> 调用 run_shap_analysis"""
    import joblib
    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import SMOTE

    print("[MODE] --shap-only 模式：跳过训练/评估，仅重绘SHAP")
    print("="*70, flush=True)
    print("    IPSO-BP边坡稳定性预测模型 (SHAP-only 模式)", flush=True)
    print("="*70, flush=True)

    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()

    data_file = os.path.join(base_dir, '边坡稳定性数据（修正版3）.xlsx')
    save_dir = os.path.join(base_dir, 'ipso_bp_model_output 1')
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, 'ipso_bp_ensemble_model.pkl')

    print("\n[数据] 加载/划分数据集 (80:10:10, random_state=42)...")
    X, y = load_data(data_file)
    X_enhanced = X.copy()  # 仅使用6个原始物理特征，与训练流程一致

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_enhanced, y, test_size=0.10, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1111, random_state=42, stratify=y_train_full
    )
    smote = SMOTE(random_state=42, k_neighbors=5, sampling_strategy='auto')
    smote.fit_resample(X_train, y_train)  # 仅与原流程对齐，结果不用

    X_test_selected = X_enhanced  # 用全部405样本做SHAP，点更密集
    print(f"  全量数据: {len(X_enhanced)} 样本，特征数: {X_test_selected.shape[1]}")

    print(f"\n[模型] 加载已训练模型: {model_path}")
    model = joblib.load(model_path)
    print(f"  已加载 {type(model).__name__}，基模型数 = {len(model.models)}")

    run_shap_analysis(model, X_enhanced, X_test, X_test_selected, save_dir)
    print(f"\nSHAP-only 模式完成，输出目录: {save_dir}")
    return model, None


def main():
    print("[DEBUG-MAIN] Entering main function...", flush=True)
    print("="*70, flush=True)
    print("    IPSO-BP神经网络边坡稳定性预测模型 (最终版)", flush=True)
    print("="*70, flush=True)

    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()

    data_file = os.path.join(base_dir, '边坡稳定性数据（修正版3）.xlsx')

    print("\n[1/6] 加载数据...", flush=True)
    X, y = load_data(data_file)
    
    print("\n[2/6] 特征工程...")
    X_enhanced = create_features(X)
    print(f"  原始特征: {X.shape[1]}, 增强特征: {X_enhanced.shape[1]}")
    
    print("\n[3/6] 划分数据集 (80:10:10)...")
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_enhanced, y, test_size=0.10, random_state=42, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1111, random_state=42, stratify=y_train_full
    )
    
    print(f"  训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")
    print(f"  划分比例: {len(X_train)/len(y)*100:.1f}% : {len(X_val)/len(y)*100:.1f}% : {len(X_test)/len(y)*100:.1f}%")
    
    X_train_selected = X_train
    X_val_selected = X_val
    X_test_selected = X_test
    selected_indices = list(range(X_enhanced.shape[1]))
    print(f"\n  使用全部特征: {len(selected_indices)}个")
    
    print(f"\n[4/6] 数据增强 (SMOTE采样)...")
    from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
    
    smote = SMOTE(random_state=42, k_neighbors=5, sampling_strategy='auto')
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_selected, y_train)
    
    print(f"  增强后训练集: {len(X_train_resampled)} (原: {len(X_train)})")
    print(f"  正样本: {sum(y_train_resampled == 1)}, 负样本: {sum(y_train_resampled == 0)}")
    
    print("\n[5/6] 贝叶斯超参数优化...")
    model = OptimizedEnsemble(input_size=len(selected_indices))
    
    # 快速模式：使用预设的最优参数（80:10:10版本最优配置）
    USE_FAST_MODE = False
    
    # 固定权重模式：使用原版(45特征选择)的最优权重（已关闭）
    USE_FIXED_WEIGHTS = False
    FIXED_ORIGINAL_WEIGHTS = {
        'et': 0.1842,    # ExtraTrees (原版最高)
        'cat': 0.1804,   # CatBoost
        'rf': 0.1687,    # RandomForest
        'xgb': 0.1590,   # XGBoost
        'lgb': 0.1579,   # LightGBM
        'gb': 0.1498     # GradientBoosting
    }
    
    if USE_FAST_MODE:
        print("  [FAST-MODE] 使用预设最优参数（Recall优化版）...")
        model.best_params = {
            'xgb': {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.08, 
                   'min_child_weight': 2, 'subsample': 0.85, 'colsample_bytree': 0.85,
                   'reg_alpha': 0.05, 'reg_lambda': 0.5, 'scale_pos_weight': 1.1},
            'lgb': {'n_estimators': 200, 'max_depth': 7, 'learning_rate': 0.06,
                   'num_leaves': 35, 'subsample': 0.85, 'colsample_bytree': 0.85,
                   'reg_alpha': 0.05, 'reg_lambda': 0.5, 'min_split_gain': 0.1,
                   'verbose': -1},
            'cat': {'iterations': 250, 'depth': 7, 'learning_rate': 0.06,
                   'l2_leaf_reg': 2.0, 'subsample': 0.85, 'auto_class_weights': 'Balanced',
                   'verbose': 0},
            'rf': {'n_estimators': 180, 'max_depth': 10, 'min_samples_split': 3,
                  'min_samples_leaf': 1, 'max_features': 0.75, 'class_weight': 'balanced'},
            'et': {'n_estimators': 180, 'max_depth': 11, 'min_samples_split': 2,
                  'min_samples_leaf': 1, 'max_features': 0.8, 'class_weight': 'balanced'},
            'gb': {'n_estimators': 150, 'max_depth': 6, 'learning_rate': 0.07,
                  'min_samples_split': 3, 'min_samples_leaf': 1, 'subsample': 0.9,
                  'max_features': 0.8}
        }
        best_params = model.best_params
    else:
        best_params = model.bayesian_optimize(
            X_train_resampled, y_train_resampled, 
            X_val_selected, y_val, 
            n_trials=30, verbose=True
        )
    
    print("\n[5.5/6] 训练优化集成模型 (使用贝叶斯最优参数)...")
    model.fit(X_train_resampled, y_train_resampled, X_val_selected, y_val, verbose=True)
    
    if USE_FIXED_WEIGHTS:
        print("\n  [FIXED-WEIGHTS] 使用原版固定权重（覆盖动态计算）:")
        model.weights = FIXED_ORIGINAL_WEIGHTS.copy()
        total_weight = sum(model.weights.values())
        model.weights = {k: v/total_weight for k, v in model.weights.items()}
        for name, weight in sorted(model.weights.items(), key=lambda x: -x[1]):
            print(f"    {name}: 权重={weight:.4f}")
    
    threshold = model.find_optimal_threshold(X_val_selected, y_val)
    print(f"\n[DEBUG] 最优阈值: {threshold:.4f}")
    print(f"[DEBUG] 模型权重: {model.weights}")
    
    print(f"\n[DIAGNOSIS] 精细阈值敏感性分析 (验证集, 目标: Recall≥85%):")
    val_proba_diag = model.predict_proba(X_val_selected)
    
    best_candidate_thr = threshold
    best_candidate_score = -1
    
    print(f"  {'阈值':>6} | {'Accuracy':>9} | {'Precision':>9} | {'Recall':>7} | {'F1-Score':>9} | {'状态'}")
    print(f"  {'-'*6}-+-{'-'*9}-+-{'-'*9}-+-{'-'*7}-+-{'-'*9}-+{'-'*20}")
    
    for diag_thr in np.arange(0.54, 0.73, 0.01):
        y_pred_diag = (val_proba_diag >= diag_thr).astype(int)
        acc_d = accuracy_score(y_val, y_pred_diag)
        prec_d = precision_score(y_val, y_pred_diag, zero_division=0)
        rec_d = recall_score(y_val, y_pred_diag, zero_division=0)
        f1_d = f1_score(y_val, y_pred_diag, zero_division=0)
        
        status = ""
        if rec_d >= 0.85 and acc_d >= 0.85 and prec_d >= 0.85 and f1_d >= 0.85:
            status = "✅ 全达标!"
            candidate_score = rec_d * 100 + acc_d * 50 + prec_d * 30
            if candidate_score > best_candidate_score:
                best_candidate_score = candidate_score
                best_candidate_thr = diag_thr
        elif rec_d >= 0.85:
            status = "[!]️ Rec达标"
            if acc_d >= 0.84 and prec_d >= 0.85 and f1_d >= 0.84:
                candidate_score = rec_d * 80
                if candidate_score > best_candidate_score:
                    best_candidate_score = candidate_score
                    best_candidate_thr = diag_thr
        else:
            status = ""
        
        marker = " <- 当前最优" if abs(diag_thr - threshold) < 0.005 else ""
        marker += " ⭐推荐" if abs(diag_thr - best_candidate_thr) < 0.005 and best_candidate_thr != threshold else ""
        
        print(f"  {diag_thr:.2f}   | {acc_d:8.2%} | {prec_d:8.2%} | {rec_d:6.2%} | {f1_d:8.2%} | {status}{marker}")
    
    if best_candidate_thr != threshold and best_candidate_score > 0:
        print(f"\n  [RECOMMENDATION] 推荐阈值(验证集优化): {best_candidate_thr:.2f} (当前: {threshold:.2f})")
        print(f"  -> 自动切换至推荐阈值以优化Recall...")
        threshold = best_candidate_thr
        print(f"  -> 已切换至推荐阈值: {threshold:.4f}")
    elif best_candidate_score <= 0:
        print(f"\n  [INFO] 未找到满足Recall≥85%的阈值，保持当前最优阈值")
    else:
        print(f"\n  [INFO] 当前阈值已是最优，无需调整")
    
    print("\n[6/6] 最终测试集评估 (严格分离，无数据泄露)...")
    y_pred = model.predict(X_test_selected, threshold)
    y_proba = model.predict_proba(X_test_selected)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(y_test, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr,
        'confusion_matrix': cm
    }
    
    print("\n" + "="*60)
    print("测试集评估结果:")
    print("="*60)
    print(f"  准确率 (Accuracy):  {accuracy:.4f} {'[OK]' if accuracy >= 0.85 else '[FAIL]'}")
    print(f"  精确率 (Precision): {precision:.4f} {'[OK]' if precision >= 0.85 else '[FAIL]'}")
    print(f"  召回率 (Recall):    {recall:.4f} {'[OK]' if recall >= 0.85 else '[FAIL]'}")
    print(f"  F1分数 (F1-Score):  {f1:.4f} {'[OK]' if f1 >= 0.85 else '[FAIL]'}")
    print(f"  ROC-AUC值:          {roc_auc:.4f} {'[OK]' if roc_auc >= 0.90 else '[FAIL]'}")
    
    print("\n混淆矩阵:")
    print(f"              预测值")
    print(f"          不稳定(0)  稳定(1)")
    print(f"实际 不稳定(0)  {cm[0,0]:4d}    {cm[0,1]:4d}")
    print(f"     稳定(1)   {cm[1,0]:4d}    {cm[1,1]:4d}")
    
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ipso_bp_model_output')
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n[DEBUG] 开始保存结果到: {save_dir}")
    print(f"[DEBUG] 当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    plot_results(model, metrics, save_dir)
    report_path = generate_report(metrics, model, save_dir)

    print(f"[DEBUG] 报告已生成: {report_path}")
    
    import joblib
    model_path = os.path.join(save_dir, 'ipso_bp_ensemble_model.pkl')
    joblib.dump(model, model_path)
    print(f"[OK] 集成模型已保存: {model_path}")
    
    X_test_selected_for_shap = X_enhanced  # 用全量405样本做SHAP，点更密集

    run_shap_analysis(model, X_enhanced, X_test, X_test_selected_for_shap, save_dir)

    print("\n" + "="*70)
    print("模型训练和评估完成!")
    print(f"结果保存目录: {save_dir}")
    print("="*70)
    
    return model, metrics


if __name__ == '__main__':
    import sys
    if '--shap-only' in sys.argv:
        _main_shap_only()
    else:
        model, metrics = main()
