# -*- coding: utf-8 -*-
"""
模型训练入口
用法:
    python train_model.py            # 优先使用真实数据，缺失时使用内置参考数据集
    python train_model.py --reference  # 强制使用内置参考数据集
训练完成后模型保存至 models/ipso_bp_ensemble_model.pkl
"""

import sys
from model_utils import train_and_save

if __name__ == "__main__":
    source = "reference" if "--reference" in sys.argv else "auto"
    model, meta = train_and_save(data_source=source, verbose=True)
    print("\n训练完成。数据来源:", meta["data_source"])
