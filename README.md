# ⛰️ 边坡稳定性智能预测系统

基于 **改进粒子群优化算法（IPSO）优化的多模型加权集成神经网络** 的边坡稳定性二分类预测 Web 应用。

输入六个边坡物理参数（γ、C、φ、β、H、ru），系统基于 45 维物理增强特征与六模型集成（XGBoost / LightGBM / CatBoost / 随机森林 / 极端随机树 / 梯度提升）输出 **稳定 / 不稳定** 判别结果及概率。

## 功能模块

| 模块 | 说明 |
| --- | --- |
| 参数输入 | 六参数数值输入框（两侧 −/+ 步进按钮微调），带单位、取值范围校验与参数提示 |
| 智能预测 | 一键调用预训练集成模型，45 维特征工程 + 加权概率融合 |
| 结果展示 | 彩色状态卡片（绿=稳定 / 红=不稳定）+ 概率进度条 + 决策阈值说明 |
| 历史记录 | 预测记录自动持久化（history/predictions.csv），支持表格查看与 CSV 导出 |

## 项目结构

```
.
├── app.py                          # Streamlit 主应用
├── model_utils.py                  # 模型加载/训练/预测工具
├── train_model.py                  # 模型训练入口脚本
├── ipso_bp_slope_stability_fixed.py  # 原始训练管线（特征工程 + 集成模型类）
├── models/
│   ├── ipso_bp_ensemble_model.pkl  # 预训练集成模型
│   └── model_meta.json             # 模型元信息（阈值/权重/指标）
├── history/
│   └── predictions.csv             # 预测历史记录（自动生成）
├── .streamlit/config.toml          # Streamlit 主题与服务配置
├── requirements.txt                # Python 依赖
└── .gitignore
```

## 本地运行

```bash
# 1. 创建并激活虚拟环境（可选）
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux

# 2. 安装依赖
pip install -r requirements.txt

# 3. （可选）重新训练模型
#    将真实数据 "边坡稳定性数据（修正版3）.xlsx" 放到项目根目录后执行：
python train_model.py             # 优先使用真实数据，缺失时用内置参考数据集

# 4. 启动应用
streamlit run app.py
```

浏览器访问 `http://localhost:8501` 即可使用。

> 说明：仓库已附带基于内置参考数据集训练的预训练模型，开箱即用。
> 若提供真实数据文件（"边坡稳定性数据（修正版3）.xlsx"，列名：
> unit weight Y(kn/m3)、cohesion C(kPa)、internal friction angle φ(°)、
> slope angleβ(°)、slope height H(m)、pore water pressure ratio ru、stability），
> 运行 `python train_model.py` 即可用完全一致的管线重训模型。

## 上传 GitHub

```bash
git init
git add .
git commit -m "feat: slope stability prediction app based on IPSO ensemble"
git branch -M main
git remote add origin https://github.com/<你的用户名>/<仓库名>.git
git push -u origin main
```

> 模型文件 `models/*.pkl` 默认被 .gitignore 忽略。如需在云端免训练直接部署，
> 请将 .gitignore 中 `models/*.pkl` 一行删除或改用 `git add -f models/ipso_bp_ensemble_model.pkl` 后提交。

## 部署到 Streamlit Community Cloud

1. 将代码推送到 GitHub 公开仓库（确认 `app.py` 位于仓库根目录）；
2. 打开 [share.streamlit.io](https://share.streamlit.io)，使用 GitHub 账号登录；
3. 点击 **New app**，选择对应仓库与分支，主文件路径填 `app.py`；
4. 点击 **Deploy**，等待构建完成即可获得公网访问链接。

## 免责声明

本系统预测结果仅供建模研究与初步评估参考，不能替代规范要求的边坡稳定性详细计算与现场勘察。
