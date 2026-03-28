# 银行客户流失预测 (Bank Customer Churn Prediction)

基于机器学习算法的银行客户流失预测分析项目，使用决策树、SVM 和神经网络三种模型对客户流失进行分类预测。

## 项目结构

```
bank-patron-losing/
├── src/
│   └── bank_patron_losing/
│       ├── __init__.py          # 模块入口
│       ├── preprocess.py        # 数据预处理（标准化/离散化）
│       ├── split.py             # 数据集划分
│       ├── train.py             # 模型训练
│       ├── analysis.py          # 结果分析
│       └── config.py            # 配置文件
├── dataset/
│   ├── Churn-Modelling-0-original.csv   # 原始数据集
│   ├── scaled/                          # 标准化数据（SVM/NN 用）
│   ├── tree/                            # 离散化数据（决策树用）
│   └── *.npy                            # 特征和标签数据
├── model/                               # 保存的模型文件
├── main.ipynb                           # Jupyter notebooks 分析脚本
└── pyproject.toml                       # 项目配置和依赖
```

## 安装

使用 [uv](https://github.com/astral-sh/uv) 安装依赖：

```bash
uv sync
```

## 使用方法

### 快速开始

```python
from bank_patron_losing import create_train_test_data, train_svm, train_neural_network, train_decision_tree
import numpy as np

# 1. 创建标准化数据（适用于 SVM 和神经网络）
preprocessor, X_train, X_test, y_train, y_test = create_train_test_data(
    "./dataset/Churn-Modelling-0-original.csv",
    output_dir="./dataset/scaled",
    discretize=False,  # False=标准化
    random_state=10
)

# 2. 训练模型
train_svm(
    feature_train=X_train, target_train=y_train,
    feature_test=X_test, target_test=y_test
)

train_neural_network(
    feature_train=X_train, target_train=y_train,
    feature_test=X_test, target_test=y_test
)

# 3. 决策树使用离散化数据（可选）
preprocessor_dt, X_train_dt, X_test_dt, y_train_dt, y_test_dt = create_train_test_data(
    "./dataset/Churn-Modelling-0-original.csv",
    output_dir="./dataset/tree",
    discretize=True,  # True=离散化
    random_state=10
)

train_decision_tree(
    feature_train=X_train_dt, target_train=y_train_dt,
    feature_test=X_test_dt, target_test=y_test_dt
)
```

### 使用预处理器

```python
from bank_patron_losing import BankDataPreprocessor
import pandas as pd

# 加载数据
df = pd.read_csv("./dataset/Churn-Modelling-0-original.csv")

# 创建预处理器
preprocessor = BankDataPreprocessor(random_state=10)

# 标准化（适用于 SVM/神经网络）
X, y = preprocessor.fit_transform(df, balance=True, discretize=False)

# 离散化（适用于决策树）
X, y = preprocessor.fit_transform(df, balance=True, discretize=True)

# 保存和加载预处理器
preprocessor.save("./dataset/preprocessor.pkl")
preprocessor = BankDataPreprocessor.load("./dataset/preprocessor.pkl")
```

### 使用 Jupyter Notebook

打开 `main.ipynb` 按单元格顺序执行即可完成完整的分析流程。

## 模型性能对比

| 模型 | 准确率 | 特点 | 预处理要求 |
|------|--------|------|-----------|
| 决策树 | ~74% | 无需标准化，可解释性强 | 可选离散化 |
| SVM | ~74% | 需要标准化，使用 `class_weight='balanced'` | 必须标准化 |
| 神经网络 | ~73% | 需要标准化，使用 `early_stopping` | 必须标准化 |

> 注：准确率基于平衡后的测试集（约 815 个样本，正负样本各约 400 个）

## 数据预处理

### 流程

1. **类别编码** - 将 `Geography` 和 `Gender` 转换为数值
2. **特征处理** - 二选一：
   - **标准化**：Z-score 标准化，适用于 SVM 和神经网络
   - **离散化**：使用分位数自动分箱 (0/1/2)，适用于决策树
3. **类别平衡** - 平衡流失 (Exited=1) 和留存 (Exited=0) 样本数量
4. **数据集划分** - 按 80%/20% 划分训练集和测试集（分层抽样）

### 自动分位数 vs 硬编码

旧版本使用硬编码的阈值（如 584, 718, 48000 等），新版本使用自动分位数：
- Q1 (33% 分位点)
- Q2 (66% 分位点)

这样可以适应不同的数据分布，提高泛化能力。

## 结果分析

- **ROC 曲线** - 评估模型分类性能
- **混淆矩阵** - 分析预测结果的分布
- **K 折交叉验证** - 验证模型稳定性（支持 5 折、10 折、15 折）

## 依赖

- Python >= 3.10
- numpy >= 2.2.6
- pandas >= 2.3.3
- scikit-learn >= 1.7.2
- matplotlib >= 3.10.8
- joblib >= 1.5.3
- jupyterlab >= 4.5.6

## API 参考

### 预处理

| 函数/类 | 说明 |
|---------|------|
| `create_train_test_data()` | 一站式完成数据加载、预处理、划分 |
| `BankDataPreprocessor` | 预处理器类，支持 fit/transform 模式 |
| `load_data()` | 加载原始数据 |
| `split_from_dataframe()` | 从 DataFrame 直接划分数据集 |

### 训练

| 函数 | 说明 |
|------|------|
| `train_decision_tree()` | 训练决策树（无需标准化） |
| `train_svm()` | 训练 SVM（自动标准化） |
| `train_neural_network()` | 训练神经网络（自动标准化） |

### 分析

| 函数 | 说明 |
|------|------|
| `draw_roc_curve()` | 绘制 ROC 曲线 |
| `draw_confusion_matrix()` | 绘制混淆矩阵 |
| `fold_cross_validation()` | K 折交叉验证 |

## 作者

- bacxper
