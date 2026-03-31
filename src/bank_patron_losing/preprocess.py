import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

"""
银行客户流失预测 - 数据预处理模块

预处理流程：
1. 类别编码 - 将 Geography 和 Gender 转换为数值
2. 特征处理 - 二选一：
   - 离散化：使用分位数自动分箱 (0/1/2)，适用于决策树
   - 标准化：Z-score 标准化，适用于 SVM/神经网络
3. 类别平衡 - 平衡流失 (Exited=1) 和留存 (Exited=0) 样本数量
"""

# 不参与特征工程的列
DROP_COLUMNS = ['RowNumber', 'CustomerId', 'Surname', 'Exited', 'EB']
# 类别特征列
CATEGORICAL_COLS = ['Geography', 'Gender']
# 需要处理的连续特征列
CONTINUOUS_COLS = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']


def load_data(dataPath):
    """加载原始数据"""
    return pd.read_csv(dataPath)


def encode_categorical(df):
    """
    对类别特征进行编码

    返回编码后的 DataFrame 和编码器映射关系
    """
    df = df.copy() # 避免修改原始 DataFrame
    encoders = {}

    for col in CATEGORICAL_COLS:
        unique_vals = df[col].unique() # 获取该列中所有不重复的值
        # 创建映射字典
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        # 使用pandas的map方法把当列的每一个文本值替换为mapping字典中对应的数字
        df[col] = df[col].map(mapping)
        # 将当前列的映射规则保存到 encoders 字典中。
        encoders[col] = mapping

    return df, encoders


def balance_classes(df, random_state=10):
    """
    平衡类别分布，使正负样本数量相等

    使用随机欠采样多数类
    """
    df = df.copy()

    n_churned = (df["Exited"] == 1).sum()
    n_retained = (df["Exited"] == 0).sum()
    target = min(n_churned, n_retained)

    df_churned = df[df["Exited"] == 1].sample(n=target, random_state=random_state)
    df_retained = df[df["Exited"] == 0].sample(n=target, random_state=random_state)

    df_balanced = pd.concat([df_churned, df_retained]).sample(frac=1, random_state=random_state)

    return df_balanced


def prepare_features(df, drop_columns=DROP_COLUMNS):
    """
    准备特征矩阵 X 和标签 y

    参数:
        df: DataFrame
        drop_columns: 需要删除的列名列表

    返回:
        feature_df: 特征 DataFrame
        target: 标签 Series
    """
    df = df.copy()
    feature_df = df.drop(columns=[col for col in drop_columns if col in df.columns])
    target = df['Exited'].copy() if 'Exited' in df.columns else None

    return feature_df, target


class BankDataPreprocessor:
    """
    银行数据预处理器

    参数:
        random_state: 随机种子

    用法:
        # 为决策树准备数据（离散化）
        preprocessor = BankDataPreprocessor(random_state=10)
        X, y = preprocessor.fit_transform(df, balance=True, discretize=True)

        # 为 SVM/神经网络准备数据（标准化）
        preprocessor = BankDataPreprocessor(random_state=10)
        X, y = preprocessor.fit_transform(df, balance=True, discretize=False)
    """

    def __init__(self, random_state=10):
        self.random_state = random_state
        self.encoders = {}
        self.scaler = None
        self.discretize = False
        self.is_fitted = False

    def fit(self, df, discretize=False):
        """
        拟合预处理器

        参数:
            df: 原始 DataFrame（包含 Exited 列）
            discretize: 是否对连续特征进行离散化（仅对决策树有用）
        """
        self.discretize = discretize

        # 拟合类别编码器
        _, self.encoders = encode_categorical(df)

        # 如果需要离散化，计算分位点
        if discretize:
            self.quantiles = {}
            for col in CONTINUOUS_COLS:
                self.quantiles[col] = {
                    'q1': df[col].quantile(0.33),
                    'q2': df[col].quantile(0.66)
                }

        # 拟合标准化器（仅当不使用离散化时）
        if not discretize:
            self.scaler = StandardScaler()
            temp_df, _ = encode_categorical(df)
            self.scaler.fit(temp_df[CONTINUOUS_COLS])

        self.is_fitted = True
        return self

    def transform(self, df, balance=False):
        """
        转换数据

        参数:
            df: 原始 DataFrame
            balance: 是否平衡类别（仅在训练时设为 True）

        返回:
            X: 特征矩阵 (numpy array)
            y: 标签向量 (numpy array)
        """
        if not self.is_fitted:
            raise ValueError("预处理器尚未 fit，请先调用 fit() 方法")

        df = df.copy()

        # 类别编码
        df, _ = encode_categorical(df)

        # 平衡类别（可选）
        if balance:
            df = balance_classes(df, self.random_state)

        # 准备特征和标签
        feature_df, target = prepare_features(df)

        # 离散化或标准化
        if self.discretize:
            # 离散化
            for col in CONTINUOUS_COLS:
                q1 = self.quantiles[col]['q1']
                q2 = self.quantiles[col]['q2']
                feature_df[col] = pd.cut(feature_df[col],
                                          bins=[-np.inf, q1, q2, np.inf],
                                          labels=[0, 1, 2],
                                          include_lowest=True).astype(int)
            X = feature_df.values.astype(float)
        else:
            # 标准化
            feature_df[CONTINUOUS_COLS] = self.scaler.transform(feature_df[CONTINUOUS_COLS])
            X = feature_df.values

        y = target.values if target is not None else None

        return X, y

    def fit_transform(self, df, balance=False, discretize=False):
        """
        拟合并转换数据

        参数:
            df: 原始 DataFrame
            balance: 是否平衡类别
            discretize: 是否离散化

        返回:
            X: 特征矩阵
            y: 标签向量
        """
        self.fit(df, discretize=discretize)
        return self.transform(df, balance=balance)

    def save(self, filepath):
        """保存预处理器"""
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath):
        """加载预处理器"""
        return joblib.load(filepath)


def create_train_test_data(original_csv_path, output_dir="./dataset",
                           test_size=0.2, random_state=10, discretize=False):
    """
    一站式完成数据加载、预处理、划分

    参数:
        original_csv_path: 原始 CSV 文件路径
        output_dir: 输出目录
        test_size: 测试集比例
        random_state: 随机种子
        discretize: 是否离散化（True=决策树用，False=SVM/神经网络用）

    返回:
        preprocessor: 预处理器对象
        X_train, X_test, y_train, y_test: 划分好的数据
    """
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    df = load_data(original_csv_path)

    # 创建预处理器并转换数据
    preprocessor = BankDataPreprocessor(random_state=random_state)
    X, y = preprocessor.fit_transform(df, balance=True, discretize=discretize)

    # 划分训练集和测试集（分层抽样）
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 保存数据
    np.save(os.path.join(output_dir, 'feature.npy'), X)
    np.save(os.path.join(output_dir, 'target.npy'), y)
    np.save(os.path.join(output_dir, 'feature_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'feature_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'target_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'target_test.npy'), y_test)

    # 保存预处理器
    preprocessor.save(os.path.join(output_dir, 'preprocessor.pkl'))

    print(f"数据已保存到 {output_dir}/")
    print(f"训练集形状：{X_train.shape}, 测试集形状：{X_test.shape}")
    print(f"类别分布 - 训练集：{np.bincount(y_train)}, 测试集：{np.bincount(y_test)}")

    return preprocessor, X_train, X_test, y_train, y_test


def main():
    """
    主函数：演示完整的预处理流程
    """
    print("=" * 60)
    print("银行客户流失预测 - 数据预处理")
    print("=" * 60)

    # 方案 1：为决策树准备数据（离散化）
    print("\n【方案 1】决策树专用数据（离散化 + 类别平衡）")
    preprocessor_dt, *_ = create_train_test_data(
        "./dataset/Churn-Modelling-0-original.csv",
        output_dir="./dataset/tree",
        discretize=True,
        random_state=10
    )

    # 方案 2：为 SVM/神经网络准备数据（标准化）
    print("\n【方案 2】SVM/神经网络专用数据（标准化 + 类别平衡）")
    preprocessor_svm, *_ = create_train_test_data(
        "./dataset/Churn-Modelling-0-original.csv",
        output_dir="./dataset/scaled",
        discretize=False,
        random_state=10
    )


if __name__ == "__main__":
    main()
