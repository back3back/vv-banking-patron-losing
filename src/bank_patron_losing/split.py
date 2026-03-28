import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_raw_data(dataPath):
    """
    加载原始数据，返回处理好的 DataFrame

    自动删除无关列：RowNumber, CustomerId, Surname, EB
    """
    df = pd.read_csv(dataPath)
    drop_cols = ['RowNumber', 'CustomerId', 'Surname', 'EB']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])
    return df


def split_from_dataframe(df, test_size=0.2, random_state=10, balance=False):
    """
    直接从 DataFrame 划分训练集和测试集

    参数:
        df: DataFrame（必须包含 Exited 列）
        test_size: 测试集比例
        random_state: 随机种子
        balance: 是否平衡类别

    返回:
        X_train, X_test, y_train, y_test: 特征和标签
    """
    df = df.copy()

    # 分离特征和标签
    X = df.drop(columns=['Exited'])
    y = df['Exited']

    # 类别平衡（可选）
    if balance:
        n_minority = min(y.value_counts())
        indices_0 = y[y == 0].index
        indices_1 = y[y == 1].index

        indices_0_sampled = np.random.RandomState(random_state).choice(indices_0, n_minority, replace=False)
        indices_1_sampled = np.random.RandomState(random_state).choice(indices_1, n_minority, replace=False)

        indices_balanced = np.concatenate([indices_0_sampled, indices_1_sampled])
        X = X.loc[indices_balanced]
        y = y.loc[indices_balanced]

    # 分层抽样划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # 测试
    df = load_raw_data("./dataset/Churn-Modelling-0-original.csv")
    X_train, X_test, y_train, y_test = split_from_dataframe(df, balance=True)
    print(f"训练集形状：{X_train.shape}, 测试集形状：{X_test.shape}")
    print(f"类别分布 - 训练集：{np.bincount(y_train)}, 测试集：{np.bincount(y_test)}")
