"""
银行客户流失预测模块

提供数据预处理、模型训练和结果分析的完整流程
"""

from .preprocess import (
    load_data,
    encode_categorical,
    balance_classes,
    prepare_features,
    BankDataPreprocessor,
    create_train_test_data,
)

from .split import (
    load_raw_data,
    split_from_dataframe,
)

from .train import (
    train_decision_tree,
    train_svm,
    train_neural_network,
)

from .analysis import (
    draw_roc_curve,
    draw_confusion_matrix,
    fold_cross_validation,
)


def main() -> None:
    """
    主函数：演示完整的银行客户流失预测流程
    """
    import numpy as np

    print("=" * 60)
    print("银行客户流失预测系统")
    print("=" * 60)

    # 使用标准化数据
    feature_train = np.load('./dataset/scaled/feature_train.npy')
    feature_test = np.load('./dataset/scaled/feature_test.npy')
    target_train = np.load('./dataset/scaled/target_train.npy')
    target_test = np.load('./dataset/scaled/target_test.npy')

    # 训练决策树
    print("\n【1. 决策树】")
    dt_model, dt_score = train_decision_tree(
        feature_train=feature_train,
        target_train=target_train,
        feature_test=feature_test,
        target_test=target_test
    )

    # 训练 SVM
    print("\n【2. SVM】")
    svm_model, svm_score = train_svm(
        feature_train=feature_train,
        target_train=target_train,
        feature_test=feature_test,
        target_test=target_test
    )

    # 训练神经网络
    print("\n【3. 神经网络】")
    mlp_model, mlp_score = train_neural_network(
        feature_train=feature_train,
        target_train=target_train,
        feature_test=feature_test,
        target_test=target_test
    )

    # 输出对比
    print("\n" + "=" * 60)
    print("模型性能对比")
    print("=" * 60)
    print(f"决策树准确率：    {dt_score:.4f}")
    print(f"SVM 准确率：       {svm_score:.4f}")
    print(f"神经网络准确率：  {mlp_score:.4f}")


__all__ = [
    # 预处理
    "load_data",
    "encode_categorical",
    "balance_classes",
    "prepare_features",
    "BankDataPreprocessor",
    "create_train_test_data",
    # 数据划分
    "load_raw_data",
    "split_from_dataframe",
    # 训练
    "train_decision_tree",
    "train_svm",
    "train_neural_network",
    # 分析
    "draw_roc_curve",
    "draw_confusion_matrix",
    "fold_cross_validation",
    # 主函数
    "main",
]
