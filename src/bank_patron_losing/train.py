import numpy as np
from sklearn.tree import DecisionTreeClassifier
import joblib
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def train_decision_tree(criterion="gini", max_depth=6, min_samples_split=200,
                        feature_train=None, target_train=None,
                        feature_test=None, target_test=None,
                        save_path='./model/dt_model.pkl'):
    """
    训练决策树模型

    注意：决策树不需要特征标准化，直接传入原始特征即可
    """
    dt_model = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split
    )

    dt_model.fit(feature_train, target_train)
    joblib.dump(dt_model, save_path)
    print(f"模型已保存到 {save_path}")

    scores = dt_model.score(feature_test, target_test)
    print(f"决策树准确率：{scores:.4f}")

    predict_results = dt_model.predict(feature_test)
    np.save('./dataset/predict_results.npy', predict_results)

    return dt_model, scores


def train_svm(feature_train=None, target_train=None,
              feature_test=None, target_test=None,
              kernel='rbf', C=1.0, gamma='scale'):
    """
    训练 SVM 模型

    注意：SVM 需要特征标准化，请输入已经标准化的特征
         并使用 class_weight='balanced' 处理类别不平衡
    """
    # 标准化特征
    scaler = StandardScaler()
    feature_train_scaled = scaler.fit_transform(feature_train)
    feature_test_scaled = scaler.transform(feature_test)

    # 保存 scaler 以便后续使用
    joblib.dump(scaler, './model/scaler.pkl')

    # SVM 分类器
    clf = svm.SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        class_weight='balanced',  # 自动处理类别不平衡
        random_state=10
    )

    clf.fit(feature_train_scaled, target_train)
    predict_results = clf.predict(feature_test_scaled)
    np.save('./dataset/predict_results_svm.npy', predict_results)

    scores_svm = clf.score(feature_test_scaled, target_test)
    print(f"SVM 准确率：{scores_svm:.4f}")

    return clf, scores_svm


def train_neural_network(feature_train=None, target_train=None,
                         feature_test=None, target_test=None,
                         hidden_layer_sizes=(10, 11),
                         alpha=1e-5, max_iter=5000):
    """
    训练神经网络模型

    注意：神经网络需要特征标准化，请输入已经标准化的特征
    """
    # 标准化特征
    scaler = StandardScaler()
    feature_train_scaled = scaler.fit_transform(feature_train)
    feature_test_scaled = scaler.transform(feature_test)

    # MLP 分类器
    mlp = MLPClassifier(
        solver='lbfgs',
        alpha=alpha,
        hidden_layer_sizes=hidden_layer_sizes,
        random_state=1,
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.1
    )
    mlp.fit(feature_train_scaled, target_train)

    # 预测结果
    predict_results = mlp.predict(feature_test_scaled)
    np.save('./dataset/predict_results_mlp.npy', predict_results)

    scores_mlp = mlp.score(feature_test_scaled, target_test)
    print(f"神经网络准确率：{scores_mlp:.4f}")

    return mlp, scores_mlp


if __name__ == "__main__":
    # 读取数据
    feature_train = np.load('./dataset/feature_train.npy')
    feature_test = np.load('./dataset/feature_test.npy')
    target_train = np.load('./dataset/target_train.npy')
    target_test = np.load('./dataset/target_test.npy')

    print("=" * 50)
    print("训练决策树...")
    print("=" * 50)
    train_decision_tree(feature_train=feature_train, target_train=target_train,
                        feature_test=feature_test, target_test=target_test)

    print("\n" + "=" * 50)
    print("训练 SVM...")
    print("=" * 50)
    train_svm(feature_train=feature_train, target_train=target_train,
              feature_test=feature_test, target_test=target_test)

    print("\n" + "=" * 50)
    print("训练神经网络...")
    print("=" * 50)
    train_neural_network(feature_train=feature_train, target_train=target_train,
                         feature_test=feature_test, target_test=target_test)
