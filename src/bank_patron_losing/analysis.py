import numpy as np
from sklearn.metrics import roc_curve #导入ROC曲线函数
import matplotlib.pyplot as plt #导入作图库
from sklearn.metrics import confusion_matrix #导入混淆矩阵函数
from sklearn.model_selection import StratifiedKFold
import joblib


def draw_roc_curve(target_test, predict_results):
    # 绘制ROC曲线
    fpr, tpr, thresholds = roc_curve(target_test, predict_results, pos_label=1)
    plt.figure(figsize=(7,7))
    plt.plot(fpr, tpr, linewidth=2, label = 'ROC curve') #作出ROC曲线
    plt.plot([0,1],[0,1],'k--',label='guess')
    plt.title("ROC Curve",fontsize=25)
    plt.xlabel('False Positive Rate',fontsize=20) #坐标轴标签
    plt.ylabel('True Positive Rate',fontsize=20) #坐标轴标签
    plt.ylim(0,1.05) #边界范围
    plt.xlim(0,1.05) #边界范围
    plt.legend(loc=4,fontsize=20) #图例
    plt.show() #显示作图结果


def draw_confusion_matrix(target_test, predict_results):
    # 绘制混淆矩阵
    cm = confusion_matrix(target_test, predict_results) #混淆矩阵
    plt.figure(figsize=(7, 7))
    plt.matshow(cm, fignum=0,cmap=plt.cm.Blues) 
    plt.colorbar() #颜色标签
    for x in range(len(cm)): #数据标签
        for y in range(len(cm)):
            plt.annotate(cm[x,y], xy=(x, y),fontsize=30, horizontalalignment='center', verticalalignment='center')
    
    plt.ylabel('Hypothesized class',fontsize=20) #坐标轴标签
    plt.xlabel('True class',fontsize=20) #坐标轴标签
    plt.show()


def fold_cross_validation(feature,target,dt_model, n_splits=10):
    skfold = StratifiedKFold(n_splits=n_splits,shuffle=False)

    x_axis=[] ; y_axis=[]
    k=0;max=0;min=100;sum=0
    for train_index,test_index in skfold.split(feature,target):
        k+=1
        skfold_feature_train=feature[train_index]
        skfold_feature_test=feature[test_index]
        skfold_target_train=target[train_index]
        skfold_target_test=target[test_index]
        dt_model.fit(skfold_feature_train,skfold_target_train)
        scores = dt_model.score(skfold_feature_test,skfold_target_test)
        x_axis.append(k)
        y_axis.append(scores)
        if scores>max:
            max=scores
        if scores<min:
            min=scores
        sum+=scores
    avg=sum/k

    # 绘图
    plt.plot(x_axis,y_axis)
    plt.ylim(0.6,0.9)
    plt.xlim(1,n_splits)
    plt.xlabel("Rounds")
    plt.ylabel('True Rate')
    plt.title("KFold Cross Validation (k=%s) avg=%s"%(k,round(avg*100,2))+"%"+" max:"+"%s"%(round(max*100,2))+"%"+" min:"+"%s"%(round(min*100,2))+"%")
    plt.show()


if __name__ == "__main__":
    # 读取数据
    target_test = np.load('./dataset/target_test.npy')
    predict_results = np.load('./dataset/predict_results.npy')
    feature = np.load('./dataset/feature.npy')
    target = np.load('./dataset/target.npy')
    dt_model = joblib.load('./model/dt_model.pkl')

    draw_roc_curve(target_test, predict_results)
    draw_confusion_matrix(target_test, predict_results)
    fold_cross_validation(feature,target,dt_model)
