from sklearn.metrics import roc_auc_score
import matplotlib.pylab as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import numpy as np


# y_true = [0, 0, 1, 1, 1]
# y_score = [0.1, 0.2, 0.7, 0.8, 0.9]
# print(roc_auc_score(y_true, y_score))
#
# y_score = [0.7, 0.8, 0.9, 0.1, 0.2]
# print(roc_auc_score(y_true, y_score))


def readfromtxt(filename):
    res = []
    f = open(filename, "r")
    for item in f.readlines():
        item = item.strip('\n')
        res.append(float(item))
    f.close()
    return res

def ro_curve_auc(y_pred, y_label, figure_file, method_name):
    '''
        y_pred is a list of length n.  (0,1)
        y_label is a list of same length. 0/1
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
    '''
    y_label = np.array(y_label)
    y_pred = np.array(y_pred)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr[0], tpr[0], _ = roc_curve(y_label, y_pred)
    roc_auc[0] = auc(fpr[0], tpr[0])
    lw = 2
    plt.plot(fpr[0], tpr[0],
         lw=lw, label= method_name + ' (area = %0.2f)' % roc_auc[0])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    # plt.xticks(font="Times New Roman",size=18,weight="bold")
    # plt.yticks(font="Times New Roman",size=18,weight="bold")
    fontsize = 14
    plt.xlabel('False Positive Rate', fontsize = fontsize)
    plt.ylabel('True Positive Rate', fontsize = fontsize)
    #plt.title('Receiver Operating Characteristic Curve', fontsize = fontsize)
    plt.legend(loc="lower right")
    plt.savefig(figure_file + ".pdf")
    return

def ro_curve_aupr(y_pred, y_label, figure_file, method_name):
    '''
        y_pred is a list of length n.  (0,1)
        y_label is a list of same length. 0/1
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
    '''
    y_label = np.array(y_label)
    y_pred = np.array(y_pred)
    lr_precision, lr_recall, _ = precision_recall_curve(y_label, y_pred)
#   plt.plot([0,1], [no_skill, no_skill], linestyle='--')
    plt.plot(lr_recall, lr_precision, lw = 2, label= method_name + ' (area = %0.2f)' % average_precision_score(y_label, y_pred))
    fontsize = 14
    plt.xlabel('Recall', fontsize = fontsize)
    plt.ylabel('Precision', fontsize = fontsize)
    plt.title('Precision Recall Curve')
    plt.legend()
    plt.savefig(figure_file)
    return

# 读取数据
svm_y = readfromtxt("svm_y.txt")
svm_pre = readfromtxt("svm_pre.txt")
lstm_y = readfromtxt("lstm_y.txt")
lstm_pre = readfromtxt("lstm_pre.txt")

# 画auc
ro_curve_auc(svm_pre, svm_y, "auc_val_1","Fold svm")
ro_curve_auc(lstm_pre, lstm_y, "auc_val_1","Fold lstm")
plt.close()  # 清除缓存

# 画aupr
ro_curve_aupr(svm_pre,svm_y,"aupr_val_1","Fold svm")
ro_curve_aupr(lstm_pre,lstm_y,"aupr_val_1","Fold lstm")