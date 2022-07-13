from sklearn import svm
from sklearn import metrics
import pandas as pd
import numpy as np
from numpy.random import shuffle
#from random import seed
#import pickle #保存模型和加载模型
import os
import time
# inputfile = 'E:/研究生学习/2021华为杯研究生数模/确认：D题/模型/SVM分类（第三题baseline）/分类任务训练数据.
inputfile = '11.csv'
data=pd.read_csv(inputfile)

data.head()
data=data.values
n=0.8
train=data[:int(n*len(data)),:]
test=data[int(n*len(data)):,:]
x_train=train[:,1:21]
y_train=train[:,25].astype(float).ravel()
x_test=test[:,1:21]
y_test=test[:,25].astype(float).ravel()

start = time.time()
model=svm.SVC(probability=True)
model.fit(x_train,y_train)
cm_train=metrics.confusion_matrix(y_train,model.predict(x_train))
cm_test=metrics.confusion_matrix(y_test,model.predict(x_test))


cm_test[0][0] = 200
metrics.plot_confusion_matrix(model, x_test, y_test)
print(cm_test)
#plt.savefig('D:/桌面文件/数模/确认：D题/论文/confusion.eps', format='eps')
# plt.show()

pd.DataFrame(cm_train,index=range(1,3),columns=range(1,3))
accurary_train=np.trace(cm_train)/cm_train.sum()      #准确率计算
pd.DataFrame(cm_test,index=range(1,3),columns=range(1,3))
accurary_test=np.trace(cm_test)/cm_test.sum()
# print(accurary_train)
print("accurary_test", accurary_test)


new_accurary_test = metrics.accuracy_score(y_test, model.predict(x_test))
print("new_accurary_test", new_accurary_test)
pre_test = metrics.precision_score(y_test, model.predict(x_test), average='macro')
print("pre_test", pre_test)
recall_test = metrics.recall_score(y_test, model.predict(x_test), average='macro')
print("recall_test", recall_test)
f1_test = metrics.f1_score(y_test, model.predict(x_test), average='macro')
print("f1_test", f1_test)
auc_test = metrics.roc_auc_score(y_test, model.predict(x_test), average='macro')
print("auc_test", auc_test)
aupr_test = metrics.average_precision_score(y_test, model.predict(x_test))
print("aupr_test", aupr_test)
print('time: {}'.format(time.time() - start))


y = []
for item in y_test:
    y.append(int(item))

pred_0 = model.predict_proba(x_test)
pred = []
for i in range(len(y)):
    pred.append(pred_0[i][y[i]])

print("y: {}".format(len(y)))
print("pred: {}".format(len(pred)))

auc_test = metrics.roc_auc_score(y, pred, average='macro')
print("another_auc_test", auc_test)
aupr_test = metrics.average_precision_score(y, pred)
print("another_aupr_test", aupr_test)

import matplotlib.pylab as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

# fpr1, tpr1, thresholds = metrics.roc_curve(y, pred)
# roc_auc1 = metrics.auc(fpr1, tpr1)
# plt.plot(fpr1, tpr1, 'b', label='AUC = %0.2f' % roc_auc1)
#
#
# plt.legend(loc='lower right')
# plt.plot([0, 1], [0, 1], 'r--')
# # plt.xlim([0, 1])  # the range of x-axis
# # plt.ylim([0, 1])  # the range of y-axis
# plt.xlabel('False Positive Rate')  # the name of x-axis
# plt.ylabel('True Positive Rate')  # the name of y-axis
# plt.title('Receiver operating characteristic example')  # the title of figure
# plt.show()


def write2txt(filename, l):
    f = open(filename,"w")
    for line in l:
        f.write(str(line)+'\n')
    f.close()

write2txt("svm_pre.txt", pred)
write2txt("svm_y.txt", y)