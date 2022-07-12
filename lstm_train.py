import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
# import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score, roc_auc_score, average_precision_score
import time

dtype = torch.FloatTensor
embedding_dim = 50
n_hidden = 5
num_classes = 2  # 0 or 1
# sentences = ["i love you", "he loves me", "she likes baseball", "i hate you", "sorry for that", "this is awful"]
# labels = [1, 1, 1, 0, 0, 0]

adv_data = pd.read_csv("C:/Users/22174/Desktop/qq.csv")
X = adv_data.iloc[:,0].values
labels = adv_data.iloc[:,5].values
sentences = []
for item in X:
       string = ""
       for tmp in item:
              string += tmp + " "
       sentences.append(string)

# 构建词表，把数据集中出现过的词拿出来并给它一个编号
word_list = " ".join(sentences).split()
word_list = list(set(word_list))
word_dict = {w: i for i, w in enumerate(word_list)}
vocab_size = len(word_dict)


# 定义输入输出
inputs = []
for sen in sentences:
    xxx = np.asarray([word_dict[n] for n in sen.split()])
    xxx = xxx.tolist()
    if len(xxx) < embedding_dim:
        for i in range(len(xxx), embedding_dim):
            xxx.append(0)
    else:
        tmp = xxx[:embedding_dim]
        for i in range(embedding_dim):
            for j in range(i+50, len(xxx), 50):
                tmp[i] = abs(xxx[j]-tmp[i])
        xxx = tmp
    xxx = np.array(xxx)
    inputs.append(xxx)

targets = []
for out in labels:
    targets.append(out)

input_batch = Variable(torch.LongTensor(inputs))
target_batch = Variable(torch.LongTensor(targets))

# print(input_batch)
print(input_batch.shape)
# print(target_batch.shape)

# 构造模型
class BiLSTM_Attention(nn.Module):
    def __init__(self):
        super(BiLSTM_Attention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, n_hidden, bidirectional=True)
        self.out = nn.Linear(n_hidden * 2, num_classes)
    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, n_hidden * 2, 1)
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights.data.numpy()

    def forward(self, X):
        input = self.embedding(X)
        input = input.permute(1, 0, 2)
        hidden_state = Variable(torch.zeros(1*2, len(X), n_hidden))
        cell_state = Variable(torch.zeros(1*2, len(X), n_hidden))
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (hidden_state, cell_state))
        output = output.permute(1, 0, 2)
        attn_output, attention = self.attention_net(output, final_hidden_state)
        return self.out(attn_output), attention

# 模型实例化，并确定损失函数，优化器
start = time.time()
model = BiLSTM_Attention()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练并测试
best_f1 = 0
best_rec = 0
best_acc = 0
best_pre = 0
best_epoch = 0
best_auc = 0
best_aupr = 0
last_attention = []
all_time = []
for epoch in range(250):
    tmp_time = time.time()
    optimizer.zero_grad()
    output, attention = model(input_batch)
    pred = output.data.max(1, keepdim=True)[1]
    # print(f"均方误差(MSE)：{mean_squared_error(pred, target_batch)}")
    acc = accuracy_score(target_batch, pred)
    pre = precision_score(target_batch, pred, average='macro')
    rec = recall_score(target_batch, pred, average='macro')
    f1 = f1_score(target_batch, pred, average='macro')
    auc = roc_auc_score(target_batch, pred, average='macro')
    aupr = average_precision_score(target_batch, pred)
    if f1 > best_f1:
        best_f1 = f1
        best_rec = rec
        best_acc = acc
        best_pre = pre
        last_attention = attention
        best_epoch = epoch
        best_auc = auc
        best_aupr = aupr
    # print("accuracy_score", acc)
    # print("precision_score", pre)
    # print("recall_score", rec)
    # print("f1_score", f1)
    loss = criterion(output, target_batch)
    if (epoch + 1) % 5 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
    loss.backward()
    optimizer.step()
    all_time.append(time.time()-tmp_time)


print("best_epoch", best_epoch)
print("best_accuracy_score", best_acc)
print("best_precision_score", best_pre)
print("best_recall_score", best_rec)
print("best_f1_score", best_f1)
print("best_auc", best_auc)
print("best_aupr", best_aupr)
print("time: ", time.time()-start)
atime = 0
for time in all_time:
    atime += time
print('avg_time: {}'.format(atime/len(all_time)))
# Test
# test_text = 'sorry hate you'
# tests = [np.asarray([word_dict[n] for n in test_text.split()])]
# test_batch = Variable(torch.LongTensor(tests))
# # Predict
# predict, _ = model(test_batch)
# predict = predict.data.max(1, keepdim=True)[1]
# if predict[0][0] == 0:
#     print(test_text,"is Bad Mean...")
# else:
#     print(test_text,"is Good Mean!!")

last_attention = last_attention[:50]

xlabel_csl = []
for i in range(50):
    xlabel_csl.append('d'+str(i))
ylabel_csl = []
for i in range(50):
    ylabel_csl.append('batch_'+str(i))

# import matplotlib.ticker as ticker
# fig = plt.figure(figsize=(50, 50))
# ax = fig.gca()
# data = last_attention
# im = ax.imshow(data)
# cb1 = plt.colorbar(im, fraction=0.03, pad=0.05)
# tick_locator = ticker.MaxNLocator(nbins=5)  # colorbar上的刻度值个数
# cb1.locator = tick_locator
# cb1.set_ticks([np.min(data), 0.04, 0.08, 0.12, np.max(data)])
# cb1.update_ticks()
#
# tick1=np.arange(-1, 50, 1)
# tick2=np.arange(-1, 50, 1)
# ax.set_yticks(tick2)
# ax.set_xticks(tick1)
#
# ax.set_xticklabels(['']+xlabel_csl, fontdict={'fontsize': 9}, rotation=90)
# ax.set_yticklabels(['']+ylabel_csl, fontdict={'fontsize': 9})
# #plt.savefig('D:/桌面文件/数模/论文转化/attention_weight.eps',bbox_inches='tight',transparent = True)
# # plt.show()

# xlabel_csl = []
# for i in range(50):
#     xlabel_csl.append('d'+str(i))
# ylabel_csl = []
# for i in range(50):
#     ylabel_csl.append('batch_'+str(i))
#
# fig = plt.figure(figsize=(50, 50)) # [batch_size, n_step]
# ax = fig.add_subplot(1, 1, 1)
# ax.matshow(last_attention, cmap='viridis')
#
# tick1=np.arange(-1, 50, 1)
# tick2=np.arange(-1, 50, 1)
# ax.set_yticks(tick2)
# ax.set_xticks(tick1)
#
# ax.set_xticklabels(['']+xlabel_csl, fontdict={'fontsize': 9}, rotation=90)
# ax.set_yticklabels(['']+ylabel_csl, fontdict={'fontsize': 9})
#
# plt.show()

