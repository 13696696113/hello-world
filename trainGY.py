from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
from sklearn import metrics
import pickle as pkl

from utils import *
from models import GNN, MLP

# Set random seed
# seed = 123
# np.random.seed(seed)
# tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'bio', 'Dataset string.')  # 'mr','ohsumed','R8','R52'
flags.DEFINE_string('model', 'gnn', 'Model string.') 
flags.DEFINE_float('learning_rate', 0.005, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('batch_size', 128, 'Size of batches per epoch.')
flags.DEFINE_integer('input_dim', 75, 'Dimension of input.')
flags.DEFINE_integer('hidden', 16, 'Number of units in hidden layer.') # 32, 64, 96, 128
flags.DEFINE_integer('steps', 2, 'Number of graph layers.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0, 'Weight for L2 loss on embedding matrix.') # 5e-4
flags.DEFINE_integer('early_stopping', -1, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.') # Not used

# Load data
train_adj, train_feature, train_y, val_adj, val_feature, val_y, test_adj, test_feature, test_y = load_data(FLAGS.dataset)
print("train_y", train_y)
print("type(train_y)", type(train_y))
print("test_y.shape", len(test_y))
print("train_feature.shape", len(train_feature))
print("train_y.shape", len(train_y))

print("test_y[1]: ", type(test_y[1]))
print("test_y[2]: ", test_y[2])

train_mainv, val_mainv, test_mainv, pred_mainv = load_data_csl()
# train_mainv = train_y
# val_mainv = val_y
# test_mainv = test_y
print("train_y.shape: ", train_y.shape)
print("train_mainv.shape: ", train_mainv.shape)
print("val_y.shape: ", val_y.shape)
print("val_mainv.shape: ", val_mainv.shape)
print("test_y.shape: ", test_y.shape)
print("test_mainv.shape: ", test_mainv.shape)
print("pred_mainv.shape: ", pred_mainv.shape)

pred_adj, pred_feature, pred_y = load_pred_csl(FLAGS.dataset)

print("pred_y.shape: ", pred_y.shape)

# load增加20向量
# train_mainv, val_mainv, test_mainv, train_adj, train_feature, train_y, val_adj, val_feature, val_y, test_adj, test_feature, test_y = load_data(FLAGS.dataset)  # csl



# Some preprocessing
print('loading training set')
train_adj, train_mask = preprocess_adj(train_adj)
train_feature = preprocess_features(train_feature)
print('loading validation set')
val_adj, val_mask = preprocess_adj(val_adj)
val_feature = preprocess_features(val_feature)
print('loading test set')
test_adj, test_mask = preprocess_adj(test_adj)
test_feature = preprocess_features(test_feature)

print('loading pred set')
pred_adj, pred_mask = preprocess_adj(pred_adj)
pred_feature = preprocess_features(pred_feature)


if FLAGS.model == 'gnn':
    # support = [preprocess_adj(adj)]
    # num_supports = 1
    model_func = GNN
elif FLAGS.model == 'gcn_cheby': # not used
    # support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GNN
elif FLAGS.model == 'dense': # not used
    # support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))


# Define placeholders
placeholders = {
    'support': tf.placeholder(tf.float32, shape=(None, None, None)),
    'features': tf.placeholder(tf.float32, shape=(None, None, FLAGS.input_dim)),
    'mask': tf.placeholder(tf.float32, shape=(None, None, 1)),
    'labels': tf.placeholder(tf.float32, shape=(None, train_y.shape[1])),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
    # 占位符增加mainv（20向量）
    'mainv': tf.placeholder(tf.float32, shape=(None, 20)),   # csl
    # 'mainweight': tf.placeholder("float", shape=(2,2))    # csl_new
}


# label smoothing
# label_smoothing = 0.1
# num_classes = y_train.shape[1]
# y_train = (1.0 - label_smoothing) * y_train + label_smoothing / num_classes


# Create model
model = model_func(placeholders, input_dim=FLAGS.input_dim, logging=True)

# Initialize session
sess = tf.Session()

# merged = tf.summary.merge_all()
# writer = tf.summary.FileWriter('logs/', sess.graph)

# Define model evaluation function
# def evaluate(features, support, mask, labels, placeholders):
def evaluate(features, support, mask, labels, mainv, placeholders):  # csl
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, mask, labels, placeholders)
    # feed_dict增加mainv（20向量）
    feed_dict_val.update({placeholders['mainv']: mainv})   # csl

    outs_val = sess.run([model.loss, model.accuracy, model.embeddings, model.preds, model.labels], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test), outs_val[2], outs_val[3], outs_val[4]


def predict(features1, support1, mask1, test_y1, mainv1):  # csl_new
    ckpt = tf.train.get_checkpoint_state(r'/home/gaoy/TextING-bio//model' + '/')  # ckpt地址
    saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
    with tf.Session() as sess:
        saver.restore(sess, ckpt.model_checkpoint_path)
        graph = tf.get_default_graph()
        # print([n.name for n in graph.as_graph_def().node])
        features = graph.get_tensor_by_name('Placeholder_1:0')
        support = graph.get_tensor_by_name('Placeholder:0')
        mask = graph.get_tensor_by_name('Placeholder_2:0')
        labels = graph.get_tensor_by_name('Placeholder_3:0')
        mainv = graph.get_tensor_by_name('Placeholder_5:0')
        feed_dict = {
            features: features1,
            support: support1,
            mask: mask1,
            labels: test_y1,
            mainv: mainv1,
        }
        # 根据需要配置输出
        #
        # run
        predict = sess.run([model.preds], feed_dict=feed_dict)

    return predict[0]

# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []
best_val = 0
best_epoch = 0
best_acc = 0
best_cost = 0
test_doc_embeddings = None
preds = None
labels = None

# 网络模型全部定义完整后，创建saver实例，保存所有变量
saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

print('train start...')
# Train model

rem_pre = []
rem_rec = []
rem_f1 = []

for epoch in range(FLAGS.epochs):
    t = time.time()
        
    # Training step
    indices = np.arange(0, len(train_y))
    np.random.shuffle(indices)
    print("indices: ", indices)
    print("indices len: ", len(indices))
    
    train_loss, train_acc = 0, 0
    for start in range(0, len(train_y), FLAGS.batch_size):
        end = start + FLAGS.batch_size
        idx = indices[start:end]
        # Construct feed dictionary
        feed_dict = construct_feed_dict(train_feature[idx], train_adj[idx], train_mask[idx], train_y[idx], placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        # feed_dict增加mainv（20向量）
        feed_dict.update({placeholders['mainv']: train_mainv[idx]})   # csl
        #print("feed_dict",feed_dict)

        # feed_dict以字典的方式填充占位placeholders，即替换输入
        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
        train_loss += outs[1]*len(idx)
        train_acc += outs[2]*len(idx)
    train_loss /= len(train_y)
    train_acc /= len(train_y)

    # Validation
    # val_cost, val_acc, val_duration, _, _, _ = evaluate(val_feature, val_adj, val_mask, val_y, placeholders)
    # Validation中加入mainv
    val_cost, val_acc, val_duration, _, _, _ = evaluate(val_feature, val_adj, val_mask, val_y, val_mainv, placeholders)   # csl
    cost_val.append(val_cost)  # val损失
    
    # Test
    # test_cost, test_acc, test_duration, embeddings, pred, labels = evaluate(test_feature, test_adj, test_mask, test_y, placeholders)
    # Test中加入mainv
    #print("test_y: ", test_y)
    test_cost, test_acc, test_duration, embeddings, pred, labels = evaluate(test_feature, test_adj, test_mask, test_y, test_mainv, placeholders)   # csl
    #print("pred: ",pred)
    #print("labels: ", labels)

    # csl 0713 修改 : 记录每个epoch的指标变化
    precision, recall, f_score, _ = metrics.precision_recall_fscore_support(labels, pred, average='macro')
    rem_pre.append(precision)
    rem_rec.append(recall)
    rem_f1.append(f_score)

    # lists = []
    # for index in test_y:
    #     if index == np.array([1,0]):
    #         lists.append(0)
    #     else:
    #         lists.append(1)
    # lists = np.array(lists)
    # print(labels == lists)

    if val_acc >= best_val:
        best_val = val_acc
        best_epoch = epoch
        best_acc = test_acc
        best_cost = test_cost
        test_doc_embeddings = embeddings
        preds = pred

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_loss),
          "train_acc=", "{:.5f}".format(train_acc), "val_loss=", "{:.5f}".format(val_cost),
          "val_acc=", "{:.5f}".format(val_acc), "test_acc=", "{:.5f}".format(test_acc), 
          "time=", "{:.5f}".format(time.time() - t))

    if FLAGS.early_stopping > 0 and epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")

# Best results
print('Best epoch:', best_epoch)
print("Test set results:", "cost=", "{:.5f}".format(best_cost),
      "accuracy=", "{:.5f}".format(best_acc))

print("Test Precision, Recall and F1-Score...")
#print("preds: ",preds)
print(metrics.classification_report(labels, preds, digits=4))
print("Macro average Test Precision, Recall and F1-Score...")
print(metrics.precision_recall_fscore_support(labels, preds, average='macro'))
print("Micro average Test Precision, Recall and F1-Score...")
print(metrics.precision_recall_fscore_support(labels, preds, average='micro'))

# csl 0713 修改 : 计算auc aupr
# 注意，这里的preds元素需要为概率而不是分类结果
print("best_auc: {}".format(metrics.roc_auc_score(labels, preds, average='macro')))
print("best_aupr: {}".format(metrics.average_precision_score(labels, preds)))

# csl 0713 修改 : 把epoch的指标变化存进同目录文件；存储labels preds，供画图
def write2txt(filename, l):
    full_path = FLAGS.dataset + '_' + filename + '.txt'
    f = open(full_path,"w")
    for line in l:
        f.write(str(line)+'\n')
    f.close()

write2txt("rem_pre", rem_pre)
write2txt("rem_rec", rem_rec)
write2txt("rem_f1", rem_f1)

write2txt("labels", labels)
write2txt("preds", preds)

# # 保存模型
# saver=tf.train.Saver(max_to_keep=1)
# saver.save(sess,'D:/桌面文件/数模/确认：D题/模型/TextING-master/model/my-model',global_step=10000)

# 保存模型
path = saver.save(sess, "/home/gaoy/TextING-bio/model/my-model.ckpt", global_step=200)
print("Saved model checkpoint to {}\n".format(path))

pred_res = predict(pred_feature, pred_adj, pred_mask, pred_y, pred_mainv)
print("pred: ",pred_res)
print("len(pred): ", len(pred_res))

'''
# For visualization
doc_vectors = []
for i in range(len(test_doc_embeddings)):
    doc_vector = test_doc_embeddings[i]
    doc_vector_str = ' '.join([str(x) for x in doc_vector])
    doc_vectors.append(str(np.argmax(test_y[i])) + ' ' + doc_vector_str)

doc_embeddings_str = '\n'.join(doc_vectors)
with open('data/' + FLAGS.dataset + '_doc_vectors.txt', 'w'):
    f.write(doc_embeddings_str)
'''
