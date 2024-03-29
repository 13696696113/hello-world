# https://www.jianshu.com/p/14b26f59040b

from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer  
from collections import defaultdict

filePath1='D:/CSLWork/lab/hello-world-master/tfidf/ycw.clean.txt'
filePath2='D:/CSLWork/lab/hello-world-master/tfidf/ycw.txt'
text=[]
with open(filePath1, 'r', encoding='utf-8') as fileTrainRaw:
    for line in fileTrainRaw:  # 按行读取文件
        text.append(line.strip())
print(text[1])
label = []
with open(filePath2, 'r', encoding='utf-8') as file:
    for line in file:  # 按行读取文件
        label.append(line.strip().split('\t'))
print(label[1])




# 将train中不同类病例的文本全部合在一起
train_text = {}
test_text = []
test_label = []
for i in range(len(label)):
    if label[i][1] == "train":
        if label[i][2] in train_text:
            train_text[label[i][2]] = train_text[label[i][2]] + "\t" + text[i]
        else:
            train_text[label[i][2]] = text[i]
    if label[i][1] == "test":
        test_text.append(text[i])
        test_label.append(label[i][2])
print(len(train_text), label[10][2], len(train_text[label[10][2]]))
print(len(test_text), test_text[100])
print(len(test_label), test_label[100])




tlist = []
for item in train_text:
    tlist.append(train_text[item])
vectorizer=CountVectorizer()#该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
transformer = TfidfTransformer(smooth_idf = False)#该类会统计每个词语的tf-idf权值
tfidf = transformer.fit_transform(vectorizer.fit_transform(tlist))  #第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵

word=vectorizer.get_feature_names()#获取词袋模型中的所有词语
weight=tfidf.toarray()#将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重

tfidf_list = defaultdict(list)
tfidf_avg = {}
for i in range(len(weight)):#打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for遍历某一类文本下的词语权重
    for j in range(len(word)):
        #获取每个词语对应的权重值
        tfidf_list[word[j]].append(weight[i][j])
#对每个词语求权重值平均
for i,j in tfidf_list.items():
    tfidf_avg[i] = sum(j)/len(j)

print(len(tfidf_avg))
print(tfidf_avg)





# 分析weight[i][j]，找到每类病例最突出的代表性词汇
top_word = []
for i in range(len(weight)):#打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for遍历某一类文本下的词语权重
    max_score = 0
    second_s = 0
    third = 0
    d4 = 0
    d5 = 0
    d6 = 0
    d7 = 0
    d8 = 0
    d9 = 0
    d10 = 0
    tmp = ["", "", "", "", "", "", "", "", "", ""]
    for j in range(len(word)):
        #获取每个词语对应的权重值
        if weight[i][j] > max_score:
            max_score = weight[i][j]
            tmp[0] = word[j]
        elif weight[i][j] > second_s:
            second_s = weight[i][j]
            tmp[1] = word[j]
        elif weight[i][j] > third:
            third = weight[i][j]
            tmp[2] = word[j]
        elif weight[i][j] > d4:
            d4 = weight[i][j]
            tmp[3] = word[j]
        elif weight[i][j] > d5:
            d5 = weight[i][j]
            tmp[4] = word[j]
        elif weight[i][j] > d6:
            d6 = weight[i][j]
            tmp[5] = word[j]
        elif weight[i][j] > d7:
            d7 = weight[i][j]
            tmp[6] = word[j]
        elif weight[i][j] > d8:
            d8 = weight[i][j]
            tmp[7] = word[j]
        elif weight[i][j] > d9:
            d9 = weight[i][j]
            tmp[8] = word[j]
        elif weight[i][j] > d10:
            d10 = weight[i][j]
            tmp[9] = word[j]
            
    top_word.append(tmp)

print(len(top_word), top_word)
for item in weight[10]:
    if item > 0:
        print(item)
        
        
        
        
 ache = []
for item in train_text:
    ache.append(item)
print(len(ache))

F = 0
T = 0
for i in range(len(test_label)):
    maxOne = 0
    for j in range(len(top_word)):
        num = 0
        for word in top_word[j]:
            if word in test_text[i]:
                num += 1
        if num > maxOne:
            maxOne = num
            res = j
    if test_label[i] != ache[res]:
        F += 1
    else:
        T += 1
        
print(F, T)
print(T/(F+T))

# 3
# 10253 1653
# 0.13883756089366706
# 5
# 9353 2553
# 0.21442969931127162
# 10
# 9009 2897
# 0.24332269443977828






# top3
F3 = 0
T3 = 0
for i in range(len(test_label)):
    maxOne = 0
    d2One = 0
    d3One = 0
    for j in range(len(top_word)):
        num = 0
        for word in top_word[j]:
            if word in test_text[i]:
                num += 1
        if num > maxOne:
            maxOne = num
            res1 = j
        elif num > d2One:
            d2One = num
            res2 = j
        elif num > d3One:
            d3One = num
            res3 = j
    if test_label[i] != ache[res1] and test_label[i] != ache[res2] and test_label[i] != ache[res3]:
        F3 += 1
    else:
        T3 += 1
        
print(F3, T3)
print(T3/(F3+T3))




# top5
F5 = 0
T5 = 0
for i in range(len(test_label)):
    maxOne = 0
    d2One = 0
    d3One = 0
    d4One = 0
    d5One = 0
    for j in range(len(top_word)):
        num = 0
        for word in top_word[j]:
            if word in test_text[i]:
                num += 1
        if num > maxOne:
            maxOne = num
            res1 = j
        elif num > d2One:
            d2One = num
            res2 = j
        elif num > d3One:
            d3One = num
            res3 = j
        elif num > d4One:
            d4One = num
            res4 = j
        elif num > d5One:
            d5One = num
            res5 = j
    if test_label[i] != ache[res1] and test_label[i] != ache[res2] and test_label[i] != ache[res3] and test_label[i] != ache[res4] and test_label[i] != ache[res5]:
        F5 += 1
    else:
        T5 += 1
        
print(F5, T5)
print(T5/(F5+T5))
