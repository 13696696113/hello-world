# hello-world

修改内容：

- 记录了每个epoch的metric保存在同级目录下的文件里
- 最终计算auc aupr，并把labels和preds保存在同级目录下的文件里，供后续画图

注意：

- 所有修改都有 `# csl 0713 修改` 标记
- 要正确计算auc，preds需要为概率值数组而不是分类值数组
