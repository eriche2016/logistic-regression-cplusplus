1. Data：存放训练数据和测试数据的目录

2. LROne：第一个logistic regression实现，实现的二分类分类器，要求输入特征必须为0-1向量，输出为0-1值

2. LRTwo：第二个logistic regression实现，实现的二分类分类器，要求输入特征必须为归一化的实数值，归一化范围[-1, 1]或者[0, 1]都OK，但不能太大，否则sigmoid函数计算在有限精度内会取值为0.0或者1.0，导致cost function的值为无穷大。