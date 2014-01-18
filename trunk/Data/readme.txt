Samples.txt
SamplesTest.txt
SamplesTrain.txt
http://komarix.org/ lr_trirls_20060531（LR的c++实现）的验证数据，做了处理，只包含两个分类，共10个特征

SamplesMultClassesTrain.txt
SamplesMultClassesTest.txt
http://www.openpr.org.cn/ openpr-lr_v.0.11工具包（LR的c++实现）的验证数据，包含6个分类，25334个特征
SamplesMultClassesTrainScale.txt
SamplesMultClassesTestScale.txt
在上面两个文件的基础上，将每一维特征做归一化，映射到[0,1]区间之内。


svm_ala_train.txt
svm_ala_test.txt
来自libsvm的数据集，做了一定预处理，预测是否是成年人的数据——a1a，123个特征
http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html


australian.txt
australian_scale.txt
australian_scale_test.txt
australian_scale_train.txt
来自libsvm的数据集，做了一定预处理，australian数据，14个特征