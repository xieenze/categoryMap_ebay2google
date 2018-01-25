ebay category mapping google category
背景：ebay 有12w个商品category,而google只有5000个，ebay要在google上做广告，必须将ebay的12w category map到google 5k category上去。
之前有人工标注过的数据，US的数据大约有39000+条。本project 暂时只基于US数据建模(英语语料)

采用python3编程，依赖一些第三方库
安装库建议：
1 下载并安装anaconda3
2 pip安装 sklearn, imbalanced-learn, nltk, gensim, tqdm，ConfigParser等 
ps:（nltk下载后在命令行import nltk，nltk.download()）下载语料数据
reference：http://blog.csdn.net/elikai/article/details/46848671

我提供了两种大体的解决方法：
1 tfidf_rf_tfidf_solu(先按照google category)
2 rf_2000_solu(直接分2000类)

详细的背景可见ppt

