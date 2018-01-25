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
先将ebay category分类到google root category上，再在root category下寻找最相关的leaf category.
算法流程：先通过tfidf提取ebay category的稀疏特征，并用随机森林分21类。之前数据分析出google root category一共有21类。再对于某个root category
通过tf-idf建立相似度匹配模型，用root category下的所有leaf category做语料库。返回top K个最相似的google category.

2 rf_2000_solu(直接分2000类)
数据分析出google 5k+ category中实际只采用了2000 category做mapping，因此直接建立随机森林end2end分2000类，特征抽取方法同1，采用tf-idf.

详细的背景和解决方案等信息可见ppt

数据处理部分：
data_process.py 脚本 和 config.ini文件  接收3张表的路径，保存一张表的数据。
三张表分别是：
1 ebay category表
2 ebay category_id map google category_id 表
3 google category_id map name 表
保存的表为：
ebay category name map google category name表。
