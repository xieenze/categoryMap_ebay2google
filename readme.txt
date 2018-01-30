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


方案二具体程序说明：
数据处理部分：
data_process.py 脚本 和 config.ini文件  接收3张表的路径，保存一张表的数据。
三张表分别是：
1 ebay category表
2 ebay category_id map google category_id 表
3 google category_id map name 表
保存的表为：
ebay category name map google category name表。


配置文件config.ini 解释
所有参数不需要加引号。
[file_path]下四个参数为数据预处理的四张表的路径
1 ebay_cate_path 是 ebay_category 表，包含四个字段[leaf_categ_id,site_id,move_to,leaf_categ_name]
2 leaf_id2gpcid_path是leaf_id2gpcid表，包含三个字段[leaf_categ_id,site_id,gpc_id]
3 gpc_id2name_path是gpc_id2name表，包含两个字段[GPC_ID,GPC_NAME]
4 save_path 是处理结果存储的位置，结果表包含两个字段，[leaf_categ_name,GPC_NAME]

[rf_2000_model]下四个参数为建模所需的四个参数
1 ebay2gg_path 为数据预处理保存的结果表的路径，结果表包含两个字段，[leaf_categ_name,GPC_NAME]
2 gpc_id2name_path是gpc_id2name表，包含两个字段[GPC_ID,GPC_NAME]
3 res_save_path是模型输出预测结果存储的位置
4 save_mode 是模型输出结果的种类，一共为两种mode，[id,name] 按id存储，输出结果为[leaf_id,site_id,gpc_id]三列，按name存储，输出结果为[leaf_name,gpc_name]
两列


如何运行程序
1 cd conf 并修改配置文件的参数值
2 cd python
3 运行数据预处理程序  python us/data_preprocess.py ../conf/config.ini
4 运行模型程序   python us/rf_2000_model.py ../conf_config.ini