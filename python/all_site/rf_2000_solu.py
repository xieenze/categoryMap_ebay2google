import gensim
from gensim import utils

import random
import os
from collections import Counter
from pprint import pprint
import numpy as np
import pandas as pd
from tqdm import trange
import configparser,sys


from imblearn.combine import SMOTEENN 
from imblearn.over_sampling import RandomOverSampler,SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from imblearn.metrics import classification_report_imbalanced
from imblearn.over_sampling import ADASYN 
from imblearn.ensemble import BalanceCascade
import imblearn

import nltk.stem as stem
from nltk.tokenize import word_tokenize
from nltk.tokenize import wordpunct_tokenize

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn import random_projection


def getData(ebay2gg_path,gpcid2name_path,ebay2gg_sep=',',gpcid2name_sep='\t'):
	'''
	X是ebay cate，如 Jewelry & Watches:Vintage & Antique Jewelry:Fine:Designer, Signed:Rings
	y是对应google cate的label(类别),在0-1997之间的数
	d是一个dict，key为label值,value为对应的google cate的名字 如  {0: 'Animals & Pet Supplies'}
	'''
	df_finn = pd.read_csv(ebay2gg_path,sep=ebay2gg_sep)
	df_gg2id = pd.read_csv(gpcid2name_path,sep=gpcid2name_sep)
	gg_cate = list(df_gg2id['GPC_NAME'])
	sql = 'GPC_NAME=="need_replace"'
	
	X=[]
	y=[]
	d=[]
	X_1 = []
	y_1 = []
	label=0
	print("data loading!")
	for i in trange(len(gg_cate)):
	    new_sql = sql.replace("need_replace",gg_cate[i])
	    res_list = list(df_finn.query(new_sql)['leaf_categ_name'])
	    #如果样本数量大于等于3
	    #小于3的类别不参与预测，只参与训练
	    if len(res_list)>=3:
	        d.append([label,gg_cate[i]])
	        for j in res_list:
	            X.append(j)
	            y.append(label)
	        label+=1
	    elif len(res_list)>0 and len(res_list)<3:
	        d.append([label,gg_cate[i]])
	        for j in res_list:
	            X_1.append(j)
	            y_1.append(label)
	        label+=1
	d=dict(d)
	#X,y=shuffle(X,y)
	'''打乱数据，分割训练，验证集，可以设置按每一类比例随机抽（stratify =y）'''
	#X_train , X_test , y_train, y_test = train_test_split(X,y,test_size=0.2)
	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,stratify=y,shuffle=True) 

	X_train = np.concatenate((X_train,X_1),axis=0)
	y_train = np.concatenate((y_train,y_1),axis=0)
	#X_train , X_test , y_train, y_test = train_test_split(X,y,test_size=0.2,stratify =y) 
	print("data loaded!")
	return X_train,X_test,y_train,y_test,d


def save_tocsv(ebay_cate,pred_gg_cate,save_path,save_mode):
	'''
	输入 ebay category数组，和predict的google category数组，和保存的路径
	将结果一一对应保存下来
	要求len(ebay category) == len(pred gg category)
	example:
	X_test = ['a','b','c']
	y_pred = model.pred(X_test)
	save_tocsv(X_test,y_pred,'../csv/result.csv')
	'''

	
	print("saving result to csv file....")
	cf = configparser.ConfigParser()
	config_path = sys.argv[1]
	cf.read(config_path)
	ebay_path = cf.get('file_path','ebay_cate_path')
	
	'''
	模式1：保存结果为3列[leaf_id,site_id,gpc_id]
	'''
	if save_mode == "id":
		df_ebay  = pd.read_csv(ebay_path,sep=',')

		def get_leafid_siteid(df,leaf_name):
			if "'" in leaf_name:
				arr = df.query('leaf_categ_name=="{}"'.format(leaf_name))[['leaf_categ_id','site_id']].values
			else:
				arr = df.query("leaf_categ_name=='{}'".format(leaf_name))[['leaf_categ_id','site_id']].values
			leaf_id = arr[0][0]
			site_id = arr[0][1]
			return [leaf_id,site_id]



		arr = [[get_leafid_siteid(df_ebay,ebay_cate[i])[0],\
				get_leafid_siteid(df_ebay,ebay_cate[i])[1],\
				pred_gg_cate[i][0] ] for i in range(len(pred_gg_cate))]



		#arr = [[ebay_cate[i],pred_gg_cate[i]] for i in range(len(pred_gg_cate))]
		df = pd.DataFrame(arr,columns=['ebay_category_id',"site_id","pred_gpc_id"])
		df.to_csv(save_path,header=True,index=False)
		print("csv file is saved to {}".format(os.path.abspath(save_path)))
	'''
	模式2 保存结果为2列 [leaf_name,gpc_name]
	'''
	if save_mode == "name":
		arr = [[ebay_cate[i],pred_gg_cate[i][1]] for i in range(len(pred_gg_cate))]
		df = pd.DataFrame(arr,columns=['ebay_category',"pred_google_category"])
		df.to_csv(save_path,header=True,index=False)
		print("csv file is saved to {}".format(os.path.abspath(save_path)))

	else:
		print("save_mode should be set in one of [id,name],else nothing will be saved")






class rf_2000_model(object):
	'''
	构建pipeline，包括了tfidf抽取特征,降维和 rf分类
	'''
	def __init__(self,d,n_estimators=150):
		
		self.pipeline = make_pipeline(TfidfVectorizer(),\
                         random_projection.SparseRandomProjection(),\
                         RandomForestClassifier(n_estimators=n_estimators,n_jobs=-1,oob_score=True))
     
		self.d = d
	def train(self,X_train, y_train):
		print("training...")
		self.pipeline.fit(X_train, y_train)
		print("training complete!")
	def validation(self,X_val,y_val):
		from sklearn.metrics import classification_report
		y_pred = self.pipeline.predict(X_val)
		return classification_report(y_val, y_pred)
	def cross_valicadion_score(self,X_train, y_train):
		from sklearn.cross_validation import cross_val_score
		#pipeline = make_pipeline(TfidfVectorizer(),RandomForestClassifier(n_estimators=20,n_jobs=8,verbose=0))
		scores = cross_val_score(self.pipeline, X_train, y_train,n_jobs=1,cv=5,verbose=1)
		return scores.mean()

	def predict(self,X_test):
		label_pred = self.pipeline.predict(X_test)
		cate_pred = [[i,self.d[i]] for i in label_pred]
		return cate_pred
	def score(self,X_test,y_test):
		return self.pipeline.score(X_test,y_test)

	def save(self,save_path,compress=3):
		from sklearn.externals import joblib
		joblib.dump(self.pipeline, save_path,compress=3)
		print("model is saved in {}".format(os.path.abspath(save_path)))

	def load_model(self,save_path):
		'''
		load 现成的model,初始化时指定n_estimators=150就无意义。
		那个model的参数是多少就是多少。
		'''
		from sklearn.externals import joblib
		print("loading exist model from {}...".format(os.path.abspath(save_path)))
		#model = rf_2000_model(self.d)
		#model.pipeline = joblib.load(save_path)
		self.pipeline = joblib.load(save_path)
		print("model loaded!")
		#return model

	
   



if __name__=='__main__':
	cf = configparser.ConfigParser()
	if len(sys.argv)!=2:
		print("please input config_file path,such as python xxx.py ./config.ini")
		sys.exit()
	config_path = sys.argv[1]
	cf.read(config_path)


	ebay2gg_path = cf.get('rf_2000_model','ebay2gg_path')
	gpcid2name_path = cf.get('rf_2000_model','gpcid2name_path')
	res_save_path = cf.get('rf_2000_model','res_save_path')
	save_mode = cf.get('rf_2000_model','save_mode')


	X_train,X_test,y_train,y_test,d = getData(ebay2gg_path,gpcid2name_path)
	
	#model = rf_2000_model(d,n_estimators=20)
	# model.train(X_train, y_train)
	#metrics = model.validation(X_test,y_test)
	#mean_score = model.cross_valicadion_score(X_train, y_train)
	#print(mean_score)
	# y_pred = model.predict(X_test)
	# for i in range(len(y_pred)):
	# 	print(X_test[i])
	# 	print(y_pred[i])
	# 	
	# 	
	# print(model.score(X_test,y_test))

	# model.save('../model/rf6.pkl',compress=3)

	model = rf_2000_model(d)
	model.load_model('../model/rf.pkl')
	y_pred = model.predict(X_test)
	#print(model_1.score(X_test,y_test))
	# metrics = model.validation(X_test,y_test)
	# print(metrics)

	#model.save('../model/rf.pkl',compress=3)
	save_tocsv(X_test,y_pred,res_save_path,save_mode)
