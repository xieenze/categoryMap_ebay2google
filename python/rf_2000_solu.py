import gensim
from gensim import utils

import random
import os
from collections import Counter
from pprint import pprint
import numpy as np
import pandas as pd
from tqdm import trange

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
from sklearn.cross_validation import cross_val_score

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
	label=0

	print("data loading!")
	for i in trange(len(gg_cate)):
	    new_sql = sql.replace("need_replace",gg_cate[i])
	    #查询结果
	    res_list = list(df_finn.query(new_sql)['leaf_categ_name'])
	    #如果样本数量大于0就添加
	    if len(res_list)>0:
	        d.append([label,gg_cate[i]])
	        for j in res_list:
	            X.append(j)
	            y.append(label)
	        label+=1
	d=dict(d)
	X,y=shuffle(X,y)
	'''打乱数据，分割训练，验证集，可以设置按每一类比例随机抽（stratify =y）'''
	X_train , X_test , y_train, y_test = train_test_split(X,y,test_size=0.2)
	#X_train , X_test , y_train, y_test = train_test_split(X,y,test_size=0.2,stratify =y) 
	print("data loaded!")
	return X_train,X_test,y_train,y_test,d


def save_tocsv(ebay_cate,pred_gg_cate,save_path):
	arr = [[ebay_cate[i],pred_gg_cate[i]] for i in range(len(pred_gg_cate))]
	df = pd.DataFrame(arr,columns=['ebay_category',"pred_google_category"])
	df.to_csv(save_path,header=True,index=False)
	print("csv file is saved to {}".format(save_path))



class rf_2000_model(object):
	'''
	构建pipeline，包括了tfidf抽取特征和 rf分类
	'''
	def __init__(self,d):
		self.pipeline = make_pipeline(TfidfVectorizer(),RandomForestClassifier(n_estimators=20,n_jobs=8,verbose=0))
		self.d = d
	def train(self,X_train, y_train):
		self.pipeline.fit(X_train, y_train)

	def validation(self,X_val,y_val):
		from sklearn.metrics import classification_report
		y_pred = self.pipeline.predict(X_val)
		return classification_report(y_val, y_pred)
	def cross_valicadion_score(self,X_train, y_train):
		#pipeline = make_pipeline(TfidfVectorizer(),RandomForestClassifier(n_estimators=20,n_jobs=8,verbose=0))
		scores = cross_val_score(self.pipeline, X_train, y_train,n_jobs=1,cv=5,verbose=1)
		return scores.mean()

	def predict(self,X_test):
		label_pred = self.pipeline.predict(X_test)
		cate_pred = [self.d[i] for i in label_pred]
		return cate_pred


if __name__=='__main__':
	ebay2gg_path = '../data/ebay2gg_table'
	gpcid2name_path = '../data/gpc_id2name.tsv'
	X_train,X_test,y_train,y_test,d = getData(ebay2gg_path,gpcid2name_path)
	
	model = rf_2000_model(d)
	model.train(X_train, y_train)
	metrics = model.validation(X_test,y_test)
	#mean_score = model.cross_valicadion_score(X_train, y_train)
	#print(mean_score)
	y_pred = model.predict(X_test[:1000])
	# for i in range(len(y_pred)):
	# 	print(X_test[i])
	# 	print(y_pred[i])
	#print(metrics)
	save_tocsv(X_test,y_pred,'../data/demo.csv')
