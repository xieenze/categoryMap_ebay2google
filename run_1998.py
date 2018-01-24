import gensim
from gensim import utils

import random
import os
from collections import Counter
from pprint import pprint
import numpy as np
import pandas as pd

from imblearn.over_sampling import RandomOverSampler,SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from imblearn.metrics import classification_report_imbalanced
import imblearn

import nltk.stem as stem
from nltk.tokenize import word_tokenize
from nltk.tokenize import wordpunct_tokenize

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

df_finn = pd.read_csv('./data/ebay2gg_table')
df_gg2id = pd.read_csv('./data/gpc_id2name.tsv',sep="	")
from tqdm import trange

gg_cate = list(df_gg2id['GPC_NAME'])

sql = 'GPC_NAME=="need_replace"'
X=[]
y=[]
label=0
for i in trange(len(gg_cate)):
    new_sql = sql.replace("need_replace",gg_cate[i])
    res_list = list(df_finn.query(new_sql)['leaf_categ_name'])
    
    if len(res_list)>0:
        for j in res_list:
            X.append(j)
            y.append(label)
        label+=1


X_train , X_test , y_train  , y_test = train_test_split(X,y,test_size=0.3) 

#classifer =  RandomForestClassifier(n_estimators=50,n_jobs=2)
#classifer = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60, min_samples_leaf=20, max_features='sqrt', subsample=0.8, random_state=10,verbose=2)
#classifer = LogisticRegression(penalty='l2',verbose=1,n_jobs=8) 
classifer = SVC(kernel='rbf', probability=True,verbose=True) 
pipeline = make_pipeline(TfidfVectorizer(), classifer)
pipeline.fit(X_train, y_train)

print(pipeline.score(X_test,y_test))
