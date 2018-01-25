from tqdm import trange
import numpy as np
import pandas as pd
from collections import Counter
import configparser
import sys

def get_ebay2gg_cate_table(ebay_cate_path,leaf_id2gpcid_path,gpc_id2name_path,\
						   ebay_cate_sep=',',leaf_id2gpcid_sep='\t',gpc_id2name_sep='\t',\
						   save_path=None):
	
	#读取 ebay category数据,获得 us 数据
	df = pd.read_csv(ebay_cate_path,sep=ebay_cate_sep)
	df = df[df["site_id"] == 0]

	#获取leaf_id2gpcid表的 us 数据
	df2 = pd.read_csv(leaf_id2gpcid_path,sep=leaf_id2gpcid_sep)
	df2 = df2.query('site_id==0')

	#读取gpc_id2name 的 us 数据
	df3 = pd.read_csv(gpc_id2name_path,sep=gpc_id2name_sep)

	# select df.*,df2.gpc_id from df left join df2 on df.leaf_categ_id=df2.leaf_categ_id
	# 过滤掉gpc_id为空的
	df_new = pd.merge(df,df2[["gpc_id"]],how='right',left_on=df.leaf_categ_id,right_on=df2.leaf_categ_id)
	filtered_df = df_new[df_new['leaf_categ_id'].notnull()]

	# df_finn为处理后的表，包含ebay_cate 和 gg_cate一一对应关系
	df_fin = pd.merge(filtered_df,df3[["GPC_NAME"]],how='left',left_on=filtered_df.gpc_id,right_on=df3.GPC_ID)
	df_finn = df_fin[['leaf_categ_name','GPC_NAME']]
	df_finn =df_finn[df_finn['GPC_NAME'].notnull()]

	df_finn.to_csv(save_path,header=True,index=False)
	print("table has been saved in {}".format(save_path))




if __name__ == '__main__':
	cf = configparser.ConfigParser()
	if len(sys.argv)!=2:
		print("please inpyt config_file path,such as python xxx.py ./config.ini")
		sys.exit()
	config_path = sys.argv[1]
	cf.read(config_path)
	
	


	ebay_cate_path = cf.get("file_path", "ebay_cate_path")
	leaf_id2gpcid_path = cf.get("file_path", "leaf_id2gpcid_path")
	gpc_id2name_path = cf.get("file_path", "gpc_id2name_path")
	save_path = cf.get("file_path", "save_path")
	get_ebay2gg_cate_table(ebay_cate_path,leaf_id2gpcid_path,gpc_id2name_path,save_path=save_path)

