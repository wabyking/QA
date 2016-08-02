#coding=utf8
from __future__ import division
import pandas as pd 
import os
import subprocess
import collections
import numpy as np 
from nltk.metrics import edit_distance
from gensim import models
import jieba
import pynlpir
import urllib2
import random,time,chardet
import cPickle as pickle
from gensim.models.doc2vec import TaggedLineDocument,Doc2Vec
from sklearn.linear_model import LogisticRegression ,LinearRegression  
from sklearn.tree import DecisionTreeClassifier
import math,re
import featureExtract
import evaluation
def splitDatabyDf(df,rate=0.5):
	# df=pd.read_csv(qa_path,header=None,sep="\t",names=["question","answer","flag"],quoting =3)
	#df=done()
	questions= df["question"].unique()

	flags=[True] * int(size*rate) + [False] *  (size-int(size*rate))
	random.seed(822)
	random.shuffle(flags)
	
	trainQuestions= [questions[i] for i in range(len(questions)) if flags[i]==True]
	# reverse_flags=[not item  for item in flags]
	# testQustions= df["question"][reverse_flags]
	train=df[df.question.isin(trainQuestions)]
	test=df[~df.question.isin(trainQuestions)]
	
	return train,test

def splitData(qa_path,rate=0.8):
	df=pd.read_csv(qa_path,header=None,sep="\t",names=["question","answer","flag"],quoting =3)
	return splitDatabyDf(df)



def train_predicted(train,test):
	train,names=featureExtract.getFeatureSofQA(train)
	x=train[names]
	y=train["flag"]

	
	clf = LinearRegression()
	clf.fit(x, y)
	print clf.coef_ 

	test,names=featureExtract.getFeatureSofQA(test)
	test_x=test[names]
	
	predicted=clf.predict(test_x)


	print evaluation.eval(predicted,test)

	return predicted



def main( run_type=3):
	if run_type==1:
		train,test=splitData()
		test_file="test.csv"
		test.to_csv(test_file,index=False,header =False,sep="\t",encoding="utf-8")
		# del test["flag"]

	elif run_type==2:

		train_file="nlpcc-iccpol-2016.dbqa.training-data1"
		test_file="nlpcc-iccpol-2016.dbqa.training-data2"
		train=pd.read_csv(train_file,header=None,sep="\t",names=["question","answer","flag"],quoting =3)
		test=pd.read_csv(test_file,header=None,sep="\t",names=["question","answer","flag"],quoting =3)
	else:

		test_file="data/nlpcc-iccpol-2016.dbqa.testing-data"
		train_file="data/nlpcc-iccpol-2016.dbqa.training-data"
		train=pd.read_csv(train_file,header=None,sep="\t",names=["question","answer","flag"],quoting =3)
		test=pd.read_csv(test_file,header=None,sep="\t",names=["question","answer","flag"],quoting =3)
	
	print "laod data over"
	
	train_predicted(train, test)
	# train=dataPrepare(train)
	
	# test=dataPrepare(test)
	

	# print "data prepare over"
	# # l2r(train, test)
	# # exit()

	# predicted=cls_train_test(train,test)
	# print "train over"
	# test["predicted"]=predicted
	# #test["predicted"].to_csv("predicted")
	
	
	# test["predicted"].to_csv("train.QApair.TJU_IR_QA.score",index=False,sep="\t")
	# print "predict over"
	# evaluationPredict(predicted,test_file)

if __name__=="__main__":

	main()	