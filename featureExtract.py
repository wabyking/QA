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
debug=True
pynlpir.open()
stopwords={word.decode("utf-8") for word in open("model/chStopWordsSimple.txt").read().split()}
print "stop words load over"

w2v= models.word2vec.Word2Vec.load_word2vec_format('model/dbqa.word2vec', binary=False)

types=["number","time","organization","person","place","others"]

#number
pattern_1 = re.compile(u".*(第)?(几|(多(少(?!年)|大|小|(长(?!时间))|短|高|低|矮|远|近|厚|薄))).*")
#time
pattern_2 = re.compile(u".*((哪一?(年|月|天|日)|(什么时(候|间))|(多久)|(时间是)|(多长时间)|(多少年))).*")
#organization
pattern_3 = re.compile(u".*(((什么)|(哪(一|两|三|几|些)?(个|家)?)).{0,3}(班底|前身|团|公司|企业|组织|集团|机构|学校|基地|媒体|开发商|商场|购物中心|工作室)|((班底|前身|团|公司|企业|组织|集团|机构|学校|大学|基地|媒体|开发商|商场|购物中心|工作室)(是|叫))).*")
#person
pattern_4 = re.compile(u".*((什么名字)|((?<!开发商是)谁)|((哪(一|两|三|几|些)?)(个|位)?(人才?|作家|作者|演员|皇帝|主持人|统治者|角色|主角|名字?|主席))|((人|作家|作者|演员|皇帝|主持人|统治者|主席|角色|主角|名字?|子|儿)(是|((叫|演)什么)))).*")
#place
pattern_5 = re.compile(u".*(((什么|哪)(一|两|三|几|些)?(个|座)?(里|地方|地区|国家?|城?市|县|村|州|洲|行政区))|(在哪(?!((一|两|三|几|些)?(场|次|个|集|批|级|部|播出|网站))))|(是哪(?![\u4e00-\u9fa5]))).*")


# patterns= {line.decode("utf-8").split("\t")[0]:line.decode("utf-8").split("\t")[1] for line in open("model/template")}

pattern_number = re.compile(u".*(([\d]+)|([零一二两俩三仨四五六七八九十百千万亿壹贰叁肆伍陆柒捌玖拾佰仟萬]+)).*")
pattern_time = re.compile(u".*(([春夏秋冬])|(((([\d]+)|([零一二两三四五六七八九十百千万亿]+))[年月日天时点刻分秒]))).*")

ner_dict=pickle.load(open("model/ner.dict","rb"))



def cut_sentence(sentence):
	try:
		words= pynlpir.segment(sentence, pos_tagging=False)
	except:
		words= jieba.cut(str(sentence))
	
	words=[word for word in words if word not in stopwords]
	return "#".join(words)

def cut(sentence,debug=debug):
	if debug:
		return [word for word in sentence.split("#")]
	else:
		return cut_sentence(sentence)


def w2vSimilarity(row,model=w2v,idf=None):

	va=model.vocab
	initializer=np.array([0.000001]*model.vector_size)
	question=cut(row["question"])
	answer=cut(row["answer"])
	# print chardet.detect(question[0])
	if idf!=None:
		sentence1= reduce(lambda x,y:x+y, [ model[word]*getIDF(word,idf) for word in question if word in va],initializer)
		sentence2= reduce(lambda x,y:x+y, [ model[word]*getIDF(word,idf) for word in answer if word in va],initializer) 
	else:
	#sentence1= reduce(lambda x,y:x+y, [ model[word.encode("utf-8")] for word in question if word.encode("utf-8") in va],initializer)
	#sentence2= reduce(lambda x,y:x+y, [ model[word.encode("utf-8")] for word in answer if word.encode("utf-8") in va],initializer) 
		sentence1= reduce(lambda x,y:x+y, [ model[word]*((position+1)**3.2)  for position,word in enumerate( question) if word in va],initializer) #*((position+1)**2)
		sentence2= reduce(lambda x,y:x+y, [ model[word] for word in answer if word in va],initializer) 

	#sentence2=sentence2/len(sentence2)
	#print sentence1
	return np.dot(sentence1,sentence2)/(np.linalg.norm(sentence1)*np.linalg.norm(sentence2))


def questionType(sentence):

	sentence=sentence.decode("utf-8")

	if pattern_1.match(sentence):
		num = 0
	elif pattern_2.match(sentence):
		num = 1
	elif pattern_3.match(sentence):
		num = 2
	elif pattern_4.match(sentence):
		num = 3
	elif pattern_5.match(sentence):
		num = 4
	else:
		num = 5

	return types[num]
def typeinfo(row):


	type_array=np.zeros(5)
	question=row["question"]
	answer=str(row["answer"])
	#print question+":",
	q_type=questionType(question)
	# print "%s -> %s " %(question,q_type) ,
	# print answer+":",
 	if q_type=="others":
		return 0
	elif q_type=="number":
		if pattern_number.match(answer.decode("utf-8")) :
			# print "number"
			return 1
		

	elif q_type=="time":

		if pattern_time.match(answer.decode("utf-8")):
			# print "time"
			return 1
	else:
		if ner_dict.has_key(answer):
			
			ner_info= ner_dict[answer]
			# print ner_info,
			if q_type=="organization":
				if "Ni" in ner_info:
					# print "organization"
					return 1
			elif q_type=="person":
				if "Nh" in ner_info:
					# print "person"
					return 1
			elif q_type=="place":
				if "Ns" in ner_info:
					# print "place"
					return 1
		

	return 0

def typeinfo_array(row):


	type_array=np.zeros(5)
	question=row["question"]
	answer=row["answer"]
	#print question+":",
	q_type=questionType(question)
	# print "%s -> %s " %(question,q_type) ,
	# print answer+":",
 	if q_type=="others":
		pass
	elif q_type=="number":
		if pattern_number.match(answer.decode("utf-8")) :
			# print "number"
			type_array[0]=1
		

	elif q_type=="time":

		if pattern_time.match(answer.decode("utf-8")):
			# print "time"
			type_array[1]=1
	else:
		if ner_dict.has_key(answer):
			
			ner_info= ner_dict[answer]
			# print ner_info,
			if q_type=="organization":
				if "Nh" in ner_info:
					# print "organization"
					type_array[2]=1
			elif q_type=="person":
				if "Ns" in ner_info:
					# print "person"
					type_array[3]=1
			elif q_type=="place":
				if "Ni" in ner_info:
					# print "place"
					type_array[4]=1
		

	return pd.Series(type_array,index=types[:-1])
def jaccard(s1, s2):
    " takes two lists as input and returns Jaccard coefficient"
    st1=set(s1)
    st2=set(s2)
    u = set(st1).union(st2)
    i = set(st1).intersection(st2)
    return len(i)/len(u)

def score_jaccard(row):
	

	question=cut(row["question"])
	
	answer=cut(row["answer"])
	
	return jaccard(question,answer)


def score_word_overlaps_new(row):
	question=cut(row["question"])
	
	answer=cut(row["answer"])

	overlap= set(answer).intersection(set(question)-stopwords) 
	return len(overlap)
	global num,length
	num += 1
	question = cut(str1)
	answer = cut(str2)
	for w in stopwords:
		if w in question:
			question.remove(w)
		if w in answer:
			answer.remove(w)

	length += len(question)

	overlap= set(answer).intersection(set(question))
	
	weight_position = dict({})
	for k in range(len(question) ):
	 	if question[k] in overlap:
	 		weight_position[question[k] ] = ((k+1)/len(question)*1.0 )**3.2
	 		#weight_position[question[k] ] = math.atan(2*k - 8*len(question)/7)*3.14159/4 + 2

	di_question = []
	di_answer = []
	for w in question:
		b = w.decode('utf-8')
		for i in range(len(b) ):
			di_question.append(b[i])
	for w in answer:
		b = w.decode('utf-8')
		for i in range(len(b) ):
			di_answer.append(b[i])

	di_overlap = set(di_question).intersection(set(di_answer) )

	di_weight_p = dict({})
	for k in range(len(di_question) ):
	 	if di_question[k] in di_overlap:
	 		di_weight_p[di_question[k] ] = ((k+1)/len(di_question) *1.0 )**3.2
	 		#di_weight_p[di_question[k] ] = math.atan(2*k - 8*len(di_question)/7)*3.14159/4 + 2

	di_weight_all = 0.0
	for k in di_overlap:
		di_weight_all += di_weight_p[k]

	weight_all = 0.0
	for k in overlap:
		weight_all += weight_position[k]

	return (di_weight_all+weight_all)

def ma_overlap(row):
	question = cut(row["question"])
	answer = cut(row["answer"])

	overlap= set(answer).intersection(set(question))
	
	weight_position = dict({})
	for k in range(len(question) ):
	 	if question[k] in overlap:
	 		weight_position[question[k] ] = ((k+1)/(len(question)+1))**3.2

	weight_all = 0.0
	for k in overlap:
		weight_all += weight_position[k]
	return weight_all 


def ma_overlap_zi(row):
	question = cut(row["question"])
	answer = cut(row["answer"])
	
	di_question = []
	di_answer = []
	for w in question:
		

		for i in range(len(w) ):
			di_question.append(w[i])
	for w in answer:
		
		for i in range(len(w) ):
			di_answer.append(w[i])

	di_overlap = set(di_question).intersection(set(di_answer) )

	di_weight_p = dict({})
	for k in range(len(di_question) ):
	 	if di_question[k] in di_overlap:
	 		di_weight_p[di_question[k] ] = ((k+1)/len(di_question))**3.2


	di_weight_all = 0.0
	for k in di_overlap:
		di_weight_all += di_weight_p[k]




	return di_weight_all /(len(di_answer)+40)

def word_overlap(row):
	question=cut(row["question"]) 
	answer=cut(row["answer"])

	overlap= set(answer).intersection(set(question)-stopwords) 
	return len(overlap)

def answerLen(row):
	return len(row["answer"])


def dataPrepare(orginal):
	df=orginal.copy()
	#df[types[:-1]]=df.apply(typeinfo_array,axis=1)
	df["type_score"]=df.apply(typeinfo,axis=1)
	
	if debug:
		df["question"]=df["question"].apply(cut_sentence)
		df["answer"]=df["answer"].apply(cut_sentence)
	
	return df
def getFeatureSofQA(df):
	
	df=dataPrepare(df)

	methods={"ma":ma_overlap,"ma_overlap_zi":ma_overlap_zi,"w2vSimilarity":w2vSimilarity,"answerLen":answerLen,"word_overlap":word_overlap} #"word_overlap":word_overlap,"jacard":score_jaccard,
	names=[]
	for name,method in methods.items():
		df[name]=df.apply(method,axis=1)
		names.append(name)

	#names.extend(types[:-1])
	names.append("type_score")

	return df,names


if __name__=="__main__":
	test_file="data/nlpcc-iccpol-2016.dbqa.testing-data"
	train_file="data/nlpcc-iccpol-2016.dbqa.training-data"
	df=pd.read_csv(train_file,header=None,sep="\t",names=["question","answer","flag"],quoting =3)
	features=getFeatureSofQA(df)
	print features