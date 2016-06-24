#coding=utf8
import pandas as pd 
import os
import subprocess
import collections
import numpy as np 
from nltk.metrics import edit_distance
from gensim import models
import jieba
import pynlpir


from gensim.models.doc2vec import TaggedLineDocument,Doc2Vec
# model= Doc2Vec.load("Doc2vec.model")
# model_vocab=model.vocab.keySet()

stopwords=set(u", ? 、 。 “ ” 《 》 ！ ， ： ； ？ ".split())
print "#".join(stopwords)
pynlpir.open()
#pynlpir.nlpir.ImportUserDict("user.dict")
release=False
qa_path="nlpcc-iccpol-2016.dbqa.training-data"
w2v= models.word2vec.Word2Vec.load_word2vec_format('embedding/dbqa.word2vec', binary=False)
va=w2v.vocab
initializer=np.array([0.000001]*300)
qe_dict={}


def evaluation(modelfile,resultfile="result.text",input=qa_path):
	cmd="test.exe " + " ".join([input,modelfile,resultfile])
	print modelfile[19:-6]+":" # 
	subprocess.call(cmd, shell=True)


def jaccard(s1, s2):
    " takes two lists as input and returns Jaccard coefficient"
    st1=set(s1)
    st2=set(s2)
    u = set(st1).union(st2)
    i = set(st1).intersection(st2)
    return len(i)/len(u)

def baseline():
	path="model"
	for root, dirs, files in os.walk(path):
		for file in  files:
			model=os.path.join(root,file)

			evaluation(model)

def cut(sentence):
	try:
		words= pynlpir.segment(sentence, pos_tagging=False)

	except:
		words= jieba.cut(sentence)

	words=[word for word in words if word not in stopwords]

	return words
	


def score_edit_distance(row):
	
	return edit_distance(row["question"],row["answer"])

def score_word_overlaps(str1,str2):
	question=cut(str1)
	answer=cut(str2)
	overlap= set(answer).intersection(set(question))
	return len(overlap)

def queryExpansion(sentence):
	if qe_dict.has_key(sentence):
		return sentence
	else:

		print sentence
		qe_dict[sentence]=pynlpir.get_key_words(sentence, weighted=True)
		for word,weight in qe_dict[sentence]:
			print "%s : %s" %(word,weight) ,
		print 
		return sentence

def score_jaccard(row):
	
	question=cut(row["question"])
	answer=cut(row["answer"])
	return jaccard(question,answer)

def word_overlap(row):
	
	
	question=row["question"]
	
	answer=row["answer"]
	return score_word_overlaps(question,answer)

	 #[item for item in question if item in set(answer)]


def w2vSimilarity(row):
	question=cut(row["question"])
	
	answer=cut(row["answer"])

	sentence1= reduce(lambda x,y:x+y, [ w2v[word] for word in question if word in va],initializer)
	sentence2= reduce(lambda x,y:x+y, [ w2v[word] for word in answer if word in va],initializer) 

	return np.dot(sentence1,sentence2)/(np.linalg.norm(sentence1)*np.linalg.norm(sentence2))

def doc2vSimilarity(row):
	try:
		question=pynlpir.segment(row["question"], pos_tagging=False)
		answer=pynlpir.segment(row["answer"], pos_tagging=False)
	except:
		print row["question"]
		print row["answer"]
		question=jieba.cut(row["question"])
		answer=jieba.cut(row["answer"])
	sentence1=set (question)  & model_vocab
	sentence2=set (answer)  & model_vocab

	return model.n_similarity(sentence1,sentence2)
	

def work():
	print "start"

	df=pd.read_csv(qa_path,header=None,sep="\t",names=["question","answer","flag"],quoting =3)#encoding ="utf-8",

	# if release:
	# 	df["edit_distance"]=df.apply(score_edit_distance,axis=1)
	# 	df["edit_distance"]=df["edit_distance"]*1.0/df["edit_distance"].max()


	methods={"word2vec":w2vSimilarity,"wordOverlap":word_overlap,"score_jaccard":score_jaccard}  #"doc2vec":doc2vSimilarity,

	for name,method in methods.items():
		df[name]=df.apply(method,axis=1)
		df[name].to_csv("model/train.QApair."+name+".score",index=False,sep="\t")

	df.to_csv("temp.csv",index=False,sep="\t")





if __name__=="__main__":
	#haddle()
	#work()
	work()
	baseline()
	