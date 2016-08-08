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

word_weight=[0.06743412225621209, 0.06743412225621209, 0.06743412225621209, 0.06743412225621209, 0.06747149564050972, 0.06750421901368836, 0.06749267252864376, 0.06749278470622501, 0.06747799067840497, 0.06760920790284575, 0.06769137578556826, 0.06828107365425193, 0.06809951188789166, 0.0683210998485378, 0.06884291460189773, 0.06877787015342755, 0.06892681309150277, 0.06935108856478546, 0.06903755526405286, 0.0689801417024249, 0.07089276930644368, 0.07148767583561094, 0.07192030262446841, 0.07169163390235406, 0.07225273341059102, 0.07553697563788381, 0.07764088168018299, 0.07744863355276281, 0.07793181935750866, 0.07920327275165985, 0.07973251028806584, 0.07997977755308393, 0.08016692856590844, 0.084966527112028, 0.09024801614731662, 0.0904071773636991, 0.09068660051608098, 0.09139500570696807, 0.09160894852426575, 0.09342254416765738, 0.09932156751114907, 0.10477924442421484, 0.10469626447070056, 0.10712205454064072, 0.11183290608249825, 0.11445032700093428, 0.11537357696355113, 0.1164609319459704, 0.1164972634870993, 0.11648681250611287, 0.1317436444789443, 0.1559830866807611, 0.15601098668920346, 0.15666246321984026, 0.15613305613305614, 0.15669410656509808, 0.16064447340362517, 0.17207100591715976, 0.18479509766373037, 0.18577113938977055, 0.1994342906875544, 0.21631451333564253, 0.21523141337557994, 0.21787881905153791, 0.23585076246157555, 0.2402605696768425, 0.23997951082084773, 0.2561741314357472, 0.28258513931888546, 0.2820551567812618, 0.2855891371003066, 0.30209069077553324, 0.3162459883771359, 0.31576770350761085, 0.32279489904357067, 0.3281238426783201, 0.3474845893815868, 0.3465230166503428, 0.35126081697217826, 0.3599875866349436, 0.3661527680448493, 0.36773983032412444, 0.36747178567661637, 0.3727151561309977, 0.3746658919233004, 0.3749285795909039, 0.377666999002991, 0.3775884510708792, 0.3757436593257489, 0.37907488986784144, 0.37588018330166534, 0.37307558745225144, 0.3730914935139479, 0.37460054444312935, 0.37678207739307534, 0.3767643865363735, 0.3778154516832163, 0.3781227261702644, 0.3781227261702644, 0.3781227261702644, 0.3781227261702644,0.3781227261702644]
zi_weight  =[0.06311305070656692, 0.06311305070656692, 0.06319525153577951, 0.06322935252978937, 0.06325283094498446, 0.06282205261322692, 0.06247913653054412, 0.06295595288410298, 0.06293300509934939, 0.06351564676385922, 0.06278948222106079, 0.06271108922037714, 0.0627009959654628, 0.06303167650391517, 0.06286119666401356, 0.06277040669166427, 0.06292851549817613, 0.06328375133404482, 0.06365102265976406, 0.06347040979339807, 0.06366393065235683, 0.06379672478270522, 0.06417016275159568, 0.06475865014950875, 0.06480753437822159, 0.06538604720422903, 0.06590763309813984, 0.06713679440952168, 0.06761667260420377, 0.06846182564868397, 0.06913552124180349, 0.06990179087232813, 0.07029111719333167, 0.07311283880779454, 0.0752825022175942, 0.07507766345238737, 0.07610065982284657, 0.07693470374848851, 0.07914720095040859, 0.07998036365009535, 0.08305282565672627, 0.08560462896218513, 0.08592441564115634, 0.0900318565713439, 0.09484885582446557, 0.09652035661816499, 0.09867783906690808, 0.09907701182578599, 0.1017010907931592, 0.10224740410601678, 0.10821088446720149, 0.1183561792273934, 0.11820220032327024, 0.11772694781987134, 0.12200116287888364, 0.12560969477548148, 0.13100934718468749, 0.1356386543358799, 0.13897011724733338, 0.14031429434874584, 0.14755518301201453, 0.15148924978631073, 0.1550028918449971, 0.16164279134771145, 0.16721584183755078, 0.17357905366416618, 0.17962024298012713, 0.186878413790926, 0.19268799335272124, 0.19708087097925672, 0.2057518488085456, 0.2094852047020673, 0.2132127890003296, 0.21645673014917674, 0.21985431636972144, 0.22485383860367858, 0.22947885235002302, 0.23223253690458073, 0.23672688843037593, 0.2366802749211927, 0.24115120699017623, 0.24028190727893403, 0.24168271984172437, 0.24329907555647368, 0.2460151632738344, 0.24704108977221975, 0.250964612501378, 0.2486659551760939, 0.2505322924059617, 0.25259633391975483, 0.24995722840034218, 0.2493079686671771, 0.25403544418105883, 0.253505160789929, 0.2532853285328533, 0.2542948717948718, 0.25465880144791525, 0.2565368567454798, 0.2575371549893843, 0.2574264445232187, 0.2574264445232187,0.2574264445232187]

def ma_overlap(row):
	question = cut(row["question"])
	answer = cut(row["answer"])

	overlap= set(answer).intersection(set(question))
	
	weight_position = dict({})
	for k in range(len(question) ):
	 	if question[k] in overlap:
	 		# print int(100*((k+1)/(len(question)+1)) )
	 		weight_position[question[k] ] =((k+1)/(len(question)+1))**3.2# word_weight[ int(100*((k+1)/(len(question)+1)) )] #

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
	 		# print int(100*((k+1)/(len(question)+1)) )
	 		di_weight_p[di_question[k] ] =((k+1)/len(di_question))**3.2# zi_weight[ int(100*((k+1)/(len(di_question)+1)) )]#((k+1)/len(di_question))**3.2


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

	methods={"ma":ma_overlap,"ma_overlap_zi":ma_overlap_zi,"w2vSimilarity":w2vSimilarity,"answerLen":answerLen} #"word_overlap":word_overlap,"jacard":score_jaccard,
	names=[]
	for name,method in methods.items():
		df[name]=df.apply(method,axis=1)
		df[name]=(df[name] -df[name].mean())/df[name].std(ddof=0)
		# df["embedding_sigmod"]=    1.0 / (1.0 + np.exp( -df.embedding_mean ))

		print "%s feature process over" % name
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