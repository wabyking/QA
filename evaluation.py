import pandas as pd 
import subprocess
import platform

qa_path="data/nlpcc-iccpol-2016.dbqa.testing-data"

def mrr_metric(group):
	candidates=group.sort_values(by='score',ascending=False).reset_index()
	rr=candidates[candidates["flag"]==1].index.min()+1
	return 1.0/rr
def map_metric(group):
	ap=0
	candidates=group.sort_values(by='score',ascending=False).reset_index()
	correct_candidates=candidates[candidates["flag"]==1]
	if len(correct_candidates)==0:
		return 1
	for i,index in enumerate(correct_candidates.index):
		ap+=1.0* (i+1) /(index+1)
	#print ap/len(correct_candidates)
	return ap/len(correct_candidates)

def evaluation_plus(modelfile, groundtruth=qa_path):
	answers=pd.read_csv(groundtruth,header=None,sep="\t",names=["question","answer","flag"],quoting =3)
	answers["score"]=pd.read_csv(modelfile,header=None,sep="\t",names=["score"],quoting =3)
	print answers.groupby("question").apply(mrr_metric).mean()
	print answers.groupby("question").apply(map_metric).mean()

def eval(predicted,groundtruth=qa_path):
	if  'Windows' in platform.system():
		modelfile=write2file(predicted)
		evaluationbyFile(modelfile)
		return 

	if type(groundtruth)!= str :
		answers=groundtruth
	else:
		answers=pd.read_csv(groundtruth,header=None,sep="\t",names=["question","answer","flag"],quoting =3)
	answers["score"]=predicted
	mrr= answers.groupby("question").apply(mrr_metric).mean()
	map= answers.groupby("question").apply(map_metric).mean()
	return map,mrr
def evaluate(predicted,groundtruth):
	filename=write2file(predicted)
	evaluationbyFile(filename,groundtruth=groundtruth)
def write2file(datas,filename="train.QApair.TJU_IR_QA.score"):
	with open(filename,"w") as f:
		for data in datas:
			f.write(str(data)+"\n")
	return filename


def evaluationbyFile(modelfile,resultfile="result.text",groundtruth=qa_path):
	cmd="test.exe " + " ".join([groundtruth,modelfile,resultfile])
	print modelfile[19:-6]+":" # 
	subprocess.call(cmd, shell=True)