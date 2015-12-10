#! /usr/bin/python
import sys,getopt,subprocess,re,math,commands,time,copy,random
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model , cross_validation, metrics, tree, svm
from sklearn.linear_model import SGDRegressor
from pylab import savefig
from sklearn.cross_validation import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
import pylab as Py

GlobalNumModels=11
ChoseModel=8
GlobalNumFolds=10

def usage():
	print "\n\t Usage: DistCalc.py -x <Training file name> \n\t Optional: \n\t\t -o <Output file name> \n\t\t -y <Test-file-name> "
	#\n\t\t -c <correlation-flag> 0: No correlaiton 1: IpStats correlation"
	sys.exit()
	
def RemoveWhiteSpace(Input):
	temp=re.sub('^\s*','',Input)
	Output=re.sub('\s*$','',temp)
	
	return Output

def shuffle(df):
	#credits: Jerome Zhao : http://stackoverflow.com/questions/15772009/shuffling-permutation-a-dataframe-in-pandas
    index = list(df.index)
    random.shuffle(index)
    df = df.ix[index]
    df.reset_index()
    return df

def IsNumber(s):
# Credits: StackExchange: DanielGoldberg: http://stackoverflow.com/questions/354038/how-do-i-check-if-a-string-is-a-number-in-python
	try:
		float(s)
		return True
	except ValueError:
		return False

def RestofCols(ListOfCols,Column):
	return filter(lambda x: x!=Column, ListOfCols)

def CheckTolerance(Num,Tolerance,List):
	return next( (x for x in List if( ( x > ( Num*(1-Tolerance) ) ) & ( x < ( Num*(1+Tolerance) ) ) ) ) , False)

def OldBinParam(Data,Param,Bins):
	BinnedData={}
	for CurrBin in Bins:
		BinnedData[CurrBin[0]]=Data[ ( Data[Param] > CurrBin[0] ) & ( Data[Param] <= CurrBin[1] ) ]
		print "\t Bin: "+str(CurrBin)+" length "+str(BinnedData[CurrBin[0]].shape)
	return BinnedData

def FindInDF(Prop1,Prop2,Prop1Val,Prop2Val,DF):
	for idx,CurrRowData in DF.iterrows():
		if( ( CurrRowData[Prop1]==Prop1Val ) and ( CurrRowData[Prop2]==Prop2Val ) ):
			#print "\t CurrRowData[Prop1] "+str(Prop1Val )+" CurrRowData[Prop2]: "+str(Prop2Val)
			#print "\t Kernel-Name: "+str(CurrRowData["KERNELNAME0"])
			return CurrRowData
			#sys.exit()
	print "\t Couldn't find it!! :-| "
	sys.exit()

def Norm(IpData,TestFlag=True):
	Data=IpData.copy(deep=True)
	Params=Data.columns
	if TestFlag:
		TabooParams=['KERNEL','Fre','FRE','LoopID']
	else:
		TabooParams=['File','Loop']	
	for CurrParam in Params:
		TabooFound=False
		for CurrTabooParam in TabooParams:
			RegExStr='.*'+str(CurrTabooParam)+'.*'
			CheckTaboo=re.match(RegExStr,CurrParam)
			if(CheckTaboo):
				TabooFound=True
		if(not(TabooFound)):
			Mean=Data[CurrParam].mean()
			Std=Data[CurrParam].std()
			Data[CurrParam]=Data[CurrParam].apply(lambda d: (d-Mean)/(Std) )
			print "\t Processed-parameter: "+str(CurrParam)
		else:
			print "\t Did not process-parameter: "+str(CurrParam)
	return Data

def AllPositiveNorm(IpData,TestFlag=True):
	Data=IpData.copy(deep=True)
	Params=Data.columns
	if TestFlag:
		TabooParams=['KERNEL','FRE','LoopID'] # 'Fre',
	else:
		TabooParams=['File','Loop']#,'Iter']	
	for CurrParam in Params:
		TabooFound=False
		for CurrTabooParam in TabooParams:
			RegExStr='.*'+str(CurrTabooParam)+'.*'
			CheckTaboo=re.match(RegExStr,CurrParam)
			if(CheckTaboo):
				TabooFound=True
		if(not(TabooFound)):
			Min=Data[CurrParam].min()
			Data[CurrParam]=Data[CurrParam].apply(lambda d: d + abs(Min) ) 
			"""  # Comment Data[CurrParam] in above lines if following block will be uncommented. 
			Alternatively, if the data passed is not already normalized. 
			Mean=Data[CurrParam].mean()
			Std=Data[CurrParam].std()			
			Data[CurrParam]=Data[CurrParam].apply(lambda d: (d-Mean+abs(Min))/(Std) ) """
			print "\t Processed-parameter: "+str(CurrParam)
		else:
			print "\t Did not process-parameter: "+str(CurrParam)
	return Data
 		
def DistCalc(Data,DistDF,Params,DistParam,TestFlag=True,AbsFlag=True):
	LenDF=len(Data.index)
	Dist=[ 0 for Idx in range(LenDF) ]
	TabooParams=['Iterations','POWER','RAP','Runtime','vec']
	print "\t DistParam: "+str(DistParam)
	if TestFlag:
		TabooParams.append('KERNEL')
		#TabooParams.append('Fre')
		TabooParams.append('FRE')
		TabooParams.append('LoopID')
	else:
		TabooParams.append('File')
		TabooParams.append('Loop')
		#TabooParams.append('Fre')
		#TabooParams.append('FRE')		
		
	if(DistParam==''):
		DistParam=''
		for CurrParam in range(len(Params)-1):
			DistParam+=str(CurrParam)+'_'
		DistParam+=str(Params[len(Params)-1])
	for CurrParam in Params:
		TabooFound=False
		for CurrTabooParam in TabooParams:
			RegExStr='.*'+str(CurrTabooParam)+'.*'
			CheckTaboo=re.match(RegExStr,CurrParam)
			if(CheckTaboo):
				TabooFound=True
		if(not(TabooFound)):
			print "\t Processing parameter: "+str(CurrParam)
			if(AbsFlag):
				DistIdx=0
				for Idx,CurrRow in Data.iterrows():
					Dist[DistIdx]+=abs(CurrRow[CurrParam]) # Abs-all 
					DistIdx+=1
			else:
				DistIdx=0
				for Idx,CurrRow in Data.iterrows():
					Dist[DistIdx]+=(CurrRow[CurrParam]) # Assuming I/P dataframe is N(0,1), with the min- having been moved to 0. # Alternatively, can use Abs-all param.
					DistIdx+=1
		else:
			print "\t Not processing paramter: "+str(CurrParam)
			
	#print "\t Idx: "+str(Idx)+"\t Tmp "+str(Tmp)
	for Idx in range(LenDF):
		Dist[Idx]=round(Dist[Idx],4)		
	#Meh={DistParam:Dist}
	#Meh1=pd.DataFrame(Meh)
	DistDF[DistParam]=Dist
	print "\t DistParam: "+str(DistParam)
	return DistDF	

def RestofCols(ListOfCols,Column):
	return filter(lambda x: x!=Column, ListOfCols)

def BinwiseCorr(BinnedData):
	CorrCollection={}
	for CurrBin in BinnedData:
		CorrCollection[CurrBin]=[]
		AllParams=BinnedData[CurrBin].columns
		RemainingParams=RestofCols(AllParams,'Index')
		RemainingParams=RestofCols(RemainingParams,'Slowdown')
		RemainingParams=RestofCols(RemainingParams,'Frequency1')
		RemainingParams=RestofCols(RemainingParams,'RelativeEXECTIME')
		break
	print "\t MyCols: "+str(RemainingParams)
	for CurrBin in BinnedData:
		CorrCollection[CurrBin]=[]
		#AllParams=BinnedData[CurrBin].columns
		#RemainingParams=RestofCols(AllParams,'Index')
		#RemainingParams=RestofCols(RemainingParams,'Slowdown')
		print "\n\n\t ------ CurrBin: "+str(CurrBin)+" --------- "
		for CurrParam in RemainingParams:
			Corr=BinnedData[CurrBin][CurrParam].corr(BinnedData[CurrBin]['Slowdown'])
			print "\t CurrParam: "+str(CurrParam)+"\t corr-- "+str(Corr)
			ParamStr= str(CurrParam)+'_Slowdown'
			CorrCollection[CurrBin].append( (ParamStr,Corr ) )

		for PrevParam in RemainingParams:
			for CurrParam in RemainingParams:
				ReExprStr='.*'+str(PrevParam)+'.*'
				CheckSimilarParams=re.match(ReExprStr,CurrParam)
				if( (CheckSimilarParams) and (PrevParam!=CurrParam) ):
					Corr= BinnedData[CurrBin][CurrParam].corr(BinnedData[CurrBin][PrevParam])
					ParamStr=str(CurrParam)+'_'+str(PrevParam)
					CorrCollection[CurrBin].append( ( ParamStr, Corr ) )
					print "\t PrevParam: "+str(PrevParam)+" is similar to CurrParam "+str(CurrParam)+" corr: "+str(Corr)		
		sys.exit()			
	sys.exit()			

def BinParam(Data,Param,BinLength,Store=False,StoreName=''):
	BinMin=Data[Param].min()
	BinMax=Data[Param].max()
	BinPeriod= round( (BinMax-BinMin)/BinLength , 4)
	
	Bins=[ [round(BinMin+(i*BinPeriod),4),round(BinMin+((i+1)*BinPeriod),4)] for i in range(BinLength) ]
	BinnedData={}
	for Idx,CurrBin in enumerate(Bins):
		BinnedData[Idx]=Data[ (Data[Param]>CurrBin[0]) & ( Data[Param]<=CurrBin[1]) ]
		print "\t Bin-range "+str(CurrBin)+" bin: "+str(Idx)+" length "+str(BinnedData[Idx].shape)
	if Store:
		StoreFileName='Store_'+str(StoreName)
		#WriteStore=open(StoreFileName,'w')
		for CurrBin in BinnedData:
			#print "\t CurrBin-identifier: "+str(CurrBin)
			BinnedData[CurrBin].to_csv(StoreFileName,sep='\t',mode='a')			
	return BinnedData

def ImpDistCalc(IpStats,InputFileName,IpTestFlag,IpAbsFlag):
	AllParams=["Frequency1","l3mpi","idu2","pbranchops","fprat","pintops","l3hr","pfpops","l1hr","l2hr","pstores","fdu","ploads","l1mpi","fdu2","l2mpi","idu","mdu2","pmemops","bytespermop"]
	AllParams1=["l3mpi","idu2","pbranchops","fprat","pintops","l3hr","pfpops","l1hr","l2hr","pstores","fdu","ploads","l1mpi","fdu2","l2mpi","idu","mdu2","pmemops","bytespermop"]
	AllParams2=["idu2","pbranchops","fprat","pintops","l3hr","pfpops","l1hr","l2hr","pstores","fdu","ploads","l1mpi","fdu2","l2mpi","idu","mdu2","pmemops","bytespermop"]
	AllParams3=["pbranchops","fprat","pintops","l3hr","pfpops","l1hr","l2hr","pstores","fdu","ploads","l1mpi","fdu2","l2mpi","idu","mdu2","pmemops","bytespermop"]
	
	MostXParams=["pintops","pbranchops","l1hr","l2hr","l3hr","l1mpi","l2mpi","l3mpi","fdu2","mdu2"]

 	ImpParams=["Frequency1","l3mpi","idu2","pbranchops","fprat","pintops"]	
 	ReallyImpParams=["Frequency1"]
 	ReallyImpParams1=["Frequency1","l3mpi"]	
 	ReallyImpParams2=["Frequency1","l3mpi","pbranchops"]	
 	ReallyImpParams3=["Frequency1","l3mpi","pbranchops","idu2"]
 	ReallyImpParams4=["Frequency1","l3mpi","pbranchops","idu2","fprat"]
 	
	PercentParams=["pintops","pmemops","ploads","pstores","pfpops","pbranchops"]
	CacheParams=["l1hr","l2hr","l3hr","l1mpi","l2mpi","l3mpi"]
	DefUseParams=["idu","fdu","idu2","fdu2","mdu2","fprat","bytespermop"] 	
 		
 	ReallyImpParams2=["pintops","pstores","l3mpi"]	
	SpatParams=["spat0","spat2","spat4","spat8","spat16","spat32","spat64","spat128","spatOther"]
	SpatParams1=[] #"spat0","spat8","spat16","spat32","spat64","spat128","spatOther"]
	SpatParams2=["spat0","spat8","spatOther"]
	
	DistIpData=pd.DataFrame(index=[],columns=[])
	
	for Idx,CurrRow in IpStats.iterrows():
		#print "\t CurrItem: "+str(CurrItem)
		CurrRow['RelativeEXECTIME']=round(CurrRow['RelativeEXECTIME'],4)
	#DistIpData['Slowdown']=Meh
	Meh=IpStats['RelativeEXECTIME']
	#print "\t Meh.shape: "+str(len(Meh))#.shape)
	#print "\t Meh: "+str(Meh)
	
	print "\n\n\t ------- Method-3: AllParams being used!! --------- "	
	DistIpData=DistCalc(IpStats,DistIpData,AllParams,'l1All',TestFlag=IpTestFlag,AbsFlag=IpAbsFlag)
	DistIpData=DistCalc(IpStats,DistIpData,AllParams1,'l1Allp1',TestFlag=IpTestFlag,AbsFlag=IpAbsFlag)
	DistIpData=DistCalc(IpStats,DistIpData,AllParams2,'l1Allp2',TestFlag=IpTestFlag,AbsFlag=IpAbsFlag)
	DistIpData=DistCalc(IpStats,DistIpData,AllParams3,'l1Allp3',TestFlag=IpTestFlag,AbsFlag=IpAbsFlag)
	print "\t Cols: "+str(DistIpData.columns)+" shape "+str(DistIpData.shape) 
	
	print "\n\n\t ------- Method-1: ImpParams being used!! --------- "	
	DistIpData=DistCalc(IpStats,DistIpData,ReallyImpParams,'ReallyImpParams',TestFlag=IpTestFlag,AbsFlag=IpAbsFlag)
	DistIpData=DistCalc(IpStats,DistIpData,ReallyImpParams1,'ReallyImpParams1',TestFlag=IpTestFlag,AbsFlag=IpAbsFlag)
	DistIpData=DistCalc(IpStats,DistIpData,ReallyImpParams2,'ReallyImpParams2',TestFlag=IpTestFlag,AbsFlag=IpAbsFlag)
	DistIpData=DistCalc(IpStats,DistIpData,ReallyImpParams3,'ReallyImpParams3',TestFlag=IpTestFlag,AbsFlag=IpAbsFlag)
	DistIpData=DistCalc(IpStats,DistIpData,ReallyImpParams4,'ReallyImpParams4',TestFlag=IpTestFlag,AbsFlag=IpAbsFlag)
	print "\t Cols: "+str(DistIpData.columns)+" shape "+str(DistIpData.shape)
	
	print "\n\n\t ------- Method-2: Class-specific params being used!! --------- "	
	DistIpData=DistCalc(IpStats,DistIpData,PercentParams,'PercentParams',TestFlag=IpTestFlag,AbsFlag=IpAbsFlag)
	DistIpData=DistCalc(IpStats,DistIpData,CacheParams,'CacheParams',TestFlag=IpTestFlag,AbsFlag=IpAbsFlag)
	DistIpData=DistCalc(IpStats,DistIpData,DefUseParams,'DefUseParams',TestFlag=IpTestFlag,AbsFlag=IpAbsFlag)
	
	OpFileName=''
	if(IpAbsFlag):
		OpFileName='DistAbsL1_'+str(InputFileName)
		DistIpData.to_csv(OpFileName,sep='\t')
	else:
		OpFileName='DistPosL1_'+str(InputFileName)
		DistIpData.to_csv(OpFileName,sep='\t')
	print "\n\n\t OpFileName: "+str(OpFileName)
	print "\t ------- Method-4: Non-significant params pending!! --------- "	

def main(argv):
	InputFileName=''
	IpAbsFlag=''
	IpTestFlag=''
	OutputFileName=''
	TestFileName=''
	CorrelationFlag=''
	verbose=False 
	try:
	   opts, args = getopt.getopt(sys.argv[1:],"x:a:t:o:y:c:h:v",["training=","absflag=","testflag=","output=","test=","correlation","help","verbose"])
	except getopt.GetoptError:
		#print str(err) # will print something like "option -a not recognized"
	   usage()
	   sys.exit(2)
	  
	for opt, arg in opts:
		print "\n\t Opt: "+str(opt)+" argument "+str(arg)	
		if opt == '-h':
			usage()		
		elif opt in ("-x", "--training"):
			InputFileName=RemoveWhiteSpace(arg)
			print "\n\t Input file is "+str(InputFileName)+"\n";
		elif opt in ("-a", "--absflag"):
			IpAbsFlag=int(RemoveWhiteSpace(arg))
			if(IpAbsFlag>0):
				IpAbsFlag=True
			else:
				IpAbsFlag=False
			print "\n\t AbsFlag is "+str(IpAbsFlag)+"\n";			
		elif opt in ("-t", "--testflag"):
			IpTestFlag=int(RemoveWhiteSpace(arg))
			if(IpTestFlag>0):
				IpTestFlag=True
			else:
				IpTestFlag=False
			print "\n\t TestFlag is "+str(IpTestFlag)+"\n";						
		elif opt in ("-y", "--test"):
			TestFileName=RemoveWhiteSpace(arg)
			print "\n\t Test file is "+str(TestFileName)+"\n";			
		elif opt in ("-c", "--correlation"):
			CorrelationFlag=RemoveWhiteSpace(arg)
			print "\n\t Correlation flag is "+str(CorrelationFlag)+"\n";			
		elif opt in ("-o", "--output"):
			OutputFileName=RemoveWhiteSpace(arg)
			print "\n\t Source file is "+str(OutputFileName)+"\n";			
		else:
   			usage()

	if(len(opts)==0):
		usage()

	if(InputFileName==''):
		usage()
	if(OutputFileName==''):
		OutputFileName='DefaultOutputFile.log'
		print "\n\t INFO: Using default output file name: "+str(OutputFileName)
	if(CorrelationFlag==''):
		CorrelationFlag=0
		print "\t\t INFO: Using default correlation flag: "+str(CorrelationFlag)	
	if(IpAbsFlag==''):
		IpAbsFlag=True
		print "\t INFO: Using default AbsFlag-- "+str(IpAbsFlag)
	if(IpTestFlag==''):
		IpTestFlag=False
		print "\t INFO: Using default TestFlag-- "+str(IpTestFlag)
		
	IpStats=pd.read_csv(InputFileName,sep='\t',header=0)
	#IpStats=shuffle(IpStats);IpStats=shuffle(IpStats);IpStats=shuffle(IpStats)
	print "\t IpStats.shape: "+str(IpStats.shape)
	
	YParams=["RelativeEXECTIME"] #["RelativeACPOWER"]
	#YParams=["ACPOWER"] =["pintops","pstores","pfpops"]
	XYParams=[]
	
	PercentParams=["pintops","pmemops","ploads","pstores","pfpops","pbranchops"]
	CacheParams=["l1hr","l2hr","l3hr","l1mpi","l2mpi","l3mpi"]
	DefUseParams=["idu","fdu","idu2","fdu2","mdu2","fprat","bytespermop"]
	
	AllParams=["pintops","pmemops","ploads","pstores","pfpops","pbranchops","l1hr","l2hr","l3hr","l1mpi","l2mpi","l3mpi","idu","fdu","idu2","fdu2","mdu2","fprat","bytespermop"]	

	IpStats=IpStats[IpStats.Frequency1!=2.9]
	#NormData=Norm(IpStats,IpTestFlag); NormData.to_csv('Norm'+str(InputFileName),sep='\t') ; sys.exit();	#IpStats=NormData
	NormData=AllPositiveNorm(IpStats,IpTestFlag);NormData.to_csv('Pos'+str(InputFileName),sep='\t') ; sys.exit();
	
	#ImpDistCalc(IpStats,InputFileName,IpTestFlag,IpAbsFlag); sys.exit()
	BinIpStats10=BinParam(IpStats,'Slowdown',10,Store=False) #,StoreName=''):
	print "\n\t Now checking BinIpStats10[0]--- "
	StoreName='Bin0_MostPopulus_'+str(InputFileName)
	Blah=BinParam(BinIpStats10[0],'l1All',5,True,StoreName)
 		
	print "\n\n" 	
	  	
  	
if __name__ == "__main__":
   main(sys.argv[1:])
	
	
