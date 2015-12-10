def AllLevelsAvg(ParamList,IpSet):
	#IpSet=(L1Training,L2Training,L3Training,MemTraining)
	AvgCollection={}
	for CurrParam in ParamList:
		print "\t CurrParam: "+str(CurrParam)
		AvgCollection[CurrParam]={}
		for SetIdx,CurrSet in enumerate(IpSet):
			#print "\t SetIdx: "+str(SetIdx)+" Shape: "+str(CurrSet.shape)
			AvgCollection[CurrParam][SetIdx]=CurrSet[CurrParam].mean()
	
	for CurrParam in ParamList:
		AvgStr=''
		for CurrSet in range(len(AvgCollection[CurrParam])):
			if(CurrSet in AvgCollection[CurrParam]):
				AvgStr+='\t'+str(AvgCollection[CurrParam][CurrSet])
			else:
				print "\t Param: "+str(CurrParam)+" was not found for set "+str(CurrSet)
		print "\t CurrParam: "+str(CurrParam)+str(AvgStr)
		
def BinParam(Data,Param,Bins):
	BinnedData={}
	for CurrBin in Bins:
		BinnedData[CurrBin]=Data[Data[Param]==CurrBin]
		print "\t Bin: "+str(CurrBin)+" length "+str(BinnedData[CurrBin].shape)
	return BinnedData

def BinParam(Data,Param,BinLength):
	BinMin=Data[Param].min()
	BinMax=Data[Param].max()
	BinPeriod= round( (BinMax-BinMin)/BinLength , 4)
	Bins=[ [round(i*BinPeriod,4),round((i+1)*BinPeriod,4)] for i in range(BinLength) ]
	BinnedData={}
	for Idx,CurrBin in enumerate(Bins):
		BinnedData[Idx]=Data[ (Data[Param]>CurrBin[0]) & ( Data[Param]<=CurrBin[1]) ]
		print "\t Bin: "+str(CurrBin)+" length "+str(BinnedData[Idx].shape)
	return BinnedData
		
def ProcessRelativeEXECTIME(Data):	
	ProcessedRelativeEXECTIME=[]
	for idx,row in Data.iterrows():
		Temp= (row["RelativeEXECTIME"] - row["Frequency2"])
		ProcessedRelativeEXECTIME.append(Temp)
	Data['RelExeTime2']=ProcessedRelativeEXECTIME
	return Data
		
def ChoseLearnModel(UseModel=0,NumModels=GlobalNumModels):
	if(UseModel==0):
		LearnModel= linear_model.LinearRegression()
		print " \t Linear regression!! "
		Method='LinearRegression'
	elif(UseModel==1):
		alpha=0.5
		LearnModel = linear_model.RidgeCV(alphas=[0.1,0.15,0.2,0.25])#,0.01,0.7,0.75,0.80,0.90,0.93])
		print " \t Ridge-regression-- alpha: "+str(alpha)
		Method='RidgeRegression'
	elif(UseModel==2):
		alpha=0.01
		LearnModel= linear_model.LassoCV(cv=10) #(alphas=[0.0001,0.001,0.01,0.04,]) #Lasso(alpha=alpha) #alpha=alpha)
		print " \t Lasso-- alpha: "+str(alpha)
		Method='Lasso'
	elif(UseModel==3):
		alpha=0.001
		LearnModel= linear_model.ElasticNet(alpha=alpha) #alpha=alpha)
		print " \t ElasticNet-- alpha: "+str(alpha)
		Method='ElasticNet'
	elif(UseModel==4):
		alpha=0.4
		LearnModel= linear_model.LassoLars(alpha=alpha)#,normalize=False) #alpha=alpha)
		print " \t LassoLars-- alpha: "+str(alpha)
		Method='LassoLars'
	elif(UseModel==5):
		Method='DecisionTreeRegressor'
		LearnModel=tree.DecisionTreeRegressor(max_depth=3)
		print "\t Decision tree regressor!! "
	elif(UseModel==6):
		Method='SVMRegressor'
		LearnModel=svm.SVR() #kernel='rbf', C=1e3, gamma=0.05)
		print "\t SVM regressor!! "
	elif(UseModel==7):
		Method='SGDRegressor'
		LearnModel=SGDRegressor(loss="huber") #"epsilon_insensitive")
		print "\t SGD regressor!! "
	elif(UseModel==8):
		Method='RandomForestRegressor'
		LearnModel=RandomForestRegressor(n_estimators=10) # (loss="huber") #"epsilon_insensitive")
		print "\t Random Forest Regressor!! "
	elif(UseModel==9):
		Method='GradientBoostingRegressor'
		LearnModel=GradientBoostingRegressor(n_estimators=5) # (loss="huber") #"epsilon_insensitive")
		print "\t Gradient Boosting Regressor!! "
	elif(UseModel==9):
		Method='GradientBoostingRegressor'
		LearnModel=GradientBoostingRegressor(n_estimators=5) # (loss="huber") #"epsilon_insensitive")
		print "\t Gradient Boosting Regressor!! "
	elif(UseModel==10):	
		Method='AdaBoostRegressor'
		rng = np.random.RandomState(1)
		LearnModel=AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),n_estimators=100, random_state=rng) # (loss="huber") #"epsilon_insensitive")
		print "\t AdaBoost Regressor!! "
	return (LearnModel,Method)	

def Pred(Training,Test,XParams,YParams):
		TempTestSet=Test#[XYParams]		
		TestX=(TempTestSet[XParams]).as_matrix()
		TestY=(TempTestSet[YParams]).as_matrix()
		TrainX=(Training[XParams]).as_matrix()
		TrainY=(Training[YParams]).as_matrix()
		print "\t Finally the test-x and y is: "+str(TestX.shape)+" , "+str(TestY.shape)
		print "\t And the train-x and y is: "+str(TrainX.shape)+" , "+str(TrainY.shape)
		VarianceScoreCollection={}
		ManualRelativeErrorCollection={}
		for ModelIdx in range(GlobalNumModels):
			ModelIdx=ChoseModel
			print "\n"
			(LearnModel,Method)=ChoseLearnModel(UseModel=ModelIdx)
			LearnModel.fit(TrainX,TrainY.ravel())
			OutputY=LearnModel.predict(TestX)
			AbsError1=[]
			ManualMeanRelativeError=0.0
			for i in range(len(OutputY)):
				Temp=float(abs(OutputY[i]-TestY[i][0])/TestY[i][0])
				ManualMeanRelativeError+=Temp
				AbsError1.append(Temp)
			ManualMeanRelativeError/=len(OutputY)
			AbsError=( abs(OutputY- TestY[0])/TestY[0])
			MeanRelativeError=( np.mean(( abs(OutputY- TestY[0])/TestY[0]) ** 1) )
			VarianceScore=LearnModel.score(TestX,TestY)
			print "\t MeanRelativeError for the test set is: "+str(MeanRelativeError)+" Variance-score: "+str(VarianceScore)+" ManualMeanRelativeError "+str(ManualMeanRelativeError)
			TestOutputComparison=open('TestOutputComparison.dat','w')
			for i in range(len(OutputY)):
				#AbsError=float(abs(OutputY[i]-TestY[i][0])/TestY[i][0])
				TestOutputComparison.write("\n\t"+str(i)+"\t"+str(round(OutputY[i],6))+"\t"+str(round(TestY[i][0],6))+"\t"+str(round(AbsError1[i],6)))
				for CurrParam in TestX[i]:
					TestOutputComparison.write("\t"+str(CurrParam))
				#sys.exit()
			TestOutputComparison.close()	
			VarianceScoreCollection[Method]=VarianceScore
			ManualRelativeErrorCollection[Method]=ManualMeanRelativeError
			print "\t Feature importance: "+str(LearnModel.feature_importances_);
			return AbsError1

def Norm(Data):
	Params=Data.columns
	for CurrParam in Params:
		CheckKernelName=re.match('.*KERNEL.*',CurrParam)
		if(not(CheckKernelName)):
			Mean=Data[CurrParam].mean()
			Std=Data[CurrParam].std()
			Data[CurrParam]=Data[CurrParam].apply(lambda d: (d-Mean)/(Std) )
			print "\t Processed-parameter: "+str(CurrParam)
		else:
			print "\t Did not process-parameter: "+str(CurrParam)
	return Data

def DistCalc(Data,DistDF,Params,DistParam):
	Dist=[ 0 for Idx in range(len(Data.index)) ]
	for CurrParam in Params:
		CheckKernelName=re.match('.*KERNEL.*',CurrParam)
		if(not(CheckKernelName)):
			print "\t Processing parameter: "+str(CurrParam)
			for Idx,CurrRow in Data.iterrows():
				Dist[Idx]+=abs(CurrRow[CurrParam])
		else:
			print "\t Not processing paramter: "+str(CurrParam)
	#print "\t Idx: "+str(Idx)+"\t Tmp "+str(Tmp)
	Meh={DistParam:Dist}
	Meh=pd.DataFrame(Meh)
	DistDF=DistDF.append(Meh)
	return DistDF	

def AllPositiveNorm(IpData,TestFlag=True):
	Data=IpData.copy(deep=True)
	Params=Data.columns
	if TestFlag:
		TabooParams=['KERNEL','Fre','FRE','LoopID']
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
			Mean=Data[CurrParam].mean()
			Std=Data[CurrParam].std()
			#Data[CurrParam]=Data[CurrParam].apply(lambda d: d + abs(Min) ) 
			# Alternatively, if the data passed is not already normalized. 
			Data[CurrParam]=Data[CurrParam].apply(lambda d: (d-Mean+abs(Min))/(Std) )
			print "\t Processed-parameter: "+str(CurrParam)
		else:
			print "\t Did not process-parameter: "+str(CurrParam)
	return Data



		
