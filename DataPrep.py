#! /usr/bin/python
import sys,getopt,subprocess,re,math,commands,time,copy,random
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import pylab as Py
from operator import itemgetter, attrgetter, methodcaller

from sklearn import datasets, linear_model , cross_validation, metrics, tree, svm
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import SGDClassifier

from pylab import savefig
from sklearn.cross_validation import cross_val_score
from sklearn.datasets import make_blobs

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import spectral_clustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn import mixture


GlobalNumModels=11
GlobalNumClassModels=5
GlobalNumClusteringModels=6
GlobalNumClusters=10

ChoseRegModel=8
ChoseClassModel=1
ChoseClusterModel=0
GlobalNumFolds=2


GLNecessaryCombisNotDone=True
GLLeaveOneOut=False
#GLLeaveOneOut=True 
GLCheckAllModels=False 
GLCheckAllModels=True
GLModelDiagnostics=False 

"""
Regression:

0    LinearRegression
1    RidgeRegression
2    Lasso
3    ElasticNet
4    LassoLars
5    DecisionTreeRegressor
6    SVMRegressor
7    SGDRegressor
8    RandomForestRegressor
9    GradientBoostingRegressor
10    AdaBoostRegressor

Classification:

0    LogisticRegression
1    SVMClassification
2    SGDClassifier
3    DecisionTreeClassifier
4    GradientBoostingClassifier
5    RandomForestClassifier

Clustering 
0    KMeans
1    AffinityPropogation
2    MeanShift
3    Spectral Clustering
//4    Ward Hierarchial Clustering
4    Agglomerating Clustering
5    DBSCAN
6    Gaussian mixtures
"""

def usage():
    print "\t Usage: Learn.py -x <Training file name> \n\t Optional: \n\t\t -o <Output file name> \n\t\t -y <Test-file-name> \n\t\t -c <correlation-flag> 0: No correlaiton 1: IpStats correlation"
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
    
def ProcessRelativeEXECTIME(Data):    
    ProcessedRelativeEXECTIME=[]
    for idx,row in Data.iterrows():
        Temp= (row["RelativeEXECTIME"] - (2.6-row["Frequency1"]))
        ProcessedRelativeEXECTIME.append(Temp)
    Data['RelExeTime2']=ProcessedRelativeEXECTIME
    return Data

def ProcessXParams(Data,XParams):
    for CurrParam in XParams:
        for idx,row in Data.iterrows():
            Before=row[CurrParam]
            row[CurrParam]/=row["dmemops"] #row["dinsns"]
            #print "\t Before: "+str(Before)+" after: "+str(row[CurrParam]);sys.exit()    
    return Data
    
def Normalize(x,y,z):
    return (float(x)+float(y))/float(z)
    
def IsNumber(s):
# Credits: StackExchange: DanielGoldberg: http://stackoverflow.com/questions/354038/how-do-i-check-if-a-string-is-a-number-in-python
    try:
        float(s)
        return True
    except ValueError:
        return False

def RestofCols(ListOfCols,Column):
    return filter(lambda x: x!=Column, ListOfCols)

def ChoseLearnModel(UseModel,RegressionFlag=[],NumModels=GlobalNumModels):
    
    if(RegressionFlag[0]):    
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
        elif(UseModel==10):    
            Method='AdaBoostRegressor'
            rng = np.random.RandomState(1)
            LearnModel=AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),n_estimators=100, random_state=rng) # (loss="huber") #"epsilon_insensitive")
            print "\t AdaBoost Regressor!! "
    elif(RegressionFlag[1]):
        if(UseModel==0):
            LearnModel= linear_model.LogisticRegression(penalty='l1',tol=0.1)
            print " \t Classification: LogisticRegression !! "
            Method='LogisticRegression'
        elif(UseModel==1):
            alpha=0.5
            LearnModel = svm.SVC() 
            print " \t SVMClassification "
            Method='SVMClassification'
        elif(UseModel==2):
            alpha=0.01
            LearnModel= SGDClassifier(loss="hinge", penalty="l2")
            print " \t SGDClassifier "
            Method='SGDClassifier'
        elif(UseModel==3):
            alpha=0.001
            LearnModel= tree.DecisionTreeClassifier()
            print " \t DecisionTreeClassifier "
            Method='DecisionTreeClassifier'
        elif(UseModel==4):
            alpha=0.4
            LearnModel= GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
            print " \t GradientBoostingClassifier "
            Method='GradientBoostingClassifier'
        elif(UseModel==5):
            LearnModel = RandomForestClassifier()
            print "\t RandomForestClassifier "
            Method = 'RandomForestClassifier'
    elif(RegressionFlag[2]):
        if(UseModel==0):
            #LearnModel= KMeans(init='k-means++',n_clusters= GlobalNumClusters)
            LearnModel= KMeans(init='k-means++',n_clusters= GlobalNumClusters)
            print "\t Kmeans-Clustering "
            Method='KMeans-Clustering'
        elif(UseModel==1):
            LearnModel=AffinityPropagation(preference=-25)
            print "\t AffinityPropogation "
            Method='AffinityPropogation'
        elif(UseModel==2):
            LearnModel=MeanShift(bin_seeding=True)    
            print "\t MeanShift "
            Method='MeanShift'
        elif(UseModel==4):
            LearnModel=AgglomerativeClustering(n_clusters=GlobalNumClusters,affinity='euclidean')
            print "\t AgglomerativeClustering "
            Method='AgglomerativeClustering'
        elif(UseModel==5):
            LearnModel=DBSCAN(eps=0.3)
            print "\t DBSCAN "
            Method='DBSCAN'
        elif(UseModel==6):
            LearnModel=mixture.GMM(    n_components=GlobalNumClusters,covariance_type='spherical')
            print "\t GMM mixture"
            Method='GMM_Mixture'
        elif(UseModel==3):
            LearnModel= KMeans(init='k-means++',n_clusters= GlobalNumClusters)
            #LearnModel=spectral_clustering(X,n_clusters=GlobalNumClusters,eigen_solver='arpack')
            print "\t AgainKMeans" 
            Method='AgainKMeans'    
                
        """elif(UseModel==3):
            LearnModel=spectral_clustering(X,n_clusters=GlobalNumClusters,eigen_solver='arpack')
            print "\t SpectralClustering" 
            Method='SpectralClustering' """            

    return (LearnModel,Method)    

def Clustering(IpStats,XParams,RegressionFlag,UseModel=0,NumFolds=GlobalNumFolds,NumModels=GlobalNumModels):
    NecessaryCombisNotDone=GLNecessaryCombisNotDone
    LeaveOneOut=GLLeaveOneOut
    CheckAllModels=GLCheckAllModels
    ModelDiagnostics=GLModelDiagnostics
    
    AverageErrors={}
    AverageR2={}    
    
    if(RegressionFlag[0]):
        NumModels=GlobalNumModels
    elif(RegressionFlag[1]):
        NumModels=GlobalNumClassModels
    elif(RegressionFlag[2]):
        NumModels=GlobalNumClusteringModels

    for Idx,CurrCol in enumerate(XParams):    
      if(NecessaryCombisNotDone): 
        AverageErrors[CurrCol]={}
        AverageR2[CurrCol]={}
    
        #print "\t Columns: "+str(InputX.columns)
        RestOfColumns=XParams
        if(LeaveOneOut):
            print "\t Calling RestOfColumns "
            RestOfColumns=RestofCols(XParams,CurrCol)
            TmpX=IpStats[RestOfColumns]
            if(Idx==(len(XParams)-1)):
                NecessaryCombisNotDone=False    
        else:
            TmpX=IpStats[XParams]
            NecessaryCombisNotDone=False
        #TmpY=IpStats[YParams]
        print "\t Param: "+str(CurrCol)+"\t Rest of Columns: "+str(TmpX.columns)
    
        X=TmpX.as_matrix() 
        #Y=TmpY.as_matrix() 

        print "\t X shape: "+str(X.shape) #+" Y shape "+str(Y.shape)
        print "\t X-columns: "+str(TmpX.columns)
        IpStatsLen=len(IpStats)
        kf = cross_validation.KFold(IpStatsLen, n_folds=NumFolds)
        print "\t Len(K-folds): "+str(len(kf))
    
        LearnModel= linear_model.LinearRegression()
        alpha=0.0001
     
        for Idx in range(NumModels):
         Idx=NumModels-1
         AllModelsNotChecked=True
         for Idx in range(NumModels):
          if(CheckAllModels):
             print "\t CurrentModel index "+str(Idx)
          else:
             Idx=UseModel
          if(AllModelsNotChecked):
            (LearnModel,Method)=ChoseLearnModel(Idx,RegressionFlag) #=True)         
            AvgRelativeErrorAcrossFolds=0.0    
            AvgRelativeR2AcrossFolds=0.0
            if ModelDiagnostics:
                ModelDiagnosticsOutput=open('ModelDiagnosticsOutput.dat','w')
                ModelDiagnosticsOutput.write("\tIndex\tOutputY\tTestY\tAbsError\n\n")
                FoldIdx=-1
            for train_index, test_index in kf:
            
                #print("TRAIN:", train_index, "TEST:", test_index)
                TrainX, TestX = X[train_index], X[test_index]
                #TrainY, TestY = Y[train_index], Y[test_index]
                
                # PCA code.
                #pca = PCA().fit(TrainX)
                #LearnModel=KMeans(init=pca.components_,n_clusters=GlobalNumClusters,n_init=1)
                #print "\t pca comoponents: {0:s} ".format(str(pca.components_) )
                
                LearnModel.fit(TrainX) #,TrainY.ravel())
                #print "\t Params: "+str(LearnModel.feature_importances_)
                #print('Coeff: ', LearnModel.coef_)
                #print ('LinearModel: ',LearnModel)
                OutputY=LearnModel.fit_predict(TestX)
                                        
                ExplainedVarianceR2=0 #( LearnModel.score(TestX) )    # This is actually R2
 
                AvgRelativeR2AcrossFolds+=(ExplainedVarianceR2)
                #print "\t R2/Explained variance: "+str(ExplainedVarianceR2)
            
            AverageR2[CurrCol][Method]=float(AvgRelativeR2AcrossFolds/NumFolds)
            #for ClusterNum,CurrCluster in enumerate(LearnModel.cluster_centers_):
            #    print "\t {0:d} \t {1:s} ".format(ClusterNum, str(CurrCluster) )
            #print "\t NumClusters: {0:d} ".format(len(LearnModel.labels_))
            print "\t AverageR2/ExplainedVariance: "+str(AverageR2[CurrCol][Method])+" CumulativeRelativeErrorAcrossFolds \n"+str(AvgRelativeR2AcrossFolds)
            if(not(CheckAllModels)):
                AllModelsNotChecked=False

    print "\n\n\t ----- Summary -------- \n\n"
    print "\t Parameter \t LearnModel \t\t\t\t\t Avg.Error \t R2 "
    if(LeaveOneOut):        
      for CurrCol in XParams:
       if( (CurrCol in AverageR2) ):
        print "\t CurrParam: "+str(CurrCol)
        for CurrMethod in AverageR2[CurrCol]:
              print "\t "+str(CurrCol)+"\t "+str(CurrMethod)+"\t\t\t\t\t "+str(round(AverageR2[CurrCol][CurrMethod],6))    
    else:
      CurrParam=XParams[0]
      for CurrMethod in AverageR2[CurrParam]:
          if(CurrParam in AverageR2):
              print "\t "+str(CurrParam)+"\t "+str(CurrMethod)+"\t\t\t\t\t "+str(round(AverageR2[CurrParam][CurrMethod],6))
    
    
         
def CrossValidation(IpStats,XParams,YParams,RegressionFlag,UseModel=0,NumFolds=GlobalNumFolds,NumModels=GlobalNumModels):

 AverageErrors={}
 AverageR2={}

 NecessaryCombisNotDone=GLNecessaryCombisNotDone
 LeaveOneOut=GLLeaveOneOut
 CheckAllModels=GLCheckAllModels
 ModelDiagnostics=GLModelDiagnostics
 
 #if(not(RegressionFlag[0])):
 #    if(NumModels>GlobalNumClassModels):
 #        NumModels=GlobalNumClassModels
 
 if(RegressionFlag[0]):
     NumModels=GlobalNumModels
 elif(RegressionFlag[1]):
     NumModels=GlobalNumClassModels
 elif(RegressionFlag[2]):
     NumModels=GlobalNumClusteringModels
         
 for Idx,CurrCol in enumerate(XParams):    
  if(NecessaryCombisNotDone): 
    AverageErrors[CurrCol]={}
    AverageR2[CurrCol]={}
    
    #print "\t Columns: "+str(InputX.columns)
    RestOfColumns=XParams
    if(LeaveOneOut):
        print "\t Calling RestOfColumns "
        RestOfColumns=RestofCols(XParams,CurrCol)
        TmpX=IpStats[RestOfColumns]
        if(Idx==(len(XParams)-1)):
            NecessaryCombisNotDone=False    
    else:
        TmpX=IpStats[XParams]
        NecessaryCombisNotDone=False
    TmpY=IpStats[YParams]
    print "\t Param: "+str(CurrCol)+"\t Rest of Columns: "+str(TmpX.columns)
    
    X=TmpX.as_matrix() 
    Y=TmpY.as_matrix() 

    print "\t X shape: "+str(X.shape)+" Y shape "+str(Y.shape)
    print "\t X-columns: "+str(TmpX.columns)
    IpStatsLen=len(IpStats)
    kf = cross_validation.KFold(IpStatsLen, n_folds=NumFolds)
    print "\t Len(K-folds): "+str(len(kf))
    
    LearnModel= linear_model.LinearRegression()
    alpha=0.0001
 
    for Idx in range(NumModels):
     Idx=NumModels-1
     AllModelsNotChecked=True
     for Idx in range(NumModels):
      if(CheckAllModels):
         print "\t CurrentModel index "+str(Idx)
      else:
         Idx=UseModel
      if(AllModelsNotChecked):
        (LearnModel,Method)=ChoseLearnModel(Idx,RegressionFlag) #=True)         
        AvgRelativeErrorAcrossFolds=0.0    
        AvgRelativeR2AcrossFolds=0.0
        if ModelDiagnostics:
            ModelDiagnosticsOutput=open('ModelDiagnosticsOutput.dat','w')
            ModelDiagnosticsOutput.write("\tIndex\tOutputY\tTestY\tAbsError\n\n")
            FoldIdx=-1
        for train_index, test_index in kf:
            
            #print("TRAIN:", train_index, "TEST:", test_index)
            TrainX, TestX = X[train_index], X[test_index]
            TrainY, TestY = Y[train_index], Y[test_index]
            LearnModel.fit(TrainX,TrainY.ravel())
            #print "\t Params: "+str(LearnModel.feature_importances_)
            #print('Coeff: ', LearnModel.coef_)
            #print ('LinearModel: ',LearnModel)
            OutputY=LearnModel.predict(TestX)
            
            if ModelDiagnostics:
                FoldIdx+=1
                FoldIdxScaled=int(FoldIdx*IpStatsLen/NumFolds)
                for i in range(len(TestY)):
                    AbsError=abs(TestY[i][0]-OutputY[i])/TestY[i]
                    ModelDiagnosticsOutput.write("\t"+str(FoldIdxScaled+i)+"\t"+str(round(OutputY[i],6))+"\t"+str(round(TestY[i][0],6))+"\t"+str(round(AbsError,6)))
                            
            ExplainedVarianceR2=( LearnModel.score(TestX,TestY) )    # This is actually R2
            MeanRelativeError=( np.mean(( abs(LearnModel.predict(TestX)- TestY)/TestY) ** 1) )
            #MeanRelativeError=( np.mean(( (LearnModel.predict(TestX)- TestY)) ** 2) )
            
            AbsError=0.0
            for i in range(len(TestY)):
                Temp=abs(TestY[i][0]-OutputY[i])/TestY[i]
                AbsError+=Temp
            AbsError/=len(TestY)
                        
            AvgRelativeErrorAcrossFolds+=(AbsError) #(MeanRelativeError) (MeanRelativeError)
            AvgRelativeR2AcrossFolds+=(ExplainedVarianceR2)
            print "\t MeanRelativeError: "+str(AbsError)+"\t R2/Explained variance: "+str(ExplainedVarianceR2)
            if hasattr(LearnModel, 'feature_importances_'): 
                FImp=(LearnModel.feature_importances_)
                FeatureImp=[]
                for Idx,CurrParam in enumerate(RestOfColumns):
                    Temp=[]
                    Temp.append(round(FImp[Idx],5))
                    Temp.append(CurrParam)
                #print "\t CurrParam: "+str(CurrParam)+"\t Temp "+str(Temp)
                    FeatureImp.append(Temp)
                SortedFeatureImp=sorted(FeatureImp, key=itemgetter(0))
                for Idx,CurrParam in enumerate(SortedFeatureImp):
                    print "\t CurrParam: "+str(CurrParam)    
            #sys.exit()
            """if(Idx==1):
                 print("Residual sum of squares: %.5f "%MeanRelativeError+"\tVariance score: %.2f "%LearnModel.score(TestX, TestY))+"\t alpha: "+str(LearnModel.alpha_)
             else:
                 print("Residual sum of squares: %.5f "%MeanRelativeError+"\tVariance score: %.2f "%LearnModel.score(TestX, TestY))"""
            
        AverageErrors[CurrCol][Method]=    float(AvgRelativeErrorAcrossFolds/NumFolds)
        AverageR2[CurrCol][Method]=float(AvgRelativeR2AcrossFolds/NumFolds)
        if ModelDiagnostics:
            ModelDiagnosticsOutput.close()
        print "\t AvgRelativeErrorAcrossFolds: "+str(AverageErrors[CurrCol][Method])+" CumulRelativeErrorAcrossFolds "+str(AvgRelativeErrorAcrossFolds)
        print "\t AverageR2/ExplainedVariance: "+str(AverageR2[CurrCol][Method])+" CumulativeRelativeErrorAcrossFolds "+str(AvgRelativeR2AcrossFolds)
        if(not(CheckAllModels)):
            AllModelsNotChecked=False

 print "\n\n\t ----- Summary -------- \n\n"
 print "\t Parameter \t LearnModel \t\t\t\t\t Avg.Error \t R2 "
 if(LeaveOneOut):        
  for CurrCol in XParams:
   if( (CurrCol in AverageErrors) and (CurrCol in AverageR2) ):
    print "\t CurrParam: "+str(CurrCol)
    for CurrMethod in AverageErrors[CurrCol]:
          print "\t "+str(CurrCol)+"\t "+str(CurrMethod)+"\t\t\t\t\t "+str(round(AverageErrors[CurrCol][CurrMethod],6))+"\t "+str(round(AverageR2[CurrCol][CurrMethod],6))    
 else:
  CurrParam=XParams[0]
  for CurrMethod in AverageErrors[CurrParam]:
      if(CurrParam in AverageR2):
          print "\t "+str(CurrParam)+"\t "+str(CurrMethod)+"\t\t\t\t\t "+str(round(AverageErrors[CurrParam][CurrMethod],6))+"\t "+str(round(AverageR2[CurrParam][CurrMethod],6))

def CheckTolerance(Num,Tolerance,List):
    return next( (x for x in List if( ( x > ( Num*(1-Tolerance) ) ) & ( x < ( Num*(1+Tolerance) ) ) ) ) , False)

def BinParam(Data,Param,Bins):
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

def main(argv):
    InputFileName=''
    OutputFileName=''
    IpRegressionFlag=''
    IpClassificationFlag=''
    IpClusteringFlag=''
    IpCheckAllModelsFlag=''
    TestFileName=''
    CorrelationFlag=''
    verbose=False 
    try:
       opts, args = getopt.getopt(sys.argv[1:],"x:o:r:i:y:c:h:v",["training=","output=","regflag=","checkallflag=","test=","correlation","help","verbose"])
    except getopt.GetoptError:
        #print str(err) # will print something like "option -a not recognized"
       usage()
       sys.exit(2)
      
    for opt, arg in opts:
        print "\t Opt: "+str(opt)+" argument "+str(arg)    
        if opt == '-h':
            usage()        
        elif opt in ("-x", "--training"):
            InputFileName=RemoveWhiteSpace(arg)
            print "\t Input file is "+str(InputFileName)+"\n";
        elif opt in ("-y", "--test"):
            TestFileName=RemoveWhiteSpace(arg)
            print "\t Test file is "+str(TestFileName)+"\n";    
        elif opt in ("-r", "--regflag"):
            TempArg=int(RemoveWhiteSpace(arg))
            if(TempArg==0):
                IpRegressionFlag=True
                IpClassificationFlag=False
                IpClusteringFlag=False
            elif(TempArg==1):
                IpRegressionFlag=False
                IpClassificationFlag=True
                IpClusteringFlag=False
            elif(TempArg==2):
                IpRegressionFlag=False
                IpClassificationFlag=False
                IpClusteringFlag=True    
            else:
                print "\t ERROR: Illegal value provided for \"-r\" flag.     "
                usage()    
            print "\t Regr: "+str(IpRegressionFlag)+"\t Class: "+str(IpClassificationFlag)+"\t Clustering: "+str(IpClusteringFlag)
                
            print "\t IpRegressionFlag is "+str(IpRegressionFlag)+"\n";
        elif opt in ("-i", "--checkallflag"):
            TempArg=int(RemoveWhiteSpace(arg))
            if(TempArg>0):
                IpCheckAllModelsFlag=True
            else:
                IpCheckAllModelsFlag=False
            
            print "\t IpCheckAllModelsFlag is "+str(IpCheckAllModelsFlag)+"\n";                                        
        elif opt in ("-c", "--correlation"):
            CorrelationFlag=RemoveWhiteSpace(arg)
            print "\t Correlation flag is "+str(CorrelationFlag)+"\n";            
        elif opt in ("-o", "--output"):
            OutputFileName=RemoveWhiteSpace(arg)
            print "\t Source file is "+str(OutputFileName)+"\n";            
        else:
               usage()

    if(len(opts)==0):
        usage()

    if(InputFileName==''):
        usage()
    if(OutputFileName==''):
        OutputFileName='DefaultOutputFile.log'
        print "\t INFO: Using default output file name: "+str(OutputFileName)
    if(CorrelationFlag==''):
        CorrelationFlag=0
        print "\t\t INFO: Using default correlation flag: "+str(CorrelationFlag)    
    
    IpStats=pd.read_csv(InputFileName,sep='\t',header=0)
    IpStats=shuffle(IpStats);IpStats=shuffle(IpStats);IpStats=shuffle(IpStats)
    print "\t IpStats.shape: "+str(IpStats.shape)
    
    YParams= ["RelativeEXECTIME"] #["Slowdown_Class1"] #["RelativeEXECTIME"] #["RelativeACPOWER"]
    #YParams=["ACPOWER"] =["pintops","pstores","pfpops"]
    XYParams=[]
    AllParams=["pintops","pmemops","ploads","pstores","pfpops","pbranchops","l1hr","l2hr","l3hr","l1mpi","l2mpi","l3mpi","idu","fdu","idu2","fdu2","mdu2","fprat","bytespermop"]    
    MostXParams=["pintops","pbranchops","l1hr","l2hr","l3hr","l1mpi","l2mpi","l3mpi","fdu2","mdu2"]

    ReallyImpParams=["l1hr","l1mpi","l3mpi"] # Training_NestedStreams_23.dat
    ReallyImpParams=["l1hr","l3mpi"] #    Training_NestedStreams_23.dat
        
    ReallyImpParams=["pfpops","idu","l1mpi","l1hr","l3mpi"] # Training_Streams_2.dat
    ReallyImpParams=["l1hr","l3mpi"] # Training_Streams_2.dat 
    
    ReallyImpParams=["idu","l2mpi","l3mpi","l2hr","idu2","l1hr","l1mpi","pintops"] # Training_Streams_3.dat
    #ReallyImpParams=["idu2","l1mpi","pintops"] # Training_Streams_3.dat
    
    ReallyImpParams=["l1mpi","l2mpi","l3mpi"] # Training_NestedStreams_22.dat
    ReallyImpParams=["l3mpi"] # Training_NestedStreams_22.dat
    
    ReallyImpParams=["l1hr","l3mpi"] # Training_NestedStreams_23.dat
    
    ReallyImpParams=["idu","l2mpi","l3mpi","l2hr","idu2","l1hr","l1mpi","pintops"] # Training_Streams_3.dat
    #ReallyImpParams=["idu","l3mpi","l1mpi","pintops"] # Training_Streams_3.dat
    #ReallyImpParams=["l3mpi","l1hr","l2hr","pintops","l1mpi"]
    ReallyImpParams=["idu","l2mpi","l3mpi","l2hr","idu2","l1hr","l1mpi","pintops"] 
    
    SpatParams=["spat0","spat2","spat4","spat8","spat16","spat32","spat64","spat128","spatOther"]
    SpatParams1=[]#"spat0","spat8","spat16","spat32","spat64","spat128","spatOther"]
    SpatParams2=["spat0","spat8","spatOther"]

    ReallyImpParams=["l3mpi"] # Training_NestedStreams_23.dat
    XParams= ReallyImpParams
    XParams= AllParams
    
    NowUseXParams=AllParams
    XParams= NowUseXParams
    
    UsingSpatParams=False
    
    #YParams=["RelExeTime2"]
    UsingRelExe2Time=False
    
    for CurrParam in YParams:
        XYParams.append(CurrParam)
    for CurrParam in XParams:
        XYParams.append(CurrParam)
    XYParams.append('Frequency')
    for currParm in XYParams:
        print "\t %s "%(currParm)
    
    #XParams=["pintops","pmemops","ploads","pstores","pfpops","pbranchops","l1hr","l2hr","l3hr","l1mpi","l2mpi","l3mpi","idu","fdu","idu2","fdu2","mdu2","fprat","bytespermop","pvec512","pvec256","pvec128","spatial","spat0","spat2","spat4","spat8","spat16","spat32","spat64","spat128","spatOther","Frequency1","Iterations1"]
    print "\t len(XParams): "+str(len(XParams))
    TempX=IpStats[XParams]

    TestValueTolerance=0.1 # i.e, +-10% of training values around test value is accepted.
    print "\t YParams.shape: "+str(len(YParams))
    #TempTrainSet=IpStats;TempTrainSet=ProcessRelativeEXECTIME(TempTrainSet); RegModels(TempTrainSet,XParams,YParams,UseModel=0,NumFolds=GlobalNumFolds,NumModels=GlobalNumModels);sys.exit()
    ChoseModel=''
    if(IpRegressionFlag):
       ChoseModel=ChoseRegModel
    elif(IpClassificationFlag):
       ChoseModel=ChoseClassModel
    elif(IpClusteringFlag):
       ChoseModel=ChoseClusterModel
    CrossValidation(IpStats,XParams,YParams,RegressionFlag=[IpRegressionFlag,IpClassificationFlag,IpClusteringFlag],UseModel=ChoseModel,NumFolds=GlobalNumFolds,NumModels=GlobalNumModels);#sys.exit()
    (LearnModel,Method)=ChoseLearnModel(ChoseModel,RegressionFlag=[IpRegressionFlag,IpClassificationFlag,IpClusteringFlag])
    IpStats=IpStats[IpStats.Frequency!=2.9]
    #ClusterParams=["l1hr","l1mpi","l3mpi","Frequency"]
    #OutputY=LearnModel.fit_predict(IpStats[ClusterParams])
    #IpStats['Cluster']=OutputY
    #IpStats.to_csv('Clustered_'+str(InputFileName),sep='\t'); sys.exit()
    #Clustering(IpStats,XParams,RegressionFlag=[IpRegressionFlag,IpClassificationFlag,IpClusteringFlag],UseModel=ChoseModel,NumFolds=GlobalNumFolds,NumModels=GlobalNumModels);sys.exit()
    
    if(CorrelationFlag==1):
       CorrOutput=IpStats.corr(method='pearson')
       InputCorrelationFileName='CorrelationInput'+str(OutputFileName)
       CorrOutput.to_csv(InputCorrelationFileName,sep='\t')
       sys.exit()
    
    if(TestFileName!=''):
        print "\t Opening test file name: "+str(TestFileName)
        #IpFile=open(TestFileName)
        TestStats=pd.read_csv(TestFileName,sep='\t',header=0)
        CorrOutput=TestStats.corr(method='pearson')
        InputCorrelationFileName='CorrelationInput'+str(OutputFileName)
        #CorrOutput.to_csv(InputCorrelationFileName,sep='\t')
        #sys.exit()
        print "\t TestStats-shape: "+str(TestStats.shape)
        
        TestX=TestStats#[XYParams]
        print "\t TestX-shape "+str(TestX.shape)
        FilteredTestX=TestX
        """for Idx,CurrCol in enumerate(XParams):
            TempTestX=pd.DataFrame(columns=TestX.columns)
            print "\t Curr-param: "+str(CurrCol)+"\t FilteredTestX-shape: "+str(FilteredTestX.shape)
            for index,CurrRow in FilteredTestX.iterrows():
                TestVal=CurrRow[CurrCol]
                #print "\t CurrCol: "+str(TestVal)
                CheckRangeResult=CheckTolerance(TestVal,TestValueTolerance,TempX[CurrCol])            
                if(CheckRangeResult):
                    #print "\t Tolerance-check result: "+str(CheckRangeResult)
                    df=CurrRow.copy()
                    TempTestX=TempTestX.append(df,ignore_index=True)
                    
            print "\t TempTestX-shape "+str(TempTestX.shape)    
            FilteredTestX=TempTestX    
        
        #TestX=(TestStats[XParams]).as_matrix()
        #TestY=(TestStats[YParams]).as_matrix() """
        XParams.append('Frequency')    
        
        TempTestSet=FilteredTestX
        if(UsingRelExe2Time):
            TempTestSet=ProcessRelativeEXECTIME(TempTestSet)
        #TempTestSet.to_csv('RelExe2TestValuesWithoutBaseFrequency.dat',sep='\t');sys.exit()
        #TempTestSet=TempTestSet[TempTestSet.RelativeEXECTIME>1]
        #TempTestSet=TempTestSet[TempTestSet.Slowdown_Class1==5]
        #TempTestSet=TempTestSet[TempTestSet.l3hr>40]
        #TempTestSet=TempTestSet[ (TempTestSet.RelativeEXECTIME>1.05) ]
        TempTestSet=TempTestSet[TempTestSet.Frequency!=2.9]
        #TempTestSet.to_csv('RelExe2TestValuesWithoutBaseFrequency.dat',sep='\t');sys.exit()
        
        if(UsingSpatParams):
            TempTestSet=ProcessXParams(TempTestSet,SpatParams)

        if(IpCheckAllModelsFlag):
            XParams=AllParams    
            
        TestX=(TempTestSet[XParams]).as_matrix()
        TestY=(TempTestSet[YParams]).as_matrix()
        
        #TempTrainSet=IpStats[IpStats.Frequency1!=2.6]
        TempTrainSet=IpStats
        if(UsingRelExe2Time):
            TempTrainSet=ProcessRelativeEXECTIME(TempTrainSet)
        #TempTrainSet=TempTrainSet[ (TempTrainSet.Frequency1!=2.6)]
        #TempTrainSet=TempTrainSet[(TempTrainSet.Slowdown_Class1==5) ]#| (TempTrainSet.Slowdown_Class1==6)]
        TempTrainSet=TempTrainSet[TempTrainSet.Frequency!=2.9]
        #TempTrainSet.to_csv('RelExe2TrainingValuesWithoutBaseFrequency.dat',sep='\t')
        #sys.exit()
        
        if(UsingSpatParams):
            TempTestSet=ProcessXParams(TempTestSet,SpatParams)
        TrainX=(TempTrainSet[XParams]).as_matrix()
        TrainY=(TempTrainSet[YParams]).as_matrix()
        print "\t Finally the test-x and y is: "+str(TestX.shape)+" , "+str(TestY.shape)
        print "\t And the train-x and y is: "+str(TrainX.shape)+" , "+str(TrainY.shape)
        print "\t XParams: "+str(XParams)
        print "\t YParams: "+str(YParams)
        #TrainX.to_csv('FilteredTest.dat')
        #sys.exit()
        
        VarianceScoreCollection={}
        ManualRelativeErrorCollection={}
        NumModelsToUse=-1
        ModelToChose=-1
        if(IpRegressionFlag):
            NumModelsToUse=GlobalNumModels
            ModelToChose=ChoseRegModel
        elif(IpClassificationFlag):
            NumModelsToUse=GlobalNumClassModels
            ModelToChose=ChoseClassModel
        elif(IpClusteringFlag):
            NumModelsToUse=GlobalNumClusteringModels
            ModelToChose=ChoseClusteringModel    
        
        #for ModelIdx in range(GlobalNumModels):
        for ModelIdx in range(NumModelsToUse):
            if(not(IpCheckAllModelsFlag)):
                ModelIdx=ModelToChose 
            print "\n"
            (LearnModel,Method)=ChoseLearnModel(UseModel=ModelIdx,RegressionFlag=[IpRegressionFlag,IpClassificationFlag,IpClusteringFlag])#True)
            LearnModel.fit(TrainX,TrainY.ravel())
            OutputY=LearnModel.predict(TestX)
            AbsError=[]
            ManualMeanRelativeError=0.0
            for i in range(len(OutputY)):
                Temp=float(abs(OutputY[i]-TestY[i][0])/TestY[i][0])
                ManualMeanRelativeError+=Temp
                AbsError.append(Temp)
            ManualMeanRelativeError/=len(OutputY)
            MeanRelativeError=( np.mean(( abs(OutputY- TestY[0])/TestY[0]) ** 1) )
            VarianceScore=LearnModel.score(TestX,TestY)
            print "\t MeanRelativeError for the test set is: "+str(MeanRelativeError)+" Variance-score: "+str(VarianceScore)+" ManualMeanRelativeError "+str(ManualMeanRelativeError)
            TestOutputComparison=open('TestOutputComparison.dat','w')
            TestVecLen=len(TestX[0])
            TempTestSet['AbsError']=AbsError
            for i in range(len(OutputY)):
                TestOutputComparison.write("\t"+str(i)+"\t"+str(round(OutputY[i],6))+"\t"+str(round(TestY[i][0],6))+"\t"+str(round(AbsError[i],6)))
                #NecessaryRow=FindInDF(XParams[TestVecLen-1],XParams[TestVecLen-2],TestX[i][TestVecLen-1],TestX[i][TestVecLen-2],TempTestSet)
                #for CurrParam in TestX[i]:
                #    TestOutputComparison.write("\t"+str(CurrParam))
                #for CurrParam in NecessaryRow:
                    #TestOutputComparison.write("\t"+str(CurrParam))
                
                #sys.exit()
            TestOutputComparison.close()    
        
            #Meh=LearnModel.decision_function([TestX[0]])
            #print "\t Feature importance: "+str(Meh.shape[1]); sys.exit()
        
            VarianceScoreCollection[Method]=VarianceScore
            ManualRelativeErrorCollection[Method]=ManualMeanRelativeError
            if(not(IpCheckAllModelsFlag)):
                print "\t Feature importance: "+str(LearnModel.feature_importances_); sys.exit() 
                #print "\t Feature importance: "+str(LearnModel.coef_); sys.exit()

            if(ModelDiagnostics):
                AbsErrorBins=[(0.0,0.049),(0.05,0.099),(0.1,0.199),(0.2,0.24999),(0.25,2)]
                BinnedTestData=BinParam(TempTestSet,"AbsError",AbsErrorBins)
                AvgBinnedTestData={}
                VarBinnedTestData={}

                for CurrKey in BinnedTestData:
                    AvgBinnedTestData[CurrKey]={}
                    VarBinnedTestData[CurrKey]={}
                    print "\t CurrKey: "+str(CurrKey)+" shape: "+str(BinnedTestData[CurrKey].shape)
                    print "\t Average\t Variance\t Minimum\t Maximum "
                    for CurrCol in AllParams:
                    #print "\t CurrCol: "+str(CurrCol)
                        AvgBinnedTestData[CurrKey][CurrCol]=BinnedTestData[CurrKey][CurrCol].mean()
                        VarBinnedTestData[CurrKey][CurrCol]=BinnedTestData[CurrKey][CurrCol].var()
                        if(CurrCol=='l1hr'):
                            LessThan50=BinnedTestData[CurrKey][ ( BinnedTestData[CurrKey][CurrCol] < 50) ]
                            #print "\t ****Shape**** "+str(LessThan50.shape)
                        print "\t\t"+str(CurrCol)+"\t"+str(AvgBinnedTestData[CurrKey][CurrCol])+"\t"+str(VarBinnedTestData[CurrKey][CurrCol])+"\t"+str(BinnedTestData[CurrKey][CurrCol].min())+"\t"+str(BinnedTestData[CurrKey][CurrCol].max())
                
        
                print "\t MeanRelativeError for the test set is: "+str(MeanRelativeError)+" Variance-score: "+str(VarianceScore)+" ManualMeanRelativeError "+str(ManualMeanRelativeError);print "\t Feature importance: "+str(LearnModel.feature_importances_); sys.exit()
                print "\t Feature importance: "+str(LearnModel.coef_); sys.exit() #"""
        
        
        print "\t Format: <Method> <Variance-score> <ManualRelativeError> "    
        for CurrMethod in  VarianceScoreCollection:
            if(CurrMethod in ManualRelativeErrorCollection):
                print "\t "+str(CurrMethod)+"\t "+str(round(VarianceScoreCollection[CurrMethod],6))+"\t "+str(round(ManualRelativeErrorCollection[CurrMethod],6))
        
        

    print "\n\n"     
          
      
if __name__ == "__main__":
   main(sys.argv[1:])

