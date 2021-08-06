import datetime
from isoelectric import ipc #Requires Isoelectric Package
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import statistics
from propy import PyPro #Requires Installing Propy Package
import propy #Requires Installing Propy Package
from propy.PyPro import GetProDes #Requires Installing Propy Package
import pickle
def Merge(dict1, dict2): 
    return(dict2.update(dict1))
pp=[]

#Create Dictionary of Sequences
c={}
x = range(1,len(b)+1)
dic={}
for i in b:
    dic[b.index(i)+1]=i
#Calculate Properties
aggDict = pickle.load(open('aggDict.pkl', 'rb'))#Requires aggDict File
chargeDict = pickle.load(open('chargeDict.pkl', 'rb'))#Requires chargeDict File
isoElP=[]
agg=[]
chargeDensities=[]
dsss={"seq":b}
for key, val in dic.items():
    isoElP.append(ipc.predict_isoelectric_point(val,"IPC_peptide"))
    agList=[]
    chargeList=[]
    for j in list(val):
        agList.append(aggDict[j])
        chargeList.append(chargeDict[j])
    agg.append(statistics.mean(agList))
    chargeDensities.append(sum(chargeList)/ipc.calculate_molecular_weight(i))
    Des=GetProDes(val)
    aACompDes=Des.GetAAComp() #AA Composition
    dPCompDes=Des.GetDPComp() #DP Composition
    moreauBrotoAutoDes=Des.GetMoreauBrotoAuto() #
    moranAutoDes=Des.GetMoranAuto() #
    gearyAutocDes=Des.GetGearyAuto() #
    cTDDes=Des.GetCTD()
    aPAACDes=Des.GetAPAAC(lamda=5,weight=0.05) #Type II PseudoAAC
    paacDes=Des.GetPAAC(lamda=5,weight=0.05) #Type I PseudoAAC
    sOCNDes=Des.GetSOCN()
    qSODes=Des.GetQSO(maxlag=30,weight=0.1)
    Merge(qSODes, sOCNDes)
    Merge(sOCNDes, paacDes)
    Merge(paacDes, aPAACDes)
    Merge(aPAACDes, cTDDes)
    Merge(cTDDes, gearyAutocDes)
    Merge(gearyAutocDes, moranAutoDes)
    Merge(moranAutoDes, moreauBrotoAutoDes)
    Merge(moreauBrotoAutoDes, dPCompDes)
    Merge(dPCompDes, aACompDes)
    pddd=aACompDes.copy()
    pp.append(pddd)
dsss["isoElP"]=isoElP
dsss["aggregationPropensityInVivo"]=agg
dsss["chargeDensity"]=chargeDensities
df=pd.DataFrame(data=pp)
p5=pd.DataFrame(data=dsss)
p6=pd.concat([p5, df.reindex(p5.index)], axis=1)
#Rescaling Data
#Needs Final Features MinMaxes
p=[]
features=["isoElP","chargeDensity","aggregationPropensityInVivo","_NormalizedVDWVD1075","MoreauBrotoAuto_Hydrophobicity1","GearyAuto_AvFlexibility4","_SecondaryStrD1050","_PolarizabilityD2001","tausw8","tausw2","APAAC2","GearyAuto_ResidueVol12","MoreauBrotoAuto_ResidueVol2","GearyAuto_ResidueASA7","L","MoranAuto_ResidueVol15","MoranAuto_AvFlexibility15","MoranAuto_ResidueVol10","W","GearyAuto_Mutability10","MoreauBrotoAuto_Steric1","QSOgrant30","_HydrophobicityD2075","_HydrophobicityD3001","MoranAuto_Polarizability2","MoranAuto_Polarizability7","GearyAuto_ResidueVol6","GearyAuto_Mutability9","GearyAuto_Mutability7","_HydrophobicityT12","MoreauBrotoAuto_Hydrophobicity14","MoreauBrotoAuto_Hydrophobicity17","MoreauBrotoAuto_Hydrophobicity12","_PolarityT23","MoranAuto_Mutability12","QSOSW29","QSOgrant22","QSOgrant21","_PolarizabilityT23","MoranAuto_Steric8","MoreauBrotoAuto_Polarizability14","MoranAuto_ResidueASA12","GearyAuto_Hydrophobicity10","MoranAuto_FreeEnergy11","_PolarizabilityC2","_ChargeD1001","MoranAuto_Steric10","_PolarityC1","_PolarityC3","MoranAuto_Hydrophobicity17","MoranAuto_Hydrophobicity10","MoreauBrotoAuto_FreeEnergy5","QSOSW16","QSOSW12","MoreauBrotoAuto_AvFlexibility4","MoreauBrotoAuto_AvFlexibility6","_NormalizedVDWVD2001","taugrant6","GearyAuto_Steric8","_SecondaryStrT13","MoreauBrotoAuto_Steric17","MoranAuto_FreeEnergy6","_PolarizabilityD2050","MoreauBrotoAuto_Steric10","GearyAuto_Polarizability8","GearyAuto_Polarizability1","GearyAuto_Polarizability3","MoreauBrotoAuto_Steric11","MoranAuto_Mutability7","MoreauBrotoAuto_Mutability1","_NormalizedVDWVC3","_SecondaryStrD1100","_HydrophobicityC2","_PolarizabilityT13","_PolarityD3001","MoranAuto_AvFlexibility7","MoranAuto_AvFlexibility3","_SecondaryStrD2001","MoreauBrotoAuto_FreeEnergy10","MoreauBrotoAuto_FreeEnergy11","MoreauBrotoAuto_FreeEnergy12","MoreauBrotoAuto_FreeEnergy13","MoreauBrotoAuto_FreeEnergy16","_ChargeT12","_SolventAccessibilityD1075","_NormalizedVDWVT13","GearyAuto_FreeEnergy8","MoreauBrotoAuto_ResidueVol10","GearyAuto_FreeEnergy12","GearyAuto_Steric11"
]
d=[]
newds={"id":list(dic.keys())}
fMinMax=pd.read_csv("90FeaturesMinMax.csv")
for j in features:
    p=[]
    for i in p6.loc[:,j]:
        p.append((i-fMinMax.loc[0, j])/(fMinMax.loc[1, j]-fMinMax.loc[0, j]))
    newds[j]=p 
newdf=pd.DataFrame(data=newds)
newdf
rfModel = pickle.load(open('rf_tr_cv_sf_final_model.sav', 'rb'))
svcModel = pickle.load(open('svc_twice_cv_final_model.sav', 'rb'))
mr=newdf.iloc[:, 0].values
print('Random Forest Classifier')
for i in mr:
    take=np.where(mr == i)
    print(i,rfModel.predict(newdf.iloc[take[0],1:].values))
print('-------------------------')
print('Support Vector Classifier')
for i in mr:
    take=np.where(mr == i)
    print(i,svcModel.predict(newdf.iloc[take[0],1:].values))