# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 19:24:08 2019

@author: arno
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import os
from winsound import Beep
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn import svm
from sklearn.isotonic import IsotonicRegression
from scipy.stats import levene

# Importing the data

os.chdir("C:\\Users\\arno\\Downloads\\credit scoring stuff\\data")

data=[[pd.read_csv("jakobshavn11.csv"),pd.read_csv("southeast11.csv"),pd.read_csv("storstrommen11.csv")],
[pd.read_csv("jakobshavn12.csv"),pd.read_csv("southeast12.csv"),pd.read_csv("storstrommen12.csv")],
[pd.read_csv("jakobshavn13.csv"),pd.read_csv("southeast13.csv"),pd.read_csv("storstrommen13.csv")],
[pd.read_csv("jakobshavn14.csv"),pd.read_csv("southeast14.csv"),pd.read_csv("storstrommen14.csv")],
[pd.read_csv("jakobshavn15.csv"),pd.read_csv("southeast15.csv"),pd.read_csv("storstrommen15.csv")],
[pd.read_csv("jakobshavn16.csv"),pd.read_csv("southeast16.csv"),pd.read_csv("storstrommen16.csv")]]

ndata=[[pd.read_csv("njakobshavn11.csv"),pd.read_csv("nsoutheast11.csv"),pd.read_csv("nstorstrommen11.csv")],
[pd.read_csv("njakobshavn12.csv"),pd.read_csv("nsoutheast12.csv"),pd.read_csv("nstorstrommen12.csv")],
[pd.read_csv("njakobshavn13.csv"),pd.read_csv("nsoutheast13.csv"),pd.read_csv("nstorstrommen13.csv")],
[pd.read_csv("njakobshavn14.csv"),pd.read_csv("nsoutheast14.csv"),pd.read_csv("nstorstrommen14.csv")],
[pd.read_csv("njakobshavn15.csv"),pd.read_csv("nsoutheast15.csv"),pd.read_csv("nstorstrommen15.csv")],
[pd.read_csv("njakobshavn16.csv"),pd.read_csv("nsoutheast16.csv"),pd.read_csv("nstorstrommen16.csv")]]

All_years=[pd.concat([data[0][0],data[1][0],data[2][0],data[3][0],data[4][0],data[5][0]]),
           pd.concat([data[0][1],data[1][1],data[2][1],data[3][1],data[4][1],data[5][1]]),
           pd.concat([data[0][2],data[1][2],data[2][2],data[3][2],data[4][2],data[5][2]])]
nAll_years=[pd.concat([ndata[0][0],ndata[1][0],ndata[2][0],ndata[3][0],ndata[4][0],ndata[5][0]]),
           pd.concat([ndata[0][1],ndata[1][1],ndata[2][1],ndata[3][1],ndata[4][1],ndata[5][1]]),
           pd.concat([ndata[0][2],ndata[1][2],ndata[2][2],ndata[3][2],ndata[4][2],ndata[5][2]])]

#Exploratory data analysis

def differences(data):
    differences=[]
    for i in range(len(data)):
        difference=[]
        for j in range(len(data[0])):
            difference.append(data[i][j]["Elev_Oib"]-data[i][j]["Elev_Swath"])
        differences.append(difference)
    return differences            

#test effect of badly matching lidar and radar
levene(pd.concat(All_years)["Elev_Oib"]-pd.concat(All_years)["Elev_Swath"],pd.concat(nAll_years)["Elev_Oib"]-pd.concat(nAll_years)["Elev_Swath"])
np.var(pd.concat(All_years)["Elev_Oib"]-pd.concat(All_years)["Elev_Swath"])
np.var(pd.concat(nAll_years)["Elev_Oib"]-pd.concat(nAll_years)["Elev_Swath"])

plt.hist(pd.concat(All_years)["Elev_Oib"]-pd.concat(All_years)["Elev_Swath"],normed=True,bins=100, alpha=0.5, label='All',range=[-10,10])
plt.hist(pd.concat(nAll_years)["Elev_Oib"]-pd.concat(nAll_years)["Elev_Swath"],normed=True,bins=100, alpha=0.5, label='Near',range=[-10,10])
plt.show()

All_everything = pd.concat(All_years)[["Coh_Swath","Coh_SwathOverPoca","DayInYear_Swath","Dist_SwathToPoca","Heading_Swath","LeadEdgeS_Poca","LeadEdgeW_Poca","PhaseConfidence_Swath","PhaseSSegment_Swath","PowerScaled_Swath","PowerScaled_SwathOverPoca","SampleNb_Swath","SampleNb_SwathMinusLeadEdgeS","Phase_Swath","Phase_SwathOverPoca"]]
All_everything.corrwith(pd.concat(All_years)["Elev_Oib"]-pd.concat(All_years)["Elev_Swath"],0)

#one year
data_y = differences(data)
uno=data_y[0][0]
dos=data_y[0][1]
tres=data_y[0][2]

plt.hist(uno ,normed=True,bins=100, alpha=0.5, label='jakobshavn',range=[-10,10])
plt.hist(dos, bins=100, alpha=0.5, label='southeast',range=[-10,10],normed=True)
plt.hist(tres, bins=100, alpha=0.5, label='storstrommen',range=[-10,10],normed=True)
plt.title('Distribution of the difference grouped by region',fontsize='large')
plt.legend(loc='upper right')
plt.xlabel('Difference radar and LiDAR [meter]')
plt.show()

#All years report

uno=All_years[0]["Elev_Oib"]-All_years[0]["Elev_Swath"]
dos=All_years[1]["Elev_Oib"]-All_years[1]["Elev_Swath"]
tres=All_years[2]["Elev_Oib"]-All_years[2]["Elev_Swath"]

plt.hist(uno ,normed=True,bins=100, alpha=0.5, label='jakobshavn',range=[-10,10])
plt.hist(dos, bins=100, alpha=0.5, label='southeast',range=[-10,10],normed=True)
plt.hist(tres, bins=100, alpha=0.5, label='storstrommen',range=[-10,10],normed=True)
plt.title('Distribution of the difference grouped by region',fontsize='large')
plt.legend(loc='upper right')
plt.xlabel('Difference radar and LiDAR [meter]')
plt.show()

#All years near

uno=nAll_years[0]["Elev_Oib"]-nAll_years[0]["Elev_Swath"]
dos=nAll_years[1]["Elev_Oib"]-nAll_years[1]["Elev_Swath"]
tres=nAll_years[2]["Elev_Oib"]-nAll_years[2]["Elev_Swath"]

plt.hist(uno ,normed=True,bins=100, alpha=0.5, label='jakobshavn',range=[-10,10])
plt.hist(dos, bins=100, alpha=0.5, label='southeast',range=[-10,10],normed=True)
plt.hist(tres, bins=100, alpha=0.5, label='storstrommen',range=[-10,10],normed=True)
plt.title('Distribution of the difference grouped by region',fontsize='large')
plt.legend(loc='upper right')
plt.xlabel('Difference radar and LiDAR [meter]')
plt.show()

#Through the years

years=np.array([2011,2012,2013,2014,2015,2016])
plt.plot(years,np.array([np.mean(data_y[0][0]),np.mean(data_y[1][0]),np.mean(data_y[2][0]),np.mean(data_y[3][0]),np.mean(data_y[4][0]),np.mean(data_y[5][0])]),label='jakobshavn')
plt.fill_between(years,np.array([np.mean(data_y[0][0]),np.mean(data_y[1][0]),np.mean(data_y[2][0]),np.mean(data_y[3][0]),np.mean(data_y[4][0]),np.mean(data_y[5][0])])+np.array([np.std(data_y[0][0]),np.std(data_y[1][0]),np.std(data_y[2][0]),np.std(data_y[3][0]),np.std(data_y[4][0]),np.std(data_y[5][0])]),np.array([np.mean(data_y[0][0]),np.mean(data_y[1][0]),np.mean(data_y[2][0]),np.mean(data_y[3][0]),np.mean(data_y[4][0]),np.mean(data_y[5][0])])-np.array([np.std(data_y[0][0]),np.std(data_y[1][0]),np.std(data_y[2][0]),np.std(data_y[3][0]),np.std(data_y[4][0]),np.std(data_y[5][0])]),alpha=0.5)
plt.plot(years,np.array([np.mean(data_y[0][1]),np.mean(data_y[1][1]),np.mean(data_y[2][1]),np.mean(data_y[3][1]),np.mean(data_y[4][1]),np.mean(data_y[5][1])]),label='southeast')
plt.fill_between(years,np.array([np.mean(data_y[0][1]),np.mean(data_y[1][1]),np.mean(data_y[2][1]),np.mean(data_y[3][1]),np.mean(data_y[4][1]),np.mean(data_y[5][1])])+np.array([np.std(data_y[0][1]),np.std(data_y[1][1]),np.std(data_y[2][1]),np.std(data_y[3][1]),np.std(data_y[4][1]),np.std(data_y[5][1])]),np.array([np.mean(data_y[0][1]),np.mean(data_y[1][1]),np.mean(data_y[2][1]),np.mean(data_y[3][1]),np.mean(data_y[4][1]),np.mean(data_y[5][1])])-np.array([np.std(data_y[0][1]),np.std(data_y[1][1]),np.std(data_y[2][1]),np.std(data_y[3][1]),np.std(data_y[4][1]),np.std(data_y[5][1])]),alpha=0.5)
plt.plot(years,np.array([np.mean(data_y[0][2]),np.mean(data_y[1][2]),np.mean(data_y[2][2]),np.mean(data_y[3][2]),np.mean(data_y[4][2]),np.mean(data_y[5][2])]),label='storstrommen')
plt.fill_between(years,np.array([np.mean(data_y[0][2]),np.mean(data_y[1][2]),np.mean(data_y[2][2]),np.mean(data_y[3][2]),np.mean(data_y[4][2]),np.mean(data_y[5][2])])+np.array([np.std(data_y[0][2]),np.std(data_y[1][2]),np.std(data_y[2][2]),np.std(data_y[3][2]),np.std(data_y[4][2]),np.std(data_y[5][2])]),np.array([np.mean(data_y[0][2]),np.mean(data_y[1][2]),np.mean(data_y[2][2]),np.mean(data_y[3][2]),np.mean(data_y[4][2]),np.mean(data_y[5][2])])-np.array([np.std(data_y[0][2]),np.std(data_y[1][2]),np.std(data_y[2][2]),np.std(data_y[3][2]),np.std(data_y[4][2]),np.std(data_y[5][2])]),alpha=0.5)
plt.title('Average difference per year',fontsize='large')
plt.legend(loc='upper right')
plt.xlabel('time [year]')
plt.show()

#####



ndata_y=differences(ndata)
plt.scatter(ndata[0][0]["SampleNb_SwathMinusLeadEdgeS"],ndata_y[0][0],s=0.1)
plt.scatter(data[0][0]["Dist_SwathToPoca"],data_y[0][0],s=0.1)

def findTheRelationship(ys,variable):
    plt.scatter(variable,ys,s=0.1)
    

def superfunk(model,regions,variables):
    predictions=[]
    scores=np.zeros(len(regions))
    mses=np.zeros(len(regions))
    baselines=np.zeros(len(regions))
    for i in range(len(regions)):
        y_test=regions[i]["Elev_Oib"]-regions[i]["Elev_Swath"]
        X_test=regions[i][variables]
        X_train=pd.concat([x for k,x in enumerate(regions) if k!=i])
        y_train=X_train["Elev_Oib"]-X_train["Elev_Swath"]
        #print(mean_squared_error(np.ones(len(y_test))*np.mean(y_train),y_test))
        print(mean_squared_error(np.zeros(len(y_test)),y_test))
        X_train=X_train[variables]
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        model.fit(X_train,y_train)
        prediction=model.predict(X_test)
        predictions.append(prediction)
        scores[i]=model.score(X_test,y_test)
        mses[i]=mean_squared_error(y_test, prediction)
        #baselines[i] = mean_squared_error(np.ones(len(y_test))*np.mean(y_train),y_test)
        baselines[i] = mean_squared_error(np.zeros(len(y_test)),y_test)
    Beep(1000, 1000)
    return [predictions,scores,mses,baselines]

def combyfunk(model,data,variables):
    jacob = np.zeros(6)
    stors = np.zeros(6)
    south = np.zeros(6)
    jbase = np.zeros(6)
    storsbase = np.zeros(6)
    southbase = np.zeros(6)
    for i in range(6):
        temp = superfunk(model,data[i],variables)
        jacob[i]=temp[2][0]
        jbase[i]=temp[3][0]
        stors[i]=temp[2][1]
        storsbase[i]=temp[3][1]
        south[i]=temp[2][2]
        southbase[i]=temp[3][2]
    return np.array([[np.mean(jacob),np.mean(stors),np.mean(south)],[np.mean(jbase),np.mean(storsbase),np.mean(southbase)]])

combires = combyfunk(linear_model.LinearRegression(),ndata,["Coh_Swath","Coh_SwathOverPoca","DayInYear_Swath","Dist_SwathToPoca","Heading_Swath","LeadEdgeS_Poca","LeadEdgeW_Poca","PhaseConfidence_Swath","PhaseSSegment_Swath","PowerScaled_Swath","PowerScaled_SwathOverPoca","SampleNb_Swath","SampleNb_SwathMinusLeadEdgeS","Phase_Swath","Phase_SwathOverPoca"])
combi1 =  combyfunk(MLPRegressor(hidden_layer_sizes=(15,15,15,15),alpha=10,max_iter=500),data,["Coh_Swath","Coh_SwathOverPoca","DayInYear_Swath","Dist_SwathToPoca","Heading_Swath","LeadEdgeS_Poca","LeadEdgeW_Poca","PhaseConfidence_Swath","PhaseSSegment_Swath","PowerScaled_Swath","PowerScaled_SwathOverPoca","SampleNb_Swath","SampleNb_SwathMinusLeadEdgeS","Phase_Swath","Phase_SwathOverPoca"])
combi4 =  combyfunk(RandomForestRegressor(random_state=0,n_estimators=30,min_samples_split=150),ndata,["Coh_Swath","Coh_SwathOverPoca","DayInYear_Swath","Dist_SwathToPoca","Heading_Swath","LeadEdgeS_Poca","LeadEdgeW_Poca","PhaseConfidence_Swath","PhaseSSegment_Swath","PowerScaled_Swath","PowerScaled_SwathOverPoca","SampleNb_Swath","SampleNb_SwathMinusLeadEdgeS","Phase_Swath","Phase_SwathOverPoca"])
combi3 =  combyfunk(GradientBoostingRegressor(),ndata,["Coh_Swath","Coh_SwathOverPoca","DayInYear_Swath","Dist_SwathToPoca","Heading_Swath","LeadEdgeS_Poca","LeadEdgeW_Poca","PhaseConfidence_Swath","PhaseSSegment_Swath","PowerScaled_Swath","PowerScaled_SwathOverPoca","SampleNb_Swath","SampleNb_SwathMinusLeadEdgeS","Phase_Swath","Phase_SwathOverPoca"])
combi2 =  combyfunk(GradientBoostingRegressor(),data,["Coh_Swath","Coh_SwathOverPoca","DayInYear_Swath","Dist_SwathToPoca","Heading_Swath","LeadEdgeS_Poca","LeadEdgeW_Poca","PhaseConfidence_Swath","PhaseSSegment_Swath","PowerScaled_Swath","PowerScaled_SwathOverPoca","SampleNb_Swath","SampleNb_SwathMinusLeadEdgeS","Phase_Swath","Phase_SwathOverPoca"])
combi5 =  combyfunk(svm.SVR(),data,["Coh_Swath","Coh_SwathOverPoca","DayInYear_Swath","Dist_SwathToPoca","Heading_Swath","LeadEdgeS_Poca","LeadEdgeW_Poca","PhaseConfidence_Swath","PhaseSSegment_Swath","PowerScaled_Swath","PowerScaled_SwathOverPoca","SampleNb_Swath","SampleNb_SwathMinusLeadEdgeS","Phase_Swath","Phase_SwathOverPoca"])


results1=superfunk(linear_model.LinearRegression(),ndata[0],["Coh_Swath","Coh_SwathOverPoca","DayInYear_Swath","Dist_SwathToPoca","Heading_Swath","LeadEdgeS_Poca","LeadEdgeW_Poca","PhaseConfidence_Swath","PhaseSSegment_Swath","PowerScaled_Swath","PowerScaled_SwathOverPoca","SampleNb_Swath","SampleNb_SwathMinusLeadEdgeS","Phase_Swath","Phase_SwathOverPoca"])
results2=superfunk(linear_model.RANSACRegressor(),ndata[0],["Coh_Swath","Coh_SwathOverPoca","DayInYear_Swath","Dist_SwathToPoca","Heading_Swath","LeadEdgeS_Poca","LeadEdgeW_Poca","PhaseConfidence_Swath","PhaseSSegment_Swath","PowerScaled_Swath","PowerScaled_SwathOverPoca","SampleNb_Swath","SampleNb_SwathMinusLeadEdgeS","Phase_Swath","Phase_SwathOverPoca"])
results3=superfunk(linear_model.BayesianRidge(),ndata[0],["Coh_Swath","Coh_SwathOverPoca","DayInYear_Swath","Dist_SwathToPoca","Heading_Swath","LeadEdgeS_Poca","LeadEdgeW_Poca","PhaseConfidence_Swath","PhaseSSegment_Swath","PowerScaled_Swath","PowerScaled_SwathOverPoca","SampleNb_Swath","SampleNb_SwathMinusLeadEdgeS","Phase_Swath","Phase_SwathOverPoca"])
results4=superfunk(linear_model.PassiveAggressiveRegressor(),ndata[0],["Coh_Swath","Coh_SwathOverPoca","DayInYear_Swath","Dist_SwathToPoca","Heading_Swath","LeadEdgeS_Poca","LeadEdgeW_Poca","PhaseConfidence_Swath","PhaseSSegment_Swath","PowerScaled_Swath","PowerScaled_SwathOverPoca","SampleNb_Swath","SampleNb_SwathMinusLeadEdgeS","Phase_Swath","Phase_SwathOverPoca"])
results5=superfunk(linear_model.SGDRegressor(),ndata[0],["Coh_Swath","Coh_SwathOverPoca","DayInYear_Swath","Dist_SwathToPoca","Heading_Swath","LeadEdgeS_Poca","LeadEdgeW_Poca","PhaseConfidence_Swath","PhaseSSegment_Swath","PowerScaled_Swath","PowerScaled_SwathOverPoca","SampleNb_Swath","SampleNb_SwathMinusLeadEdgeS","Phase_Swath","Phase_SwathOverPoca"])
results6=superfunk(svm.SVR(),ndata[0],["Coh_Swath","Coh_SwathOverPoca","DayInYear_Swath","Dist_SwathToPoca","Heading_Swath","LeadEdgeS_Poca","LeadEdgeW_Poca","PhaseConfidence_Swath","PhaseSSegment_Swath","PowerScaled_Swath","PowerScaled_SwathOverPoca","SampleNb_Swath","SampleNb_SwathMinusLeadEdgeS","Phase_Swath","Phase_SwathOverPoca"])
results6b=superfunk(svm.SVR(),data[0],["Coh_Swath","Coh_SwathOverPoca","DayInYear_Swath","Dist_SwathToPoca","Heading_Swath","LeadEdgeS_Poca","LeadEdgeW_Poca","PhaseConfidence_Swath","PhaseSSegment_Swath","PowerScaled_Swath","PowerScaled_SwathOverPoca","SampleNb_Swath","SampleNb_SwathMinusLeadEdgeS","Phase_Swath","Phase_SwathOverPoca"])

results7=superfunk(MLPRegressor(hidden_layer_sizes=(15,15,15,15),alpha=100,max_iter=500,activation="relu"),ndata[0],["Coh_Swath","Coh_SwathOverPoca","DayInYear_Swath","Dist_SwathToPoca","Heading_Swath","LeadEdgeS_Poca","LeadEdgeW_Poca","PhaseConfidence_Swath","PhaseSSegment_Swath","PowerScaled_Swath","PowerScaled_SwathOverPoca","SampleNb_Swath","SampleNb_SwathMinusLeadEdgeS","Phase_Swath","Phase_SwathOverPoca"])

results70=superfunk(MLPRegressor(hidden_layer_sizes=(15,15,15,15),alpha=10,max_iter=500,activation="relu"),All_years,["Coh_Swath","Coh_SwathOverPoca","DayInYear_Swath","Dist_SwathToPoca","Heading_Swath","LeadEdgeS_Poca","LeadEdgeW_Poca","PhaseConfidence_Swath","PhaseSSegment_Swath","PowerScaled_Swath","PowerScaled_SwathOverPoca","SampleNb_Swath","SampleNb_SwathMinusLeadEdgeS","Phase_Swath","Phase_SwathOverPoca"])

results102=superfunk(RandomForestRegressor(random_state=0,n_estimators=30,min_samples_split=100),All_years,["Coh_Swath","Coh_SwathOverPoca","DayInYear_Swath","Dist_SwathToPoca","Heading_Swath","LeadEdgeS_Poca","LeadEdgeW_Poca","PhaseConfidence_Swath","PhaseSSegment_Swath","PowerScaled_Swath","PowerScaled_SwathOverPoca","SampleNb_Swath","SampleNb_SwathMinusLeadEdgeS","Phase_Swath","Phase_SwathOverPoca"])

results101=superfunk(GradientBoostingRegressor(),All_years,["Coh_Swath","Coh_SwathOverPoca","DayInYear_Swath","Dist_SwathToPoca","Heading_Swath","LeadEdgeS_Poca","LeadEdgeW_Poca","PhaseConfidence_Swath","PhaseSSegment_Swath","PowerScaled_Swath","PowerScaled_SwathOverPoca","SampleNb_Swath","SampleNb_SwathMinusLeadEdgeS","Phase_Swath","Phase_SwathOverPoca"])

results100=superfunk(GradientBoostingRegressor(),[ndata[3][0],ndata[3][1]],["Coh_Swath","Coh_SwathOverPoca","DayInYear_Swath","Dist_SwathToPoca","Heading_Swath","LeadEdgeS_Poca","LeadEdgeW_Poca","PhaseConfidence_Swath","PhaseSSegment_Swath","PowerScaled_Swath","PowerScaled_SwathOverPoca","SampleNb_Swath","SampleNb_SwathMinusLeadEdgeS","Phase_Swath","Phase_SwathOverPoca"])

results10=superfunk(GradientBoostingRegressor(),ndata[0],["Coh_Swath","Coh_SwathOverPoca","DayInYear_Swath","Dist_SwathToPoca","Heading_Swath","LeadEdgeS_Poca","LeadEdgeW_Poca","PhaseConfidence_Swath","PhaseSSegment_Swath","PowerScaled_Swath","PowerScaled_SwathOverPoca","SampleNb_Swath","SampleNb_SwathMinusLeadEdgeS","Phase_Swath","Phase_SwathOverPoca"])
results11=superfunk(GradientBoostingRegressor(),ndata[2],["Coh_Swath","Coh_SwathOverPoca","DayInYear_Swath","Dist_SwathToPoca","Heading_Swath","LeadEdgeS_Poca","LeadEdgeW_Poca","PhaseConfidence_Swath","PhaseSSegment_Swath","PowerScaled_Swath","PowerScaled_SwathOverPoca","SampleNb_Swath","SampleNb_SwathMinusLeadEdgeS","Phase_Swath","Phase_SwathOverPoca"])
results12=superfunk(GradientBoostingRegressor(),data[1],["Coh_Swath","Coh_SwathOverPoca","DayInYear_Swath","Dist_SwathToPoca","Heading_Swath","LeadEdgeS_Poca","LeadEdgeW_Poca","PhaseConfidence_Swath","PhaseSSegment_Swath","PowerScaled_Swath","PowerScaled_SwathOverPoca","SampleNb_Swath","SampleNb_SwathMinusLeadEdgeS","Phase_Swath","Phase_SwathOverPoca"])
moddel=GradientBoostingRegressor()
results13=superfunk(moddel,ntemporal,["Coh_Swath","Coh_SwathOverPoca","DayInYear_Swath","Dist_SwathToPoca","Heading_Swath","LeadEdgeS_Poca","LeadEdgeW_Poca","PhaseConfidence_Swath","PhaseSSegment_Swath","PowerScaled_Swath","PowerScaled_SwathOverPoca","SampleNb_Swath","SampleNb_SwathMinusLeadEdgeS","Phase_Swath","Phase_SwathOverPoca"])



results2=superfunk(RandomForestRegressor(random_state=0,n_estimators=30,min_samples_split=150),ndata[0],["Coh_Swath","Coh_SwathOverPoca","DayInYear_Swath","Dist_SwathToPoca","Heading_Swath","LeadEdgeS_Poca","LeadEdgeW_Poca","PhaseConfidence_Swath","PhaseSSegment_Swath","PowerScaled_Swath","PowerScaled_SwathOverPoca","SampleNb_Swath","SampleNb_SwathMinusLeadEdgeS","Phase_Swath","Phase_SwathOverPoca"])
results12=superfunk(MLPRegressor(hidden_layer_sizes=(15,15,15,15),max_iter=500,activation="relu"),data[1],["Coh_Swath","Coh_SwathOverPoca","DayInYear_Swath","Dist_SwathToPoca","Heading_Swath","LeadEdgeS_Poca","LeadEdgeW_Poca","PhaseConfidence_Swath","PhaseSSegment_Swath","PowerScaled_Swath","PowerScaled_SwathOverPoca","SampleNb_Swath","SampleNb_SwathMinusLeadEdgeS","Phase_Swath","Phase_SwathOverPoca"])
nresults3=superfunk(KNeighborsRegressor(),nAll_years[0],["Coh_Swath","Coh_SwathOverPoca","DayInYear_Swath","Dist_SwathToPoca","Heading_Swath","LeadEdgeS_Poca","LeadEdgeW_Poca","PhaseConfidence_Swath","PhaseSSegment_Swath","PowerScaled_Swath","PowerScaled_SwathOverPoca","SampleNb_Swath","SampleNb_SwathMinusLeadEdgeS","Phase_Swath","Phase_SwathOverPoca"])

jtemporal=[data[0][0],data[1][0],data[2][0],data[3][0],data[4][0],data[5][0]]
njtemporal=[ndata[0][0],ndata[1][0],ndata[2][0],ndata[3][0],ndata[4][0],ndata[5][0]]

stemporal=[data[0][1],data[1][1],data[2][1],data[3][1],data[4][1],data[5][1]]
nstemporal=[ndata[0][1],ndata[1][1],ndata[2][1],ndata[3][1],ndata[4][1],ndata[5][1]]

sstemporal=[data[0][2],data[1][2],data[2][2],data[3][2],data[4][2],data[5][2]]
nsstemporal=[ndata[0][2],ndata[1][2],ndata[2][2],ndata[3][2],ndata[4][2],ndata[5][2]]


results221=superfunk(GradientBoostingRegressor(),jtemporal,["Coh_Swath","Coh_SwathOverPoca","DayInYear_Swath","Dist_SwathToPoca","Heading_Swath","LeadEdgeS_Poca","LeadEdgeW_Poca","PhaseConfidence_Swath","PhaseSSegment_Swath","PowerScaled_Swath","PowerScaled_SwathOverPoca","SampleNb_Swath","SampleNb_SwathMinusLeadEdgeS","Phase_Swath","Phase_SwathOverPoca"])
results222=superfunk(GradientBoostingRegressor(),stemporal,["Coh_Swath","Coh_SwathOverPoca","DayInYear_Swath","Dist_SwathToPoca","Heading_Swath","LeadEdgeS_Poca","LeadEdgeW_Poca","PhaseConfidence_Swath","PhaseSSegment_Swath","PowerScaled_Swath","PowerScaled_SwathOverPoca","SampleNb_Swath","SampleNb_SwathMinusLeadEdgeS","Phase_Swath","Phase_SwathOverPoca"])
results223=superfunk(GradientBoostingRegressor(),sstemporal,["Coh_Swath","Coh_SwathOverPoca","DayInYear_Swath","Dist_SwathToPoca","Heading_Swath","LeadEdgeS_Poca","LeadEdgeW_Poca","PhaseConfidence_Swath","PhaseSSegment_Swath","PowerScaled_Swath","PowerScaled_SwathOverPoca","SampleNb_Swath","SampleNb_SwathMinusLeadEdgeS","Phase_Swath","Phase_SwathOverPoca"])

results231=superfunk(RandomForestRegressor(random_state=0,n_estimators=30,min_samples_split=150),jtemporal,["Coh_Swath","Coh_SwathOverPoca","DayInYear_Swath","Dist_SwathToPoca","Heading_Swath","LeadEdgeS_Poca","LeadEdgeW_Poca","PhaseConfidence_Swath","PhaseSSegment_Swath","PowerScaled_Swath","PowerScaled_SwathOverPoca","SampleNb_Swath","SampleNb_SwathMinusLeadEdgeS","Phase_Swath","Phase_SwathOverPoca"])
results232=superfunk(RandomForestRegressor(random_state=0,n_estimators=30,min_samples_split=150),stemporal,["Coh_Swath","Coh_SwathOverPoca","DayInYear_Swath","Dist_SwathToPoca","Heading_Swath","LeadEdgeS_Poca","LeadEdgeW_Poca","PhaseConfidence_Swath","PhaseSSegment_Swath","PowerScaled_Swath","PowerScaled_SwathOverPoca","SampleNb_Swath","SampleNb_SwathMinusLeadEdgeS","Phase_Swath","Phase_SwathOverPoca"])
results233=superfunk(RandomForestRegressor(random_state=0,n_estimators=30,min_samples_split=150),sstemporal,["Coh_Swath","Coh_SwathOverPoca","DayInYear_Swath","Dist_SwathToPoca","Heading_Swath","LeadEdgeS_Poca","LeadEdgeW_Poca","PhaseConfidence_Swath","PhaseSSegment_Swath","PowerScaled_Swath","PowerScaled_SwathOverPoca","SampleNb_Swath","SampleNb_SwathMinusLeadEdgeS","Phase_Swath","Phase_SwathOverPoca"])

results331=superfunk(MLPRegressor(hidden_layer_sizes=(15,15,15,15),max_iter=500,activation="relu"),jtemporal,["Coh_Swath","Coh_SwathOverPoca","DayInYear_Swath","Dist_SwathToPoca","Heading_Swath","LeadEdgeS_Poca","LeadEdgeW_Poca","PhaseConfidence_Swath","PhaseSSegment_Swath","PowerScaled_Swath","PowerScaled_SwathOverPoca","SampleNb_Swath","SampleNb_SwathMinusLeadEdgeS","Phase_Swath","Phase_SwathOverPoca"])
results332=superfunk(MLPRegressor(hidden_layer_sizes=(15,15,15,15),max_iter=500,activation="relu"),stemporal,["Coh_Swath","Coh_SwathOverPoca","DayInYear_Swath","Dist_SwathToPoca","Heading_Swath","LeadEdgeS_Poca","LeadEdgeW_Poca","PhaseConfidence_Swath","PhaseSSegment_Swath","PowerScaled_Swath","PowerScaled_SwathOverPoca","SampleNb_Swath","SampleNb_SwathMinusLeadEdgeS","Phase_Swath","Phase_SwathOverPoca"])
results333=superfunk(MLPRegressor(hidden_layer_sizes=(15,15,15,15),max_iter=500,activation="relu"),sstemporal,["Coh_Swath","Coh_SwathOverPoca","DayInYear_Swath","Dist_SwathToPoca","Heading_Swath","LeadEdgeS_Poca","LeadEdgeW_Poca","PhaseConfidence_Swath","PhaseSSegment_Swath","PowerScaled_Swath","PowerScaled_SwathOverPoca","SampleNb_Swath","SampleNb_SwathMinusLeadEdgeS","Phase_Swath","Phase_SwathOverPoca"])


results221=superfunk(GradientBoostingRegressor(),njtemporal,["Coh_Swath","Coh_SwathOverPoca","DayInYear_Swath","Dist_SwathToPoca","Heading_Swath","LeadEdgeS_Poca","LeadEdgeW_Poca","PhaseConfidence_Swath","PhaseSSegment_Swath","PowerScaled_Swath","PowerScaled_SwathOverPoca","SampleNb_Swath","SampleNb_SwathMinusLeadEdgeS","Phase_Swath","Phase_SwathOverPoca"])
results222=superfunk(GradientBoostingRegressor(),nstemporal,["Coh_Swath","Coh_SwathOverPoca","DayInYear_Swath","Dist_SwathToPoca","Heading_Swath","LeadEdgeS_Poca","LeadEdgeW_Poca","PhaseConfidence_Swath","PhaseSSegment_Swath","PowerScaled_Swath","PowerScaled_SwathOverPoca","SampleNb_Swath","SampleNb_SwathMinusLeadEdgeS","Phase_Swath","Phase_SwathOverPoca"])
results223=superfunk(GradientBoostingRegressor(),nsstemporal,["Coh_Swath","Coh_SwathOverPoca","DayInYear_Swath","Dist_SwathToPoca","Heading_Swath","LeadEdgeS_Poca","LeadEdgeW_Poca","PhaseConfidence_Swath","PhaseSSegment_Swath","PowerScaled_Swath","PowerScaled_SwathOverPoca","SampleNb_Swath","SampleNb_SwathMinusLeadEdgeS","Phase_Swath","Phase_SwathOverPoca"])

results231=superfunk(RandomForestRegressor(random_state=0,n_estimators=30,min_samples_split=150),njtemporal,["Coh_Swath","Coh_SwathOverPoca","DayInYear_Swath","Dist_SwathToPoca","Heading_Swath","LeadEdgeS_Poca","LeadEdgeW_Poca","PhaseConfidence_Swath","PhaseSSegment_Swath","PowerScaled_Swath","PowerScaled_SwathOverPoca","SampleNb_Swath","SampleNb_SwathMinusLeadEdgeS","Phase_Swath","Phase_SwathOverPoca"])
results232=superfunk(RandomForestRegressor(random_state=0,n_estimators=30,min_samples_split=150),nstemporal,["Coh_Swath","Coh_SwathOverPoca","DayInYear_Swath","Dist_SwathToPoca","Heading_Swath","LeadEdgeS_Poca","LeadEdgeW_Poca","PhaseConfidence_Swath","PhaseSSegment_Swath","PowerScaled_Swath","PowerScaled_SwathOverPoca","SampleNb_Swath","SampleNb_SwathMinusLeadEdgeS","Phase_Swath","Phase_SwathOverPoca"])
results233=superfunk(RandomForestRegressor(random_state=0,n_estimators=30,min_samples_split=150),nsstemporal,["Coh_Swath","Coh_SwathOverPoca","DayInYear_Swath","Dist_SwathToPoca","Heading_Swath","LeadEdgeS_Poca","LeadEdgeW_Poca","PhaseConfidence_Swath","PhaseSSegment_Swath","PowerScaled_Swath","PowerScaled_SwathOverPoca","SampleNb_Swath","SampleNb_SwathMinusLeadEdgeS","Phase_Swath","Phase_SwathOverPoca"])

results331=superfunk(MLPRegressor(hidden_layer_sizes=(15,15,15,15),max_iter=500,activation="relu"),njtemporal,["Coh_Swath","Coh_SwathOverPoca","DayInYear_Swath","Dist_SwathToPoca","Heading_Swath","LeadEdgeS_Poca","LeadEdgeW_Poca","PhaseConfidence_Swath","PhaseSSegment_Swath","PowerScaled_Swath","PowerScaled_SwathOverPoca","SampleNb_Swath","SampleNb_SwathMinusLeadEdgeS","Phase_Swath","Phase_SwathOverPoca"])
results332=superfunk(MLPRegressor(hidden_layer_sizes=(15,15,15,15),max_iter=500,activation="relu"),nstemporal,["Coh_Swath","Coh_SwathOverPoca","DayInYear_Swath","Dist_SwathToPoca","Heading_Swath","LeadEdgeS_Poca","LeadEdgeW_Poca","PhaseConfidence_Swath","PhaseSSegment_Swath","PowerScaled_Swath","PowerScaled_SwathOverPoca","SampleNb_Swath","SampleNb_SwathMinusLeadEdgeS","Phase_Swath","Phase_SwathOverPoca"])
results333=superfunk(MLPRegressor(hidden_layer_sizes=(15,15,15,15),max_iter=500,activation="relu"),nsstemporal,["Coh_Swath","Coh_SwathOverPoca","DayInYear_Swath","Dist_SwathToPoca","Heading_Swath","LeadEdgeS_Poca","LeadEdgeW_Poca","PhaseConfidence_Swath","PhaseSSegment_Swath","PowerScaled_Swath","PowerScaled_SwathOverPoca","SampleNb_Swath","SampleNb_SwathMinusLeadEdgeS","Phase_Swath","Phase_SwathOverPoca"])



#remove coherence
#take tha mean of all
# test all vs near for topography dependence
