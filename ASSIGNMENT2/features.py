# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np



def extract(data) :
    #feature1.1 Velocity
    feature_matrix=pd.DataFrame()
    meal=data
    max_slope_diff_feature=[]
    for k in range(len(meal)) :
        X=[29-i for i in range(0,30)]
        Y=[]
        Y=meal.iloc[k,:].values
        slope=[]
        for i in range(len(X)-1) :
            m=(Y[i+1] - Y[i])/(X[i+1]-X[i])
            slope.append(m)
        #plt.scatter(X[0:30],slope)
        slope_diff=[]
        window_size=3
        for i in range(len(slope)-window_size) :
            mn=min(slope[i:i+window_size])
            mx=max(slope[i:i+window_size])
            slope_diff.append([mx-mn,X[i-1],X[i+window_size]]) 
        max_slope_diff=max(slope_diff)[0]
        max_slope_diff_feature.append(max_slope_diff)
    feature_matrix["max_slope_diff_feature"] = max_slope_diff_feature
    #feature1.2 Zero Crossings
    max_neg_slope_diff=[]
    zero_crossing_count=[]
    for k in range(len(meal)) :
        X=[29-i for i in range(0,30)]
        Y=[]
        Y=meal.iloc[k,:].values
        slope=[]
        for j in range(len(X)-1) :
                
            m=(Y[j+1]-Y[j])/(X[j+1]-X[j])
            slope.append(m)
        neg_slope=[]
        z_count=0
        for i in range(len(slope)-2) :
            if slope[i]*slope[i+1] < 0  : 
                neg_slope.append([slope[i+1]-slope[i],X[i+1]])
                z_count=z_count+1;
               
                
        zero_crossing_count.append(z_count)                           
    #    print(k,neg_slope)
        max_neg_slope_diff.append(max(neg_slope,default=0))    
        #10    go and check for 6 th value m[i] <0 and m[i] >0
    max_zerocross=[]
    for i in range(len(max_neg_slope_diff)) :    
    #    print(max_neg_slope_diff[i][0])
        if max_neg_slope_diff[i] == 0 :
            max_zerocross.append(0)
    #        print(i,i)
    
            
        else :        
             max_zerocross.append(max_neg_slope_diff[i][0])
    #         print(i)
    
    feature_matrix['max_zero_crossing'] = max_zerocross 
    feature_matrix['zero_crossing_count'] = zero_crossing_count  
    
    
    from scipy.stats import kurtosis
    from scipy.stats import skew 
    skewness=[]
    kurtosis_of_data=[]
    for k in range(len(meal)) :
        X=[]
        Y=[]
        X=[29-i for i in range(0,30)]
        Y=meal.iloc[k,:].values
        skewness.append(skew(Y))
        kurtosis_of_data.append(kurtosis(Y))


    feature_matrix['skewness_of_data'] = skewness
    feature_matrix['kurtosis_of_data'] = kurtosis_of_data
    
    import scipy.stats 
    min_cgm=[]
    max_cgm=[]
    diff_min_max=[]
    entropy_data=[]
    for k in range(len(meal)) :
        Y=meal.iloc[k,:].values
        min_cgm.append(min(Y))
        max_cgm.append(max(Y))
        diff_min_max.append(max(Y)-min(Y))
        entropy_data.append(scipy.stats.entropy(Y))
        
    feature_matrix['entropy']= entropy_data  
    feature_matrix['min_cgm']= min_cgm
    feature_matrix['max_cgm']= max_cgm  
    feature_matrix['diff_min_max']=diff_min_max
    
    
    return feature_matrix

#print("AFTER FUNCTION")
