

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 17:34:00 2020

@author: vikky
"""

import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
CGMDatenumLunchPat2=pd.read_csv("CGMDatenumLunchPat2.csv") 
CGMSeriesLunchPat2=pd.read_csv("CGMSeriesLunchPat2.csv") 
CGMDatenumLunchPat2=CGMDatenumLunchPat2.iloc[:,0:30]
CGMSeriesLunchPat2=CGMSeriesLunchPat2.iloc[:,0:30]
l=len(CGMSeriesLunchPat2)
drop_rows=[]
for k in range(l-1) :
    if(CGMSeriesLunchPat2.iloc[k,:].isnull().sum()>8) :
        drop_rows.append(k)    
CGMDatenumLunchPat2 =CGMDatenumLunchPat2.drop(drop_rows)
CGMSeriesLunchPat2 =CGMSeriesLunchPat2.drop(drop_rows)


##1 drop last column for nan, RUN only once 
#CGMDatenumLunchPat2 = CGMDatenumLunchPat2.drop(columns=CGMDatenumLunchPat2.columns[-1])
#CGMSeriesLunchPat2 =CGMSeriesLunchPat2.drop(columns=CGMSeriesLunchPat2.columns[-1])
###PREPROCESSING
for i in range(len(CGMSeriesLunchPat2))    :    
        X=[]
        Y=[]
        X=CGMDatenumLunchPat2.iloc[i,:].values
        Y=CGMSeriesLunchPat2.iloc[i,:].values
        ##################                            
        ####################curve fitting after removing respective x when y is nan
        
        y_n = Y[np.logical_not(np.isnan(Y))]
        x_n = X[np.logical_not(np.isnan(Y))]        
    #    plt.plot(X,Y)
    #    plt.show()                   
        from numpy.polynomial import Polynomial as P
    #    import numpy as np
        x=np.array(x_n)
        y=np.array(y_n)
        p = P.fit(x, y, 5)
    
    #CGMSeriesLunchPat1[6,:].replace(to_replace = np.nan, value = 5)  ##write exception
    #replacing nans with respective p(X)   
        Y[np.isnan(Y)] = p(X[np.isnan(Y)])
    #    plt.figure()
    #    plt.plot(X,Y)

 ###############################################################
CGMDatenumLunchPat2.to_csv("preprocessed_CGMDatenumLunchPat2.csv",index=False,header=False)
CGMSeriesLunchPat2.to_csv("preprocessed_CGMSeriesLunchPat2.csv",index=False,header=False)
 
# 
 
#Draw graph for one sample
#plt.cla()
#x_new = np.linspace(x[0], x[len(x_n)-1], 50)
##plt.plot(x_new, x_new + 0, linestyle='solid')      
#plt.scatter(x_new,p(x_new),color='y',label='polynomial curve')
#plt.plot(X,Y,label='actual curve')
#plt.show()
#plt.legend()
#    
#    

#############################
### declaring feature MATRIX
feature_matrix=pd.DataFrame() ; 
###################################################################################### 
#### feature1.1 Velocity
max_slope_diff_feature=[]
for k in range(len(CGMSeriesLunchPat2)) :
    X=[]
    Y=[]
    X=CGMDatenumLunchPat2.iloc[k,:].values
    Y=CGMSeriesLunchPat2.iloc[k,:].values 
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
   
#    print(max(slope_diff)[0])

feature_matrix["max_slope_diff_feature"] = max_slope_diff_feature ##1st feature appended


### for graphh only
#plt.cla()
#plt.title('CGM velocity Shift')
#plt.xlabel('Time ')
#plt.ylabel('CGM')
#plt.axvline(max(slope_diff)[1],color='r',label="interval")
#plt.axvline(max(slope_diff)[2],color='r')
#plt.plot(X,Y)
#
#plt.scatter(X,Y)

################################ Code to draw slope vs time with axis in the middle
import numpy as np
import matplotlib.pyplot as plt


fig = plt.figure()

x = X
y_graph = slope

ax = plt.gca()
ax.scatter(x[1:len(x)],y_graph)
ax.grid(True)
#ax.spines['left'].set_position('zero')
#ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_color('none')
plt.show()



###############
#feature1.2 Zero Crossings
max_neg_slope_diff=[]
zero_crossing_count=[]
for k in range(len(CGMSeriesLunchPat2)) :
    X=[]
    Y=[]
    X=CGMDatenumLunchPat2.iloc[k,:].values
    Y=CGMSeriesLunchPat2.iloc[k,:].values 
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
 

### for graph only  
#plt.cla()      
#for i in range(len(neg_slope)) :       
#    plt.axvline(neg_slope[i][1],color='g',label='all_zero_crossings')
#    plt.plot(X,Y)
#    plt.axvline(max(neg_slope)[1],color='b',label='max_zero_crossing')
#    plt.legend()
#
#   
#


######################

#feature2
######from FFT documentation


from scipy.fftpack import fft
 
def get_fft_values(y_values, T, N, f_s):
    f_values = np.linspace(0.0, 1.0/(2.0*T), N//2)
    fft_values_ = fft(y_values)
    #print(fft_values_)
    fft_values = 2.0/N * np.abs(fft_values_[0:N//2])
    return f_values, fft_values
max_amp1=[]
max_amp2=[]
max_f1=[]
max_f2=[]
diff_peaks_fft_freq=[]


for k in range(len(CGMSeriesLunchPat2)) :
    X=[]
    Y=[]
    X=CGMDatenumLunchPat2.iloc[k,:].values
    Y=CGMSeriesLunchPat2.iloc[k,:].values  
    t_n = X[0] - X[1]
    N = len(X)
    T = t_n / N
    f_s = 1/T 
    f_values=[]
    fft_values=[]
    f_values, fft_values = get_fft_values(np.array(Y[::-1]), T, N, f_s)
#    print(1)
    ##finding the maximum and minimum peaks
    # Find the maximum y value 
    dup_f=[]
    dup_a=[]
    
    dup_f=f_values
    dup_a=fft_values
    dup_a=sorted(dup_a)
    a1=dup_a[-2]
    a2=dup_a[-3]
    f1=f_values[fft_values.tolist().index(a1)]
    f2=f_values[fft_values.tolist().index(a2)]
    max_amp1.append(a1)
    max_amp2.append(a2)
    max_f1.append(f1)
    max_f2.append(f2)  
    diff_peaks_fft_freq.append(abs(f1-f2))

#    print(a1,f1,a2,f2)

feature_matrix['fft_MAX_AMP1'] = max_amp1
feature_matrix['fft_MAX_AMP2'] = max_amp2
feature_matrix['fft_freq_at_AMP1'] =max_f1
feature_matrix['fft_freq_at_AMP2'] =max_f2





#### for graph
k=4
X=CGMDatenumLunchPat2.iloc[k,:].values
Y=CGMSeriesLunchPat2.iloc[k,:].values  
t_n = X[0] - X[1]
N = len(X)
T = t_n / N
f_s = 1/T 
f_values, fft_values = get_fft_values(np.array(Y[::-1]), T, N, f_s)
plt.cla()
plt.plot(f_values, fft_values, linestyle='-', color='blue')
plt.xlabel('Frequency [Hz]', fontsize=16)
plt.ylabel('Amplitude', fontsize=16)
plt.title("Frequency domain of the signal", fontsize=16)
plt.show()
###############################################
from scipy.signal import welch

def get_psd_values(y_values, T, N, f_s):
    f_values, psd_values = welch(y_values, fs=f_s)
    return f_values, psd_values

max_power_psd=[]
freq_max_power_psd=[]
max_psd1=[]
max_psd2=[]
max_pf1=[]
max_pf2=[]
diff_peaks_fft_pfreq=[]


for k in range(len(CGMSeriesLunchPat2)) :
    X=[]
    Y=[]
    X=CGMDatenumLunchPat2.iloc[k,:].values
    Y=CGMSeriesLunchPat2.iloc[k,:].values  
    t_n = X[0] - X[1]
    N = len(X)
    T = t_n / N
    f_s = 1/T 
    f_values, psd_values = get_psd_values(np.array(Y[::-1]), T, N, f_s)
    psd_values.sort()
    max_power_psd.append(psd_values[-2])
    dup_pf=[]
    dup_pa=[]
    
    dup_pf=f_values
    dup_pa=psd_values
    dup_pa=sorted(dup_pa)
    a1=dup_pa[-2]
    a2=dup_pa[-3]
    f1=f_values[psd_values.tolist().index(a1)]
    f2=f_values[psd_values.tolist().index(a2)]
    max_psd1.append(a1)
    max_psd2.append(a2)
    max_pf1.append(f1)
    max_pf2.append(f2)
    diff_peaks_fft_pfreq.append(abs(f1-f2))
    
    
feature_matrix['MAX_1_Power_PSD']=  max_psd1
feature_matrix['MAX_2_Power_PSD']=  max_psd2
feature_matrix['Freq_MAX_1_Power_PSD']=  max_pf1
feature_matrix['Freq_MAX_1_Power_PSD']=  max_pf2
feature_matrix['Bandwidth_at_peaks']=  diff_peaks_fft_pfreq

  
#################################
###Graph for PSD for Kth row
k=3# change k value
X=CGMDatenumLunchPat2.iloc[k,:].values
Y=CGMSeriesLunchPat2.iloc[k,:].values  
t_n = X[0] - X[1]
N = len(X)
T = t_n / N
f_s = 1/T 
f_values, psd_values = get_psd_values(np.array(Y[::-1]), T, N, f_s)

  
plt.cla()
plt.plot(f_values, psd_values, linestyle='-', color='blue')
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [V**2 / Hz]')
plt.show()






#############################################
##https://docs.scipy.org/doc/numpy/reference/generated/numpy.polynomial.polynomial.Polynomial.fit.html
    #feature4 only after preprocssing doesnt work for nan values 

from numpy.polynomial import Polynomial as P
import numpy as np
c0=[]
c1=[]
c2=[]
c3=[]
c4=[]
for k in range(len(CGMSeriesLunchPat2)) :
    X=[]
    Y=[]
    X=CGMDatenumLunchPat2.iloc[k,:].values
    Y=CGMSeriesLunchPat2.iloc[k,:].values
    x=np.array(X)
    y=np.array(Y)   
    p = P.fit(x, y, 5)
    c0.append(p.coef[0])
    c1.append(p.coef[1])
    c2.append(p.coef[2])
    c3.append(p.coef[3])
    c4.append(p.coef[4])

feature_matrix['coefficient_0']=c0
feature_matrix['coefficient_1']=c1
feature_matrix['coefficient_2']=c2
feature_matrix['coefficient_3']=c3
feature_matrix['coefficient_4']=c4

#############for graph only declare X and Y
plt.cla()
x_new = np.linspace(x[0], x[len(X)-1], 50)
#y_new = f(x_new)
plt.scatter(x_new,p(x_new),color='y',label='polynomial curve')
plt.plot(X,Y,label='actual curve')
plt.title(p.coef)
plt.legend()
#############################################



import numpy as np  
import pylab as p  
from scipy.stats import kurtosis
from scipy.stats import skew 
skewness=[]
kurtosis_of_data=[]
for k in range(len(CGMSeriesLunchPat2)) :
    X=[]
    Y=[]
    X=CGMDatenumLunchPat2.iloc[k,:].values
    Y=CGMSeriesLunchPat2.iloc[k,:].values
    skewness.append(skew(Y))
    kurtosis_of_data.append(kurtosis(Y))

    
    
feature_matrix['skewness_of_data'] = skewness
feature_matrix['kurtosis_of_data'] = kurtosis_of_data


for k in range(len(CGMSeriesLunchPat2)-20) :
    Y=[]
    Y=CGMSeriesLunchPat2.iloc[k,:].values 
    plt.title("CGM of 15 VS ON SAME TIME INTERVAL TO SHOW SKEWNESS",fontsize=25)
    plt.xlabel("TIME",fontsize=25)
    plt.ylabel("CGM of 15 MEALS",fontsize=25)
    plt.plot(X,Y)

##################Mean as a feature
mean_1=[]
mean_2=[]
for k in range(len(CGMSeriesLunchPat2)) :
    X=[]
    Y=[]
    X=CGMDatenumLunchPat2.iloc[k,:].values
    Y=CGMSeriesLunchPat2.iloc[k,:].values
    m1=np.mean(Y[0:int(len(Y)/2)])
    m2=np.mean(Y[int(len(Y)/2):-1])
    mean_1.append(m1)
    mean_2.append(m2)
    
feature_matrix['mean_1'] = mean_1
feature_matrix['mean_2'] = mean_2
########################################

plt.cla()
plt.xlabel("Time",fontsize=30)
plt.ylabel("CGM",fontsize=30)
plt.plot(X,Y,color="y")
plt.axhline(mean_1[k],color="r")
plt.axhline(mean_2[k],color="g")

##########################
feature_matrix.to_csv("FEATURE_MATRIX.CSV")

###############################PCA
import numpy as np
from sklearn.preprocessing import StandardScaler
x_std = StandardScaler().fit_transform(feature_matrix)
features = x_std.T 
covariance_matrix = np.cov(features)
#print(covariance_matrix)
eig_vals, eig_vecs = np.linalg.eig(covariance_matrix)

PCA_DATA=pd.DataFrame()

projected_X_1= x_std.dot(eig_vecs.T[0])
projected_X_2= x_std.dot(eig_vecs.T[1])
projected_X_3= x_std.dot(eig_vecs.T[2])
projected_X_4= x_std.dot(eig_vecs.T[3])
projected_X_5= x_std.dot(eig_vecs.T[4])
projected_X_6= x_std.dot(eig_vecs.T[5])
projected_X_7= x_std.dot(eig_vecs.T[6])
projected_X_8= x_std.dot(eig_vecs.T[7])
projected_X_9= x_std.dot(eig_vecs.T[8])
projected_X_10= x_std.dot(eig_vecs.T[9])
projected_X_11= x_std.dot(eig_vecs.T[10])
projected_X_12= x_std.dot(eig_vecs.T[11])
projected_X_13= x_std.dot(eig_vecs.T[12])
projected_X_14= x_std.dot(eig_vecs.T[13])
projected_X_15= x_std.dot(eig_vecs.T[14])
projected_X_16= x_std.dot(eig_vecs.T[15]) 
projected_X_17= x_std.dot(eig_vecs.T[16]) 
projected_X_18= x_std.dot(eig_vecs.T[17]) 
projected_X_19= x_std.dot(eig_vecs.T[18]) 
projected_X_20= x_std.dot(eig_vecs.T[19]) 


PCA_DATA['projected_X_1']=projected_X_1
PCA_DATA['projected_X_2']=projected_X_2
PCA_DATA['projected_X_3']=projected_X_3
PCA_DATA['projected_X_4']=projected_X_4
PCA_DATA['projected_X_5']=projected_X_5
PCA_DATA['projected_X_6']=projected_X_6
PCA_DATA['projected_X_7']=projected_X_7
PCA_DATA['projected_X_8']=projected_X_8
PCA_DATA['projected_X_9']=projected_X_9
PCA_DATA['projected_X_10']=projected_X_10
PCA_DATA['projected_X_11']=projected_X_11
PCA_DATA['projected_X_12']=projected_X_12
PCA_DATA['projected_X_13']=projected_X_13
PCA_DATA['projected_X_14']=projected_X_14
PCA_DATA['projected_X_15']=projected_X_15
PCA_DATA['projected_X_16']=projected_X_16
PCA_DATA['projected_X_17']=projected_X_17
PCA_DATA['projected_X_18']=projected_X_18
PCA_DATA['projected_X_19']=projected_X_19
PCA_DATA['projected_X_20']=projected_X_12





plt.cla()
plt.title("PCA_1 VS TIME",fontsize=50)
X_np_time=np.linspace(0,30,len(projected_X_1))
plt.xlabel("time_series",fontsize=25)
plt.ylabel("projected_X_1(PCA_1)",fontsize=25)
plt.scatter(X_np_time,projected_X_1,color="r",s=1000, alpha=0.5)


plt.title("PCA_2 VS TIME",fontsize=50)
X_np_time=np.linspace(0,30,len(projected_X_2))
plt.xlabel("time_series",fontsize=25)
plt.ylabel("projected_X_1(PCA_2)",fontsize=25)
plt.scatter(X_np_time,projected_X_2,color="y",s=1000, alpha=0.5)

plt.title("PCA_3 VS TIME",fontsize=50)
X_np_time=np.linspace(0,30,len(projected_X_3))
plt.xlabel("time_series",fontsize=25)
plt.ylabel("projected_X_1(PCA_3)",fontsize=25)
plt.scatter(X_np_time,projected_X_3,color="g",s=1000, alpha=0.5)

plt.title("PCA_4 VS TIME",fontsize=50)
X_np_time=np.linspace(0,30,len(projected_X_4))
plt.xlabel("time_series",fontsize=25)
plt.ylabel("projected_X_1(PCA_4)",fontsize=25)
plt.scatter(X_np_time,projected_X_4,color="b",s=1000, alpha=0.5)

plt.title("PCA_5 VS TIME",fontsize=50)
X_np_time=np.linspace(0,30,len(projected_X_5))
plt.xlabel("time_series",fontsize=25)
plt.ylabel("projected_X_1(PCA_5)",fontsize=25)
plt.scatter(X_np_time,projected_X_5,color="pink",s=1000, alpha=0.5)

plt.cla()
plt.xlabel("time_series",fontsize=25)
plt.ylabel("ALL TOP 5 PCA's",fontsize=25)
plt.title("PCA VS TIME",fontsize=25)
rad=250
plt.scatter(X_np_time,projected_X_1,color="r",s=rad, alpha=0.5)
plt.scatter(X_np_time,projected_X_2,color="y",s=rad, alpha=0.5)
plt.scatter(X_np_time,projected_X_3,color="g",s=rad, alpha=0.5)
plt.scatter(X_np_time,projected_X_4,color="b",s=rad, alpha=0.5)
plt.scatter(X_np_time,projected_X_5,color="pink",s=rad, alpha=0.5)






#########################################
### GRAPH for Feature Matrix
#X_np_time=np.linspace(1,feature_matrix.shape[0],feature_matrix.shape[0])
#for row in range(feature_matrix.shape[1]) :
#    plt.cla()
#    plt.xlabel("Meal",fontsize=25)
#    plt.ylabel("",fontsize=25)
#    plt.title(list(feature_matrix)[row],fontsize=25)
#    plt.plot(X_np_time,feature_matrix.iloc[:,row])
#    plt.savefig(list(feature_matrix)[row])
#    


PCA_DATA.to_csv("PCA_DATA_MATRIX.csv")


varience =[]
for i in range(len(eig_vals)) :
    varience.append(eig_vals[i]/sum(eig_vals))
    print(varience[i])

    






#dup_pf=f_values
#dup_pa=psd_values
#dup_pa=sorted(dup_pa)
#a1=dup_pa[-2]
#a2=dup_pa[-3]
#f1=f_values[psd_values.tolist().index(a1)]
#f2=f_values[psd_values.tolist().index(a2)]
#eig_df =pd.DataFrame(eig_vecs)
#list


#plt.plot(np.linspace(1,len(varience]),len(varience),varience))

#
#plt.bar(np.linspace(1,len(eig_vecs[0]),len(eig_vecs[0])),eig_vecs[0])
#plt.figure()
#
#for i in range(eig_vecs.shape[0]) :
#    plt.scatter(np.linspace(1,len(eig_vecs[i]),len(eig_vecs[i])),eig_vecs[i])
#
#
#



















