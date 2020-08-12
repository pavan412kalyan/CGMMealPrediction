
# -*- coding: utf-8 -*-


import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np

from features import extract


meal_1 = pd.read_csv(r'mealData1.csv',names=list(range(30)))
meal_2 = pd.read_csv(r'mealData2.csv',names=list(range(30)))
meal_3 = pd.read_csv(r'mealData3.csv',names=list(range(30)))
meal_4 = pd.read_csv(r'mealData4.csv',names=list(range(30)))
meal_5 = pd.read_csv(r'mealData5.csv',names=list(range(30)))

meal=pd.concat([meal_1,meal_2,meal_3,meal_4,meal_5],ignore_index=True)




l=len(meal)
drop_rows=[]
for k in range(l-1) :
    if(meal.iloc[k,:].isnull().sum()>0) :
        drop_rows.append(k)  

meal =meal.drop(drop_rows)
meal.reset_index(drop=True,inplace=True)

#fill missing values using quadratic interpolation    
meal.interpolate(method='quadratic',limit_direction='both',inplace=True)
meal.bfill(inplace=True)
meal.ffill(inplace=True)

#####################################################

nomeal_1 = pd.read_csv(r'Nomeal1.csv',names=list(range(30)))
nomeal_2 = pd.read_csv(r'Nomeal2.csv',names=list(range(30)))
nomeal_3 = pd.read_csv(r'Nomeal3.csv',names=list(range(30)))
nomeal_4 = pd.read_csv(r'Nomeal4.csv',names=list(range(30)))
nomeal_5 = pd.read_csv(r'Nomeal5.csv',names=list(range(30)))

nomeal=pd.concat([nomeal_1,nomeal_2,nomeal_3,nomeal_4,nomeal_5],ignore_index=True)

l=len(nomeal)
drop_rows_n=[]
for k in range(l-1) :
    if(nomeal.iloc[k,:].isnull().sum()>0) :
        drop_rows_n.append(k)  
nomeal =nomeal.drop(drop_rows_n)
nomeal.reset_index(drop=True,inplace=True)

#fill missing values using quadratic interpolation    
nomeal.interpolate(method='quadratic',limit_direction='both',inplace=True)
nomeal.bfill(inplace=True)
nomeal.ffill(inplace=True)




#################################
meal.to_csv("combined_meal.csv",index=False,header=False)
nomeal.to_csv("combined_nomeal.csv",index=False,header=False)
#

###########################################
X=[29-i for i in range(0,30)]



#for i in range(10) : 
#    plt.plot(X,meal.iloc[73,:],marker="X")
#    
    





#print("AFTER FUNCTION")
###call function
feature_Matrix_Meal=pd.DataFrame()
feature_Matrix_Meal=extract(meal)

feature_Matrix_NoMeal=pd.DataFrame()
feature_Matrix_NoMeal=extract(nomeal)


#
#feature_Matrix_Meal['class']=1
#feature_Matrix_NoMeal['class']=0
#

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train_meal = sc.fit_transform(feature_Matrix_Meal)
train_nomeal = sc.fit_transform(feature_Matrix_NoMeal)



pca = PCA(n_components=5)
pca_vectors=pca.fit(train_meal)

eigen_vector = pd.DataFrame(data=(pca.components_).T)
eigen_vector.to_csv('Eigen_Vectors.csv',index = False, header = False)
  
import pickle  
pkl_filename = "pca_model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(pca_vectors, file)



pca_data_meal = pca.fit_transform(train_meal)
pca_data_nomeal = pca.fit_transform(train_nomeal)



####below is to convert to dataframe

pca_data_meal=pd.DataFrame(pca_data_meal)
pca_data_meal['class']=1

pca_data_nomeal=pd.DataFrame(pca_data_nomeal)
pca_data_nomeal['class']=0




final_dataset = pca_data_meal.append([pca_data_nomeal])
final_dataset.reset_index(drop=True, inplace= True)
X_train =final_dataset.iloc[:,:-1]
Y_train =final_dataset.iloc[:,-1]

#
#from sklearn.model_selection import train_test_split
#
#X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.4, random_state=0)

###################
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 2), random_state=1)
mlp.fit(X_train, Y_train)

pkl_fileName = "multiLayerPerceptron_model.pkl"
with open(pkl_fileName, 'wb') as file:
    pickle.dump(mlp, file)

mlp.score(X_train,Y_train)
#####################################
#from sklearn import svm
#svm = svm.SVC(kernel = 'rbf', gamma=0.009, C=1)
#svm.fit(X_train, Y_train)
#svm.score(X_train,Y_train)
#






    

###################
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
kernel = 1.0 * RBF(1.0)
gpc = GaussianProcessClassifier(kernel=kernel, random_state=0)
gpc.fit(X_train, Y_train)

pkl_fileName = "Guassian_model.pkl"
with open(pkl_fileName, 'wb') as file:
    pickle.dump(gpc, file)


acc=gpc.score(X_train,Y_train)

#acc=gpc.score(X_test,Y_test)

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


gaussianNB_pred = gpc.predict(X_train)

precision = precision_score(Y_train, gaussianNB_pred, average='binary')
recall = recall_score(Y_train, gaussianNB_pred, average='binary')
F1=(2*precision*recall)/(precision+recall)
print("---------------ON TRAIN  DATA-----------------")
print(" Using Guassian Classifier\n precision="+str(precision*100)+"\nRecall="+str(recall*100))
print("Accuracy  ==" + str(acc*100))
print("F1 is="+ str(F1))
print("---------------ON TRAIN DATA--------------")




###########################################
from sklearn.model_selection import KFold
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
kernel = 1.0 * RBF(1.0)

kfold = KFold(5, True, random_state=3)
X=X_train
Y=Y_train
val_count=0
print("+++++++++++++++++ON VALIDATION SET -K FOLD++++++++++++")
for train, test in kfold.split(X, Y):
    gpc = GaussianProcessClassifier(kernel=kernel, random_state=1)
    X_train_f, X_test_f = X.iloc[train], X.iloc[test]
    Y_train_f, y_test_f = Y.iloc[train], Y.iloc[test]

    gpc.fit(X_train_f, Y_train_f)
    val_count+=1
    print("for Guassian--Validation set"+ str(val_count)+"==" +str(gpc.score(X_test_f,y_test_f)))
    
    
print("+++++++++++++++++ON VALIDATION SET - K FOLD++++++++++")


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

#
#






















#############################################

#from sklearn.model_selection import train_test_split
#
#X_train, X_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.4, random_state=0)
#
#acc=gpc.score(X_test,y_test)
#print("Accuracy on training set 1 " + str(acc*100))
#
#
#from sklearn import cross_validation 
#data_kfold = cross_validation.KFold(len(X_train), n_folds=10, indices=False) 
#


#from sklearn.utils import shuffle
#shuf_X, shuf_y = shuffle(X_train, Y_train)
#gpc.score(shuf_X.iloc[100:110],shuf_y.iloc[100:110])
#
#



















    































### for graphh only
#plt.cla()
#plt.title('CGM velocity Shift')
#plt.xlabel('Time ')
#plt.ylabel('CGM')
#plt.axvline(max(slope_diff)[1],color='r',label="interval")
#plt.axvline(max(slope_diff)[2],color='r')
#plt.plot(X,Y)
#plt.scatter(X,Y)





#meal1=pd.read_csv('mealData1.csv')
#meal2=pd.read_csv('mealData2.csv')
#meal3=pd.read_csv('mealData3.csv')
#meal4=pd.read_csv('mealData4.csv')
#meal5=pd.read_csv('mealData5.csv')

