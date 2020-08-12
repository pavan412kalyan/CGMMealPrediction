# -*- coding: utf-8 -*-
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


from features import extract

def test(file_name) :
    
#    file_name="combined_nomeal.csv"
    with open("Guassian_model.pkl", 'rb') as file:        
        Guassian_model = pickle.load(file)      
    with open("pca_model.pkl", 'rb') as file:
        pca = pickle.load(file)
        
    test_data = pd.read_csv(file_name, header=None)
    print("---")
    fm=extract(test_data)
    print("---")

    sc = StandardScaler()
    test_data_set = sc.fit_transform(fm)
    pca_dataset=pca.fit_transform(test_data_set)
    gaussianNB_pred = Guassian_model.predict(pca_dataset)
    print("Classes of your given test  "+ str(gaussianNB_pred))
    np.savetxt("output.csv", gaussianNB_pred, delimiter=",", fmt='%d')
    print("predicted class labels are saved in output.csv file")
    


if __name__ == "__main__":
    
    file_name = sys.argv[1]
    test(file_name)
        