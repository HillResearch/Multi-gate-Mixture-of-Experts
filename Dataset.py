# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 10:04:01 2023

@author: hwang147
"""

import os
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader


import warnings

warnings.filterwarnings(action='ignore', category=pd.errors.DtypeWarning)

class Dataset_patient(Dataset):
    def __init__(self, root_path, data_path, size):
        self.root_path = root_path
        self.data_path = data_path
        self.size = size
        #self.max_missing_ratio = max_missing_ratio
        
        
        
        self.__read_basic__(size,data_path[0])
        self.__read_diagnosis__(size,data_path[1])
        self.__read_treatment__(size,data_path[3])
    
    def __read_basic__(self,size,basic_profile_path):
        dfs = []
        root_path = self.root_path
        
        for i in range(size[0],size[1],500):
            path = 'basic'+'_'+str(i)+'-'+str(i+500)+'.csv'
        
            df = pd.read_csv(os.path.join(root_path,
                                          'basic',path))
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)
        df = df.drop(['Unnamed: 0'],axis = 1)
        df = df.drop(['ID'],axis = 1)
        df = df.drop(['month_of_birth/ month_of_birth_0_0'],axis = 1)
        #df = df.select_dtypes(exclude=['object'])
        #df = df.dropna(axis=1, how='all')
        df = df.fillna(0.0)
        
        
        x = df.values
        
        index = []
        for i in range(len(x.T)):
            if np.var(x[:,i])!=0:
                index.append(i)
                x[:,i] = (x[:,i]-min(x[:,i]))/(max(x[:,i])-min(x[:,i]))
        X = x[:,index]
        
        
        self.data_basic = X
    def __read_diagnosis__(self,size,diagnosis_path):
        dfs = []

        root_path = self.root_path
        for i in range(size[0],size[1],500):
            path = 'diagnosis'+'_'+str(i)+'-'+str(i+500)+'.csv'
        
            df = pd.read_csv(os.path.join(root_path,
                                          'diagnosis',path))
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)
        df = df.drop(['Unnamed: 0'],axis = 1)
        df = df.select_dtypes(exclude=['object'])
        df = df.dropna(axis=1, how='all')
        df = df.fillna(0.0)
        x = df.values
        
        index = []
        for i in range(len(x.T)):
            if np.var(x[:,i])!=0:
                index.append(i)
                x[:,i] = (x[:,i]-min(x[:,i]))/(max(x[:,i])-min(x[:,i]))
        X = x[:,index]
        self.data_diagnosis = X
        
    def __read_treatment__(self,size,treatment_path):
        dfs = []

        root_path = self.root_path
        for i in range(size[0],size[1],500):
            path = 'treatment'+'_'+str(i)+'-'+str(i+500)+'.csv'
        
            df = pd.read_csv(os.path.join(root_path,
                                          'treatment',path))
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)
        df = df.drop(['Unnamed: 0'],axis = 1)
        df1 = df.values
        
        y = []
        for i in range(0,3000,5):
            k = []
            data = df1[:,i:i+5]
            d1 = data[:,0]
            d2 = data[:,2]
            d1 = d1.astype(float)
            d2 = d2.astype(str)
            nan_mask1 = np.isnan(d1)
            nan_mask2 = d2 == 'nan'
            for j in range(len(nan_mask1)):
                
                if nan_mask1[j] == True or d1[j] == -1009.0:
                    if nan_mask2[j] == True:
                        k.append(0)
                    else:k.append(1)
                else:
                    k.append(1)
            y.append(k)
        self.data_treatment = np.array(y).T[:,6:9]
    def __len__(self):
        return len(self.data_basic)
    def __getitem__(self, index):
        'Generates one sample of data'

        # get the train data
        basic = self.data_basic[index]
        diagnosis = self.data_diagnosis[index]
        #result = self.data_result[index]
        treatment = self.data_treatment[index]
        

        return basic,diagnosis,treatment
    



# root_path= r'D:\Hill\Data'
# data_path=['basic','diagnosis','result','treatment']
# size = [0,1000]

# Data = Dataset_patient(root_path,data_path,size)

# training_data = DataLoader(Data, batch_size=10)

# for x,y,z in training_data:
#     print(z.shape)















