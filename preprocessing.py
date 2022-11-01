import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)
    return res

def preprocess_adult(path='file_path'):
    '''
    Adult income dataset
    Data are available at https://www.kaggle.com/wenruliu/adult-income-dataset
    After preprocessing, we have
    X : n=48842, p=108
    Y : n=48842, 2 classes
    '''
    adult_raw = pd.read_csv(path+'/adult.csv')
    Y = adult_raw['income']
    adult_raw = adult_raw.drop(['income'], axis=1)
    object_list = adult_raw.dtypes[adult_raw.dtypes == 'object'].index
    adult_raw = adult_raw.dropna()
    for col in object_list:
        adult_raw = encode_and_bind(adult_raw, col)
        
    X = adult_raw
    Y = pd.DataFrame({'income': LabelEncoder().fit_transform(Y)})
    X.to_csv(path+'/adult_X.csv', index=False)    
    Y.to_csv(path'/adult_Y.csv', index=False)    


def preprocess_postures(path='file_path'):
    '''
    Wearable Computing: Classification of Body Postures and Movements (PUC-Rio) Data Set
    Data are available at https://archive.ics.uci.edu/ml/machine-learning-databases/00250/
    After preprocessing, we have
    X : n=74975, p=15
    Y : n=74975, 5 classes
    '''
    postures_raw = pd.read_csv(path+'/postures.csv')
    postures_raw = postures_raw.iloc[1:,:17]
    postures_raw = postures_raw.reset_index(drop=True)
    print(postures_raw.shape)

    question_index = np.unique(np.where(postures_raw == '?')[0]).tolist()
    postures = postures_raw.drop(question_index, axis=0)
    Y = postures['Class']
    postures = postures.drop(['Class'], axis=1)
    print(postures.shape)
        
    X = postures.iloc[:,1:].astype('float')
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    Y = pd.DataFrame({'Class': LabelEncoder().fit_transform(Y)})
    X.to_csv(path+'/postures_X.csv', index=False)
    Y.to_csv(path+'/postures_Y.csv', index=False)  




