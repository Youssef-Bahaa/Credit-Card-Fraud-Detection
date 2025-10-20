#credit_fraud_utils_data.py
import pandas as pd
import numpy as np
from imblearn.combine import SMOTEENN, SMOTETomek
from scipy import stats
from sklearn.preprocessing import StandardScaler, QuantileTransformer, RobustScaler,MinMaxScaler
from imblearn.over_sampling import SMOTE,RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import yaml
import os


def load_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f'Data loaded: {df.shape}')
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f'File not found: {file_path}')
    except Exception as e:
        raise Exception(f'Error while loading the file : {file_path}')


def load_data(config):
    paths = config['paths']
    train = load_csv(paths['train'])
    val = load_csv(paths['val'])

    return train,val

def load_test(config):
    test = load_csv(config['paths']['test'])

    X_test = test.drop(config['target_column'],axis=1)
    y_test = test[config['target_column']]

    return X_test,y_test


def split_feature_target(df,target):
    X = df.drop(target,axis=1)
    y = df[target]
    return X,y

def remove_outliers(df,method):
    print(df['Class'].value_counts())
    df_clean = df.copy()
    columns = df.select_dtypes(include=[np.number]).columns
    if method == 'iqr':
        for col in columns:
            Q1 = df_clean[col].quantile(0.01)
            Q3 = df_clean[col].quantile(0.99)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]

    elif method == 'zscore':
        z_scores = np.abs(stats.zscore(df_clean[columns]))
        df_clean = df_clean[(z_scores < 3).all(axis=1)]

    else:
        print('unknown outlier removal method')
        return

    print(f'Outliers removed : {df.shape[0] - df_clean.shape[0]} rows')
    print(df_clean['Class'].value_counts())
    return df_clean


def remove_duplicates(df):
    initial = df.shape[0]
    df_clean = df.drop_duplicates()
    removed = initial - df_clean.shape[0]
    print(f'Duplicates removed : {removed} rows')
    return df_clean

def handle_missing_values(df,method):
    df_clean = df.copy()
    if method == 'drop':
        df_clean = df_clean.dropna()
    elif method =='mean':
        df_clean.fillna(df_clean.mean(), inplace=True)
    elif method == 'mode':
        df_clean.fillna(df_clean.mode().iloc[0], inplace=True)
    elif method =='median':
        df_clean.fillna(df_clean.median(), inplace=True)
    else:
        print('unknown removal method')

    return df_clean


def scale_features_train(X_train,method):
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    elif method == 'quantile_uniform':
        scaler = QuantileTransformer(output_distribution='uniform')
    else:
        print('unknown scaling method')

    X_train_scaled = scaler.fit_transform(X_train)
    return X_train_scaled,scaler


def scale_features_test(X_test,scaler):
    X_test_scaled = scaler.transform(X_test)
    return X_test_scaled


def balance_dataset(X_train,y_train,method,random_state,params):
    zeros = (y_train == 0).sum()
    ones  = (y_train == 1).sum()
    print('Data before balancing')
    print('Number of positive class 1:',ones)
    print('Number of negative class 0:', zeros)

    if method == 'smote':
        balancer = SMOTE(**params,random_state=random_state)
    elif method == 'ros':
        balancer = RandomOverSampler(**params,random_state=random_state)
    elif method =='rus':
        balancer = RandomUnderSampler(**params,random_state=random_state)
    elif method == 'smoteenn':
        balancer = SMOTEENN(**params,random_state=random_state)
    elif method == 'smotetomek':
        balancer = SMOTETomek(**params,random_state=random_state)

    X_res , y_res = balancer.fit_resample(X_train,y_train)

    zeros = (y_res == 0).sum()
    ones  = (y_res == 1).sum()

    print('Data after balancing')
    print('Number of positive class 1:',ones)
    print('Number of negative class 0:',zeros )

    return X_res , y_res
