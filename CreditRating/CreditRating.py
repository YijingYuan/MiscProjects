#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras import optimizers
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.utils import resample


# In[2]:


cr = pd.read_csv('SPRatings.csv')
cr = cr.drop(cr.columns[0], axis=1)
cr['Data Date'] = pd.to_datetime(cr['Data Date'], format='%Y%m%d')


# In[3]:


# Check for missing values
cr.isnull().values.any()


# In[4]:


# Check class imbalance
cr.iloc[:, 1].value_counts()


# In[5]:


#cr.columns


# In[6]:


cr = cr.rename(columns={'S&P Domestic Long Term Issuer Credit Rating': 'Rating'})


# In[7]:


cr['upsample'] =cr['Rating']                .apply(lambda x: 1 if x in ['BB+', 'AA', 'AAA', 'AA+'] else 0)


# In[ ]:


cr['key'] = cr['Data Date'].apply(lambda x: str(x.year) + '-' + str(x.quarter)).astype(str)
ir = pd.read_csv('IRLTLT01USQ156N.csv')
ir = ir.rename(columns={'IRLTLT01USQ156N': 'Interest_Rate'})
ir['DATE'] = pd.to_datetime(ir['DATE'])
ir['key'] = ir['DATE'].apply(lambda x: str(x.year) + '-' + str(x.quarter)).astype(str)
cr1 = pd.merge(cr, ir.iloc[:, 1:], how='left', on='key')
cr1 = cr1.drop('key', axis=1)


# In[39]:


# Split training and testing data seperately
cr_train, cr_test = train_test_split(cr, test_size=0.3)
cr_test = cr_test.drop(['upsample', 'key'], axis=1)
X_test = cr_test.iloc[:, 5:]
y_test = cr_test.iloc[:, 1]
cr_test.columns


# In[9]:


def upsample(df):
    df_maj = df[df['upsample']==0]
    
    # Upsample minority class
    df_up1 = resample(df[df['Rating']=='BB+'], replace=True, n_samples=200)
    df_up2 = resample(df[df['Rating']=='AA+'], replace=True, n_samples=200)
    df_up3 = resample(df[df['Rating']=='AA'], replace=True, n_samples=200)
    df_up4 = resample(df[df['Rating']=='AAA'], replace=True, n_samples=200)
    df_up5 = resample(df[df['Rating']=='BBB-'], replace=True, n_samples=200)

    # Combine majority class with upsampled minority class
    df_upsample = pd.concat([df_maj, df_up1, df_up2, df_up3, df_up4, df_up5])
    return df_upsample


# In[10]:


cr_train_up = upsample(cr_train)
cr_train_up = cr_train_up.drop(['upsample', 'key'], axis=1)
X_train = cr_train_up.iloc[:, 5:]
y_train = cr_train_up.iloc[:, 1]


# In[11]:


cr_test.iloc[:, 1].value_counts()


# In[12]:


cr_train_up.iloc[:, 1].value_counts()


# In[18]:


# Fit a grid search
param_grid = {'C': [200, 300, 400, 500, 600],  
              'gamma': [0.001, 0.002, 0.005], 
              'kernel': ['rbf']}  
  
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 2)
grid.fit(X_train, y_train)
grid.best_params_


# In[19]:


# Random Forest Classifier
rfc = RandomForestClassifier()

random_grid = {'n_estimators': [500, 700, 1000, 1200],
               'max_features': ['auto', 'log2'],
               'max_depth': [100, 200, 300, 500],
               'min_samples_split': [1, 2, 5],
               'min_samples_leaf': [1, 2, 5],
               'bootstrap': [True, False]}

rf_random = RandomizedSearchCV(estimator = rfc, param_distributions = random_grid, n_iter = 100,                               verbose=2, random_state=100, n_jobs = -1)
rf_random.fit(X_train, y_train)
rf_random.best_params_


# In[15]:


# Multi-Layer Perceptron
mlp = MLPClassifier()

parameter_space = {
    'hidden_layer_sizes': [(100,), (100, 50), (100, 100), (100, 50, 100)],
    'activation': ['logistic', 'tanh', 'relu']
}

mlp_random = GridSearchCV(estimator = mlp, param_grid = parameter_space, verbose=2)
mlp_random.fit(X_train, y_train)
mlp_random.best_params_


# In[16]:


# Extreme Gradient Boosting
xgb =  XGBClassifier()

random_grid = {'n_estimators': [50, 100, 200, 500, 1000],
               'max_features': ['auto', 'log2'],
               'max_depth': [50, 100, 200, 500, 1000],
               'min_samples_split': [1, 2, 5, 10],
               'min_samples_leaf': [1, 2, 5, 10],
               'bootstrap': [True, False]}

xgb_random = RandomizedSearchCV(estimator = xgb, param_distributions = random_grid, n_iter = 100,                                verbose=2, random_state=100, n_jobs = -1)

xgb_random.fit(X_train, y_train)
xgb_random.best_params_


# In[20]:


def split_upsample(df):
    df_train, df_test = train_test_split(df, test_size=0.3)
    df_test = df_test.drop(['upsample', 'key'], axis=1)
    X_test = df_test.iloc[:, 5:]
    y_test = df_test.iloc[:, 1]
    df_train_up = upsample(df_train)
    df_train_up = df_train_up.drop(['upsample', 'key'], axis=1)
    X_train = df_train_up.iloc[:, 5:]
    y_train = df_train_up.iloc[:, 1]
    return X_train, y_train, X_test, y_test


# In[35]:


df1 = pd.DataFrame()
for i in range(10):
    X_train, y_train, X_test, y_test = split_upsample(cr)
    clf1 = SVC(C=200, gamma = 0.002, kernel='rbf')
    clf1.fit(X_train, y_train)
    pred_svm = clf1.predict(X_test) 
    f1 = metrics.f1_score(y_test, pred_svm, average='micro')
    acc = metrics.accuracy_score(y_test, pred_svm)
    prec = metrics.precision_score(y_test, pred_svm, average='micro')
    recall = metrics.recall_score(y_test, pred_svm, average='micro')
    df1 = df1.append([[f1, acc, prec, recall]])
df1 = df1.rename(columns={0:'f1_score', 1:'accuracy', 2:'precision', 3:'recall'})
df1


# In[28]:


df2 = pd.DataFrame()
for i in range(10):
    X_train, y_train, X_test, y_test = split_upsample(cr)
    clf2 = MLPClassifier(hidden_layer_sizes=(100, 50), activation='tanh', max_iter=300).fit(X_train, y_train)
    pred_mlp = clf2.predict(X_test) 
    f1 = metrics.f1_score(y_test, pred_mlp, average='micro')
    acc = metrics.accuracy_score(y_test, pred_mlp)
    prec = metrics.precision_score(y_test, pred_mlp, average='micro')
    recall = metrics.recall_score(y_test, pred_mlp, average='micro')
    df2 = df2.append([[f1, acc, prec, recall]])
df2 = df2.rename(columns={0:'f1_score', 1:'accuracy', 2:'precision', 3:'recall'})
df2


# In[30]:


df3 = pd.DataFrame()
for i in range(10):
    X_train, y_train, X_test, y_test = split_upsample(cr)
    clf3 = XGBClassifier(n_estimators=1000, min_samples_split=5, min_samples_leaf=5, max_features='log2',                         max_depth=200, bootstrap=False).fit(X_train, y_train)
    pred_xgb = clf3.predict(X_test) 
    f1 = metrics.f1_score(y_test, pred_xgb, average='micro')
    acc = metrics.accuracy_score(y_test, pred_xgb)
    prec = metrics.precision_score(y_test, pred_xgb, average='micro')
    recall = metrics.recall_score(y_test, pred_xgb, average='micro')
    df3 = df3.append([[f1, acc, prec, recall]])
df3 = df3.rename(columns={0:'f1_score', 1:'accuracy', 2:'precision', 3:'recall'})
df3


# In[32]:


df4 = pd.DataFrame()
for i in range(10):
    X_train, y_train, X_test, y_test = split_upsample(cr)    
    clf4 = RandomForestClassifier(n_estimators=700, min_samples_split=5, min_samples_leaf=2, max_features='log2',                         max_depth=100, bootstrap=False).fit(X_train, y_train)
    pred_rf = clf4.predict(X_test) 
    f1 = metrics.f1_score(y_test, pred_rf, average='micro')
    acc = metrics.accuracy_score(y_test, pred_rf)
    prec = metrics.precision_score(y_test, pred_rf, average='micro')
    recall = metrics.recall_score(y_test, pred_rf, average='micro')
    df4 = df4.append([[f1, acc, prec, recall]])
df4 = df4.rename(columns={0:'f1_score', 1:'accuracy', 2:'precision', 3:'recall'})
df4


# In[ ]:


df1 = pd.DataFrame()
for i in range(10):
    X_train, y_train, X_test, y_test = split_upsample(cr1)
    clf1 = SVC(C=200, gamma = 0.002, kernel='rbf')
    clf1.fit(X_train, y_train)
    pred_svm = clf1.predict(X_test) 
    f1 = metrics.f1_score(y_test, pred_svm, average='micro')
    acc = metrics.accuracy_score(y_test, pred_svm)
    prec = metrics.precision_score(y_test, pred_svm, average='micro')
    recall = metrics.recall_score(y_test, pred_svm, average='micro')
    df1 = df1.append([[f1, acc, prec, recall]])
df1 = df1.rename(columns={0:'f1_score', 1:'accuracy', 2:'precision', 3:'recall'})
df1


# In[ ]:


df2 = pd.DataFrame()
for i in range(10):
    X_train, y_train, X_test, y_test = split_upsample(cr1)
    clf2 = MLPClassifier(hidden_layer_sizes=(100, 50), activation='tanh', max_iter=300).fit(X_train, y_train)
    pred_mlp = clf2.predict(X_test) 
    f1 = metrics.f1_score(y_test, pred_mlp, average='micro')
    acc = metrics.accuracy_score(y_test, pred_mlp)
    prec = metrics.precision_score(y_test, pred_mlp, average='micro')
    recall = metrics.recall_score(y_test, pred_mlp, average='micro')
    df2 = df2.append([[f1, acc, prec, recall]])
df2 = df2.rename(columns={0:'f1_score', 1:'accuracy', 2:'precision', 3:'recall'})
df2


# In[ ]:


df3 = pd.DataFrame()
for i in range(10):
    X_train, y_train, X_test, y_test = split_upsample(cr)
    clf3 = XGBClassifier(n_estimators=1000, min_samples_split=5, min_samples_leaf=5, max_features='log2',                         max_depth=200, bootstrap=False).fit(X_train, y_train)
    pred_xgb = clf3.predict(X_test) 
    f1 = metrics.f1_score(y_test, pred_xgb, average='micro')
    acc = metrics.accuracy_score(y_test, pred_xgb)
    prec = metrics.precision_score(y_test, pred_xgb, average='micro')
    recall = metrics.recall_score(y_test, pred_xgb, average='micro')
    df3 = df3.append([[f1, acc, prec, recall]])
df3 = df3.rename(columns={0:'f1_score', 1:'accuracy', 2:'precision', 3:'recall'})
df3


# In[ ]:


df4 = pd.DataFrame()
for i in range(10):
    X_train, y_train, X_test, y_test = split_upsample(cr)    
    clf4 = RandomForestClassifier(n_estimators=700, min_samples_split=5, min_samples_leaf=2, max_features='log2',                         max_depth=100, bootstrap=False).fit(X_train, y_train)
    pred_rf = clf4.predict(X_test) 
    f1 = metrics.f1_score(y_test, pred_rf, average='micro')
    acc = metrics.accuracy_score(y_test, pred_rf)
    prec = metrics.precision_score(y_test, pred_rf, average='micro')
    recall = metrics.recall_score(y_test, pred_rf, average='micro')
    df4 = df4.append([[f1, acc, prec, recall]])
df4 = df4.rename(columns={0:'f1_score', 1:'accuracy', 2:'precision', 3:'recall'})
df4

