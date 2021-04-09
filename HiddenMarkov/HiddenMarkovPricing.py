#!/usr/bin/env python
# coding: utf-8

# In[1]:


import yfinance as yf
import datetime as dt
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from hmmlearn import hmm
from sklearn.model_selection import train_test_split


# In[2]:


# Download AMZN stock data as demonstration
# Swap for BTC, ETH later on
am = yf.download("AMZN", start="2010-09-01", end="2020-09-01")
am


# In[3]:


# Add a new column for log return
#am['pct_change'] = am['Adj Close'].pct_change()
#am['pct_change1'] = (am['Adj Close']-am['Adj Close'].shift(1))/am['Adj Close'].shift(1)
#am['log_return1'] = np.log(1 + am['pct_change1'])

# Multiple sequences of observations
am['log_return'] = np.log(am['Adj Close']/am['Adj Close'].shift(1))
am['HiLo_Diff'] = (am['High'] - am['Low'])/am['Open']*100
am['Vol_change'] = am['Volume']/am['Volume'].shift(1) - 1

# Drop NA
am.dropna(axis=0, inplace=True)

# Reset index
#am.reset_index(inplace=True)

# Train-test_validation split
am_train, am_test = train_test_split(am, test_size=0.2, shuffle=False)
am_train, am_val = train_test_split(am_train, test_size=0.25, shuffle=False)

# Extract observation sequence and drop NA
HiLo = np.array(am_train['HiLo_Diff'])
volChg = np.array(am_train['Vol_change'])
logR = np.array(am_train['log_return'])

# Turn 1-D array into 2-D array
#X = np.array(am['log_return'].dropna())
#X = list(map(lambda a:[a], X))

# Multiple observation sequences
obs = np.column_stack((HiLo, volChg, logR))
#obs = obs[~np.isnan(obs).any(axis=1), :]


# In[4]:


# Define the Hidden Markove Model to fit
# Optimize the number of hidden states
bic_multi = np.repeat(np.nan, 9)
size = obs.shape[0]
k = 2
for m in range(1, 9):
    model = hmm.GaussianHMM(n_components=m, covariance_type="full", n_iter=500)
    model.fit(obs)
    logLike = model.score(obs)
    bic_multi[m-1] = (-2.0) * logLike + np.log(size) * (m**2 + k*m - 1)


# In[5]:


plt.plot(np.linspace(1, 9, 9), bic_multi, marker='o')
plt.xlabel('Number of Hidden States')
plt.ylabel('BIC')
plt.show()


# In[6]:


model_multi = hmm.GaussianHMM(n_components=4, covariance_type="full", n_iter=500)
model_multi.fit(obs)


# In[7]:


# Use validation set to find optimal n_day_latency
HiLo_val = np.array(am_val['HiLo_Diff'])
volChg_val = np.array(am_val['Vol_change'])
logR_val = np.array(am_val['log_return'])
price_val = np.array(am_val['Adj Close'])

obs_val = np.column_stack((HiLo_val, volChg_val, logR_val, price_val))

# Get the transition probability matrix
A = model_multi.transmat_
#print(A)
# Get the most likely next state for each state
MaxPos = np.argmax(A, axis=1)
# Get the Gaussian distribution mean matrix
meansmat = model_multi.means_
#print(model_multi.means_)

# Compute MAPE
def mape(act, pred): 
    act, pred = np.array(act), np.array(pred)
    mape =  np.mean(np.abs((act - pred) / act))
    # format "{:.10%}".format(mape)
    return mape

# Compute MSE
def mse(act, pred):
    act, pred = np.array(act), np.array(pred)
    mse = np.square(np.subtract(act, pred)).mean()
    return mse

# Initialize the mape list
mapes = np.repeat(np.nan, 28)
mses = np.repeat(np.nan, 28)
k = 0
for n in range(3, 30):
    # Initialize the lists for current states and next states
    curr_state = np.repeat(np.nan, obs_val.shape[0]-n+1)
    next_state = np.repeat(np.nan, obs_val.shape[0]-n+1)
    # Initialize the last day price list
    prices = np.repeat(np.nan, obs_val.shape[0]-n+1)
    actuals = obs_val[n-1:, 3]

    for j in range(0, obs_val.shape[0]-n+1):
        curr_state[j] = model_multi.predict(obs_val[j:j+n, 0:3])[-1]
        #print(model_multi.predict(obs_val))
        next_state[j] = MaxPos[int(curr_state[j])]
        # Find the mean log return for the most likely next state
        logR_mean = meansmat[int(next_state[j]), 2]
        prices[j] = np.exp(logR_mean)*obs_val[j+n-1, 3]
    
    mapes[k] = mape(actuals, prices)
    mses[k] = mse(actuals, prices)
    k = k+1


# In[8]:


plt.plot(np.linspace(3, 30, 28), mapes, marker='o')
plt.xlabel('Number of Latency Days')
plt.xticks(range(3, 30, 3))
plt.ylabel('MAPE')
plt.show()


# In[33]:


# First assume latencyDays = day
day = 9
HiLo_test = np.array(am_test['HiLo_Diff'])
volChg_test = np.array(am_test['Vol_change'])
logR_test = np.array(am_test['log_return'])
price_test = np.array(am_test['Adj Close'])

obs_test = np.column_stack((HiLo_test, volChg_test, logR_test, price_test))

# Initialize the lists for current states and next states
curr_state_opt = np.repeat(np.nan, obs_test.shape[0]-day+1)
next_state_opt = np.repeat(np.nan, obs_test.shape[0]-day+1)
prices_opt = np.repeat(np.nan, obs_test.shape[0]-day+1)

for j in range(0, obs_test.shape[0]-day+1):
    curr_state_opt[j] = model_multi.predict(obs_test[j:j+day, 0:3])[-1]
    #print(obs_test[j:j+14, 0:3].shape)
    #print(model_multi.predict(obs_test))
    next_state_opt[j] = MaxPos[int(curr_state_opt[j])]
    # Find the mean log return for the most likely next state
    logR_mean_opt = meansmat[int(next_state_opt[j]), 2]
    prices_opt[j] = np.exp(logR_mean_opt)*obs_test[j+day-1, 3]
    
dates = pd.to_datetime(am_test.index)[day-1:]
    
plt.figure(figsize=(15,8))
plt.plot(dates, prices_opt, label='predict', color='blue')
actuals_test = obs_test[day-1:, 3]
plt.plot(dates, actuals_test, label='actual', color="orange")
plt.xlabel('Dates')
plt.ylabel('Stock Price')
plt.legend(loc="upper left")


# In[24]:


mse(actuals_test, prices_opt)


# In[11]:


mape(actuals_test, prices_opt)

