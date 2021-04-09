#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from datetime import datetime

# Import data and set the 'Date' Column as index
Return = pd.read_csv('HW5_Q1_data.csv', index_col = "Date")
# Set the index data type to datetime
Return.index = pd.to_datetime(Return.index)
# Check index
# print(ReturnShift.index)

# Fill na values
Return = Return.fillna(0)
# Checking if there are any na values
# ReturnShift[ReturnShift.isna().any(axis=1)]

### Identify momentums and reversions
# Transform data into positive and negative signs
ReturnSign = np.sign(Return)
# Indentify momentums and reversions
# Momentum = 1; Reversion = -1
TrendSign = ReturnSign * ReturnSign.shift(1)

# Dropna
TrendSign = TrendSign.dropna(axis = 0)
# Upsampling to get month level data
MonthlyReturn = TrendSign.resample('M').mean()
# Change index to month
MonthlyReturn.index = MonthlyReturn.index.to_period('M')

### Identify change between momentum and reversion
# Aggregate over all stocks
AveSign = np.sign(MonthlyReturn.mean(axis = 1))
print(AveSign.replace({1:'Momentum', -1:'Reversion'}))

# Indentify any shifts between momentum and reversion
# Shift = -1
Shift = AveSign * AveSign.shift(1)

# Return the last month of the change
print('Last month in Shift:', Shift[Shift < 0].index[-1])

# Indentifying months with momentum or reversion
MomentumIndex = AveSign[AveSign > 0].index
ReversionIndex = AveSign[AveSign < 0].index

# Calculate averages of momentums and reversions
# Aggregate: firstly over month, then over stocks
AveMoment = MonthlyReturn.loc[MomentumIndex].mean(axis = 0).mean()
AveRever = MonthlyReturn.loc[ReversionIndex].mean(axis = 0).mean()
print('Momentum period average:', AveMoment)
print('Reversion period average:', AveRever)


# In[3]:


import pytz

# Import data and set the 'Date' Column as index
IndexPrice = pd.read_csv('HW5_Q2_data.csv', header = [0, 1])
# IndexPrice.head()
# Dropna
IndexPrice = IndexPrice.dropna(thresh = 1)

# Parse the dataframe for easier transformation
SPX = IndexPrice['SPX Index'].copy()
SPTSX = IndexPrice['SPTSX Index'].copy()
UKX = IndexPrice['UKX Index'].copy()
DAX = IndexPrice['DAX Index'].copy()
CAC = IndexPrice['CAC Index'].copy()
HSI = IndexPrice['HSI Index'].copy()
NIFTY = IndexPrice['NIFTY Index'].copy()
NKY = IndexPrice['NKY Index'].copy()

# Set index as time series and timezone
for df in (SPX, SPTSX, UKX, DAX, CAC, HSI, NIFTY, NKY):
    df.set_index('Dates', inplace = True)
    df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_localize("US/Eastern")
    
# Change timezone to local timezone
UKX.index = UKX.index.tz_convert('Europe/London')
DAX.index = DAX.index.tz_convert('Europe/Berlin')
CAC.index = CAC.index.tz_convert('Europe/Paris')
HSI.index = HSI.index.tz_convert('Asia/Hong_Kong')
NIFTY.index = NIFTY.index.tz_convert('Asia/Kolkata')
NKY.index = NKY.index.tz_convert('Asia/Tokyo')

for df in (SPX, SPTSX, UKX, DAX, CAC, HSI, NIFTY, NKY):
    df['Time'] = df.index
    df.index = df.index.date

Summary = pd.concat([SPX, SPTSX, UKX, DAX, CAC, HSI, NIFTY, NKY],
                   keys = ['SPX Index', 'SPTSX Index', 'UKX Index', 'DAX Index', 
                           'CAC Index', 'HSI Index', 'NIFTY Index', 'NKY Index'],
                   names = ['Index', 'Dates'])
# Summary.head()

# Make summary of the data
Summary = Summary.dropna()
Summary = Summary.reset_index()
Summary['Dates'] = pd.to_datetime(Summary['Dates'])
Len = Summary.groupby(['Index', 'Dates']).count()
Len = Len.rename(columns={"Close": "Daytrading Length"})
Len = Len.drop(['Time'], axis = 1)
Sta = Summary.groupby(['Index', 'Dates']).first()
Sta = Sta.rename(columns={"Time": "Start"})
Sta = Sta.drop(['Close'], axis = 1)
End = Summary.groupby(['Index', 'Dates']).last()
End = End.rename(columns={"Time": "End"})
End = End.drop(['Close'], axis = 1)
Len = Len.reset_index()
Sta = Sta.reset_index()
End = End.reset_index()
Sum = pd.merge(Sta, End, on = ['Index', 'Dates'], how = 'outer')
Sum = pd.merge(Len, Sum, on = ['Index', 'Dates'], how = 'outer')
Sum = Sum.set_index(['Dates', 'Index']).sort_index()
print(Sum.head(20))

# Find dates when not all markets are open
OpenMarket = Sum.groupby('Dates').count()
OpenMarket = Sum.resample('D', level = 0).count()
OpenMarket = OpenMarket.rename(columns = {'Daytrading Length': 'Market Count'})
DateDrop = OpenMarket.index[OpenMarket['Market Count'] < 8]

# Remove dates when not all markets are open
Summary = Summary[~Summary['Dates'].isin(DateDrop)]

# Delete days with less than 20 rows data each market
NewLen = Len.set_index('Index').sort_index()
List = NewLen['Dates'][NewLen['Daytrading Length'] < 20]
Summary = Summary[~Summary['Dates'].isin(List)]

# Keep 20 rows per open market day
Part1 = Summary.groupby(['Index', 'Dates']).head(8)
Part2 = Summary.groupby(['Index', 'Dates']).tail(12)
Summary = pd.concat([Part1, Part2])


# In[4]:


# Make summary of the clean data
# Summary = Summary.reset_index()
# Summary['Dates'] = pd.to_datetime(Summary['Dates'])
Leng = Summary.groupby(['Index', 'Dates']).count()
Leng = Leng.rename(columns={"Close": "Daytrading Length"})
Leng = Leng.drop(['Time'], axis = 1)
Star = Summary.groupby(['Index', 'Dates']).first()
Star = Star.rename(columns={"Time": "Start"})
Star = Star.drop(['Close'], axis = 1)
Last = Summary.groupby(['Index', 'Dates']).last()
Last = Last.rename(columns={"Time": "End"})
Last = Last.drop(['Close'], axis = 1)
Leng = Leng.reset_index()
Star = Star.reset_index()
Last = Last.reset_index()
Summ = pd.merge(Star, Last, on = ['Index', 'Dates'], how = 'outer')
Summ = pd.merge(Leng, Summ, on = ['Index', 'Dates'], how = 'outer')
Summ = Summ.set_index(['Dates', 'Index']).sort_index()
print(Summ.head(20))


# In[5]:


# Drop Time to keep %Y-%m-%d’ format
Summary = Summary.drop('Time', axis = 1)
# Parse the dataframe for easier transformation
SPX1 = Summary[Summary['Index'] == 'SPX Index']
SPTSX1 = Summary[Summary['Index'] == 'SPTSX Index']
UKX1 = Summary[Summary['Index'] == 'UKX Index']
DAX1 = Summary[Summary['Index'] == 'DAX Index']
CAC1 = Summary[Summary['Index'] == 'CAC Index']
HSI1 = Summary[Summary['Index'] == 'HSI Index']
NIFTY1 = Summary[Summary['Index'] == 'NIFTY Index']
NKY1 = Summary[Summary['Index'] == 'NKY Index']

for df1 in (SPX1, SPTSX1, UKX1, DAX1, CAC1, HSI1, NIFTY1, NKY1):
    df1.set_index('Dates', inplace = True)

# Concatenate dataframes for final output
Output = pd.concat([SPX1, SPTSX1, UKX1, DAX1, CAC1, HSI1, NIFTY1, NKY1],
                   keys = ['SPX Index', 'SPTSX Index', 'UKX Index', 'DAX Index', 
                           'CAC Index', 'HSI Index', 'NIFTY Index', 'NKY Index'])
Output = Output.drop('Index', axis = 1)
OP = Output.unstack(0)
# Data structure clean up
OP.columns = OP.columns.droplevel(0)
print(OP.head())


# In[6]:


# Check final output
print(OP.shape)


# In[ ]:




