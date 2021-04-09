#!/usr/bin/env python

import pandas as pd
import numpy as np

Purchase = pd.read_csv('res_purchase_2014.csv')

# Checking for errors (return rows of data needs clean)
Purchase.loc[pd.to_numeric(Purchase['Amount'], errors='coerce').isnull()]

# Clean data
Purchase = Purchase.replace({'($29.99)': -29.99, '$572.27 ': 572.27, '$12.90 ': 12.90, '452.91 zero': 452.91})
Purchase['Amount'] = pd.to_numeric(Purchase['Amount'], errors = 'coerce')

# Double check to make sure column 'Amount' is clean
# d_pur.loc[pd.to_numeric(d_pur['Amount'], errors='coerce').isnull()]

# 1. Total amount spending
t1 = Purchase['Amount'].sum()
print("Total spending:", t1)


# 2. How much was spend at WW GRAINGER?

# Check for related vendor names
colV = Purchase['Vendor'].drop_duplicates()
colV[colV.str.contains("WW GRAINGER")]
# Conditional sum
t2 = Purchase['Amount'][Purchase['Vendor'].str.contains("WW GRAINGER")].sum()
print("WW GRAINGER spending:", t2)


# 3. How much was spend at WM SUPERCENTER?

# Check for related vendor names
colV[colV.str.contains("WM SUPERCENTER")]
# Conditional sum
t3 = Purchase['Amount'][Purchase['Vendor'].str.contains("WM SUPERCENTER")].sum()
print("WM SUPERCENTER spending:", t3)


# 4. How much was spend at GROCERY STORES?

# Check related category names
colM = Purchase['Merchant Category Code (MCC)'].drop_duplicates()
colM[colM.str.contains("GROCERY STORES")]
# Conditional sum
t4 = Purchase['Amount'][Purchase['Merchant Category Code (MCC)'].str.contains("GROCERY STORES")].sum()
print("GROCERY STORES spending:", t4)


# In[3]:


# 1. Read 'Energy.xlsx' and 'EnergyRating.xlsx' as BalanceSheet and Ratings(dataframe).

BalanceSheet = pd.read_excel('Energy.xlsx')
Ratings = pd.read_excel('EnergyRating.xlsx')
# BalanceSheet.head()
# Ratings.head()
print(BalanceSheet.shape)
print(Ratings.shape)


# In[4]:


# 2. drop the column if more than 90% value in this colnmn is 0.

# Drop blank over 90% columns

BalanceSheet.dropna(axis = 1, thresh = round(len(BalanceSheet.index)* 0.1), inplace = True)
Ratings.dropna(axis = 1, thresh = round(len(Ratings.index)* 0.1), inplace = True)
# print(BalanceSheet.shape)
# print(Ratings.shape)

# Drop majority zero columns
zeros = lambda x: 1 - np.count_nonzero(x)/len(x.index)
BS_zero = BalanceSheet.apply(zeros)
BS_drop = BS_zero[BS_zero > 0.9].index.tolist()
# print(len(BS_drop))
BalanceSheet = BalanceSheet.drop(columns = BS_drop)

R_zero = Ratings.apply(zeros)
R_drop = R_zero[R_zero > 0.9].index.tolist()
# print(len(R_drop))
Ratings = Ratings.drop(columns = R_drop)

print(BalanceSheet.shape)
print(Ratings.shape)


# In[5]:


# 3. replace all None or NaN with average value of each column.

BalanceSheet = BalanceSheet.fillna(BalanceSheet.mean())
Ratings = Ratings.fillna(Ratings.mean())


# 4. Normalize the table

Norm = lambda x: (x - x.min())/(x.max() - x.min()) if str(x).isnumeric() else x
BalanceSheet = BalanceSheet.apply(Norm)
Ratings = Ratings.apply(Norm)

# 5. Calculate the correlation matrix
print(BalanceSheet[['Current Assets - Other - Total',                         'Current Assets - Total',                         'Other Long-term Assets',                         'Assets Netting & Other Adjustments']].corr())


# In[6]:


# 6. Create a new column to store the last word of company name.

from numpy import nan

ComName = lambda x: x.split(' ')[-1] if ' ' in x else nan
BalanceSheet['CO'] = BalanceSheet['Company Name'].map(ComName)

# BalanceSheet['CO'] = BalanceSheet['CO'].replace({'66': nan})
print(BalanceSheet.groupby('CO')['Company Name'].count())


# In[7]:


# 7. Merge (inner) Ratings and BalanceSheet
Matched = pd.merge(BalanceSheet, Ratings, on = ['Data Date', 'Global Company Key'])


# 8. Mapping
Map1 = pd.Series(range(13), index = ['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-',
                                 'BBB+', 'BBB','BBB-', 'BB+', 'BB', 'others'])
Matched['Rate'] = Matched['S&P Domestic Long Term Issuer Credit Rating'].map(Map1)
# Matched.loc[Matched['Rate'].isnull()]
Matched['Rate'] = Matched['Rate'].astype('Int64')
Matched.head()


# In[8]:


# 9. calculate the rating frequency of company whose name end with 'CO'.

from datetime import datetime

# format column Data Date to time stamp
Matched['Data Date'] = Matched['Data Date'].astype(str)
formtime = lambda x: datetime.strptime(x, '%Y%m%d')
Matched['Data Date'] = Matched['Data Date'].apply(formtime)

# Calculate the number of days before rating dates
Matched['Data Date Lag'] = Matched.sort_values(by = ['Company Name', 'Data Date'])                             .groupby('Company Name')['Data Date'].diff().dt.days
# Matched.head()

# Calculate the rating frequency for 'CO' companies
CO = Matched.loc[Matched['CO'] == 'CO', ['Company Name', 'Data Date Lag']]
CO.dropna(axis = 0, subset = ['Data Date Lag'], inplace = True)
Freq_CO = CO.groupby('Company Name').mean()
print(Freq_CO.rename(columns = {'Data Date Lag': 'Ave. Data Date Lag'}))


# In[9]:


# Number of recorded rating dates
print(Matched.loc[Matched['CO'] == 'CO'].groupby('Company Name')['Data Date'].count())


# In[10]:


# 10. Output final dataset.
Matched.to_csv("HW4.csv")

