#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Problem 1 - Binomial Tree

# Step1: get data 
import yfinance as yf
import datetime as dt
import pandas as pd
import numpy as np
import scipy.stats as ss
import time

# load and process the data using codes from HW1 to get implied volatility
# Record interest rate for 1, 2, 3-month
r1 = 0.91/100
r3 = 0.61/100
# Use linear interpolation to get 2-month interest rate
r2 = (r1+r3)/2

# Download AMZN option data
# NOTE: AMZN is a non-dividend paying stock
# Get 1,2,3-month maturity option chains
am_t = yf.Ticker("AMZN")
opt1_am = am_t.option_chain('2020-04-08')
opt2_am = am_t.option_chain('2020-05-14')
opt3_am = am_t.option_chain('2020-06-18')

# Record stock/ETF value at time of downloading
S_am = 1901.09

# Seperate put and call option tables
call1_am = opt1_am.calls
call2_am = opt2_am.calls
call3_am = opt3_am.calls
put1_am = opt1_am.puts
put2_am = opt2_am.puts
put3_am = opt3_am.puts

# Concatenate all dataframes
data = pd.concat([call1_am, call2_am, call3_am, put1_am, put2_am, put3_am], ignore_index=True)

# Add ticker column
data['ticker'] = data.contractSymbol.str.extract(r'([a-zA-Z]+)\d+', expand=False)
# Add expiration date column
# Used to calculate time to maturity later
data['expir'] = data.contractSymbol.str.extract('(\d\d\d\d\d\d)', expand=False)
# Add options type column
data['option'] = data.contractSymbol.str.extract(r'\d+([A-Z]+)\d+', expand=False)

# Add a price column based on bid and ask price
data['bid'] = data['bid'].fillna(0)
data['ask'] = data['ask'].fillna(0)

def get_price(row):
    if row['bid'] == 0 and row['ask'] > 0:
        return row['ask']
    if row['bid'] > 0 and row['ask'] > 0:
        return (row['bid'] + row['ask'])/2
    if row['bid'] > 0 and row['ask'] == 0:
        return row['bid']

data['price'] = data.apply(lambda row: get_price(row), axis=1)

# Get rid of options with zero bid/ask prices
data.dropna(subset=['price'], inplace=True)
data.reset_index(inplace=True, drop=True)


# In[2]:


# Add maturity column (in years)
def get_t(row):
    td = dt.datetime.strptime(row['expir'], '%y%m%d') - dt.datetime(2020, 3, 4)
    t = td.days / 365
    return t
data['time_to_expir'] = data.apply(lambda row: get_t(row), axis=1)

# Add a column to identify closeness to stock price
def get_closeness(row):
    K = row['strike']
    diff = abs(K-S_am)
    return diff
data['diff'] = data.apply(lambda row: get_closeness(row), axis=1)

# Get 20 strike prices close to stock value
data = data.sort_values(["option", "time_to_expir", "diff"], ascending = (True, True, True))
data = data.groupby(["option", "time_to_expir"]).head(20)
data.reset_index(inplace=True, drop=True)
data = data.drop(columns=['diff'])


# In[9]:


class BlackScholes:  
    ## European vanilla options pricing
    def __init__(self, vol, S0, T, K, r, q):
        self.S = S0
        self.vol = vol
        self.t = T
        self.K = K
        self.r = r
        self.q = q
        self.d1 = (np.log(S0/K) + (r-q+(vol**2)/2)*T) / (vol*np.sqrt(T))
        self.d2 = self.d1 - vol*np.sqrt(T)
    
    # Get option price based on call/put inputs
    def optPrice(self, o):
        self.o = o
        if self.o.lower() == "call" or self.o.upper() == "C":
            Price = self.S*np.exp(-self.q*self.t)*ss.norm.cdf(self.d1) - self.K*np.exp(-self.r*self.t)*ss.norm.cdf(self.d2)
        elif self.o.lower() == "put" or self.o.upper() == "P":
            Price = self.K*np.exp(-self.r*self.t)*ss.norm.cdf(-self.d2) - self.S*np.exp(-self.q*self.t)*ss.norm.cdf(-self.d1)
        else:
            print('error in input format')
        return Price


# In[13]:


# Get number of rows in the dataframe
rowmax = data.shape[0]
# Get the implied volatilities
for i in range(0, rowmax):
    S0 = S_am
    t = data.iloc[i]['time_to_expir']
    K = data.iloc[i]['strike']
    if t < 0.10:
        r = r1    #1-month risk-free interest rate
    elif t < 0.20:
        r = r2    #2-month risk-free interest rate
    elif t < 0.30:
        r = r3    #3-month risk-free interest rate
    pri = data.iloc[i]['price']  
    o = data.iloc[i]['option']
    q = 0
    
    # Bisection Method to find roots using Black-Scholes
    a = 0
    b = 5
    while (abs(a-b) > 10**(-6)):
        BS1 = BlackScholes(a, S0, t, K, r, q)
        BS2 = BlackScholes(b, S0, t, K, r, q)
        BS3 = BlackScholes((a+b)/2, S0, t, K, r, q)
        f1 = BS1.optPrice(o) - pri
        f2 = BS2.optPrice(o) - pri
        f3 = BS3.optPrice(o) - pri
        if f1*f3 < 0:
            b = (a+b)/2
        elif f2*f3 < 0:
            a = (a+b)/2
        else:
            break 
    data.at[i,'impVol'] = a


# In[14]:


# Problem 1 - a)
# Binomial Tree Method - Additive Tree
# Trigeorgis tree

class Binomial:  
    ## European vanilla options pricing
    def __init__(self, S0, N, sigma, T, K, r, opt, div):
        self.S = S0
        self.N = N
        self.sigma = sigma
        self.T = T
        self.K = K
        self.r = r
        self.dt = T/N
        self.nu = r - div - (sigma**2)/2
        self.dxu = np.sqrt((self.nu*self.dt)**2 + (sigma**2)*self.dt)
        self.dxd = -self.dxu
        self.pu = 0.5 + 0.5*self.nu*self.dt/self.dxu
        self.pd = 1 - self.pu
        self.disc = np.exp(-r*self.dt)
        self.opt = opt

    def BinomialEuro(self):
        S_t = np.repeat(np.nan, self.N+1)
        pri = np.repeat(np.nan, self.N+1)
        S_t[0] = self.S * np.exp(self.N*self.dxd)
        for j in range(1, self.N+1):
            S_t[j] = S_t[j-1] * np.exp(self.dxu - self.dxd)
        for j in range(0, self.N+1):
            if self.opt == 'C':
                pri[j] = max(0, S_t[j] - self.K)
            if self.opt == 'P':
                pri[j] = max(0, self.K - S_t[j])
        for t in range(self.N, 0, -1):
            for i in range(0, t):
                pri[i] = self.disc*(self.pu*pri[i+1] + self.pd*pri[i])
        return pri[0]
    
    def BinomialAmeri(self):
        S_t = np.repeat(np.nan, self.N+1)
        pri = np.repeat(np.nan, self.N+1)
        S_t[0] = self.S * np.exp(self.N*self.dxd)
        for j in range(1, self.N+1):
            S_t[j] = S_t[j-1] * np.exp(self.dxu - self.dxd)
        for j in range(0, self.N+1):
            if self.opt == 'C':
                pri[j] = max(0, S_t[j] - self.K)
            if self.opt == 'P':
                pri[j] = max(0, self.K - S_t[j])
        for t in range(self.N, 0, -1):
            for i in range(0, t):
                    pri[i] = self.disc*(self.pu*pri[i+1] + self.pd*pri[i])
                    S_t[i] = S_t[i] / np.exp(self.dxd)
                    if self.opt == 'C':
                        pri[i] = max(pri[i], S_t[i] - self.K)
                    if self.opt == 'P':
                        pri[i] = max(pri[i], self.K - S_t[i])
        return pri[0]


# In[15]:


# Problem 1 - b)

S_0 = S_am
N = 300
div = 0

for i in range(0, rowmax):
    t = data.iloc[i]['time_to_expir']
    strike = data.iloc[i]['strike']
    vol = data.iloc[i]['impVol']
    if t < 0.10:
        rate = r1    #1-month risk-free interest rate
    elif t < 0.20:
        rate = r2    #2-month risk-free interest rate
    elif t < 0.30:
        rate = r3    #3-month risk-free interest rate
    o = data.iloc[i]['option']
    
    # apply binomial tree method
    pri = Binomial(S_0, N, vol, t, strike, rate, o, div)
    data.at[i,'EuroPri_bi'] = pri.BinomialEuro()
    data.at[i,'AmeriPri_bi'] = pri.BinomialAmeri()


# In[16]:


# Problem 1 - c)
table = data[['strike', 'option', 'expir', 'time_to_expir', 'price', 'EuroPri_bi', 'AmeriPri_bi']]
t1c = table[(table['option'] == 'C') & (table['expir'] == '200409')]
t1p = table[(table['option'] == 'P') & (table['expir'] == '200409')]
t2c = table[(table['option'] == 'C') & (table['expir'] == '200515')]
t2p = table[(table['option'] == 'P') & (table['expir'] == '200515')]
t3c = table[(table['option'] == 'C') & (table['expir'] == '200619')]
t3p = table[(table['option'] == 'P') & (table['expir'] == '200619')]
print('Table example: \n')
print(t1c, '\n')
print(t1p)
# Note: price is the average of bid-ask and used as the BS-price


# In[115]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

fig1, ax1 = plt.subplots()
ax1.scatter(t1c['strike'], t1c['price'], label='BS')
ax1.scatter(t1c['strike'], t1c['EuroPri_bi'], label='BiTree_Euro')
ax1.scatter(t1c['strike'], t1c['AmeriPri_bi'], label='BiTree_Ameri')
ax1.legend()
ax1.set_title('1-month Call')

fig2, ax2 = plt.subplots()
ax2.scatter(t1p['strike'], t1p['price'], label='BS')
ax2.scatter(t1p['strike'], t1p['EuroPri_bi'], label='BiTree_Euro')
ax2.scatter(t1p['strike'], t1p['AmeriPri_bi'], label='BiTree_Ameri')
ax2.legend()
ax2.set_title('1-month Put')

fig3, ax3 = plt.subplots()
ax3.scatter(t2c['strike'], t2c['price'], label='BS')
ax3.scatter(t2c['strike'], t2c['EuroPri_bi'], label='BiTree_Euro')
ax3.scatter(t2c['strike'], t2c['AmeriPri_bi'], label='BiTree_Ameri')
ax3.legend()
ax3.set_title('2-month Call')

fig4, ax4 = plt.subplots()
ax4.scatter(t2p['strike'], t2p['price'], label='BS')
ax4.scatter(t2p['strike'], t2p['EuroPri_bi'], label='BiTree_Euro')
ax4.scatter(t2p['strike'], t2p['AmeriPri_bi'], label='BiTree_Ameri')
ax4.legend()
ax4.set_title('2-month Put')

fig5, ax5 = plt.subplots()
ax5.scatter(t3c['strike'], t3c['price'], label='BS')
ax5.scatter(t3c['strike'], t3c['EuroPri_bi'], label='BiTree_Euro')
ax5.scatter(t3c['strike'], t3c['AmeriPri_bi'], label='BiTree_Ameri')
ax5.legend()
ax5.set_title('3-month Call')

fig6, ax6 = plt.subplots()
ax6.scatter(t3p['strike'], t3p['price'], label='BS')
ax6.scatter(t3p['strike'], t3p['EuroPri_bi'], label='BiTree_Euro')
ax6.scatter(t3p['strike'], t3p['AmeriPri_bi'], label='BiTree_Ameri')
ax6.legend()
ax6.set_title('3-month Put')

plt.show()

# Comment:
# 1. Observed from the plots, we could clearly see that the BS price 
# and the binomial tree prices are fairly close. The 3-month option
# shows a more noticible difference between BS model and Binomial Tree
# model than the 1,2-month options.
# 2. The European price and the American price are almost identical 
# in the call options. This agrees with that American options should
# never be exercised early in the absence of dividends.
# 3. It's observed that the difference in the American vs European
# put prices are very small in 1-month and 2-month maturities and
# the American option are slightly more expensive than European options
# The difference is more pronounced in the 3-month maturity. 
# This agrees with the fact that American options carries the time value,
# which makes it more expensive than European options.
# When the options are closer to maturity, this time value gets smaller.


# In[31]:


# Problem 1 - d)
# Creat a dataframe to store results
df1 = pd.DataFrame(np.nan, index=range(15), columns=[])
step = [10, 20, 30, 40, 50, 100, 150, 200, 250, 300, 350, 400, 600, 800, 1000]

# Use the data from the downloaded AMZN data table        
# (Use the put option on row 66 as an example)
S_0 = S_am
div = 0
t = data.iloc[66]['time_to_expir']
strike = data.iloc[66]['strike']
vol = data.iloc[66]['impVol']
if t < 0.10:
    rate = r1    #1-month risk-free interest rate
elif t < 0.20:
    rate = r2    #2-month risk-free interest rate
elif t < 0.30:
    rate = r3    #3-month risk-free interest rate
o = 'P'
    
for i in range(0, 15):
    p = Binomial(S_0, step[i], vol, t, strike, rate, o, div)
    df1.at[i,'steps'] = step[i]
    df1.at[i,'BiTree_Euro'] = p.BinomialEuro()
    df1.at[i,'BS'] = BlackScholes(vol, S0, t, strike, rate, div).optPrice(o)
    df1.at[i,'error'] = abs(df1.at[i,'BiTree_Euro']-df1.at[i,'BS'])

df1['steps'] = df1['steps'].astype(int)
print(df1, '\n')
df1.plot(x='steps', y='error', kind='line')

# Comment: 
# The results obtained from the binomial tree method converge to 
# the Black-Scholes model price as the number of steps increases.
# However, the convergence trend line is not entirely smooth.
# It shows some small zig-zag pattern as it converges, especially
# in the smaller range of number of steps.


# In[32]:


# Problem 2 - Implied Volatility

# Use Binomial tree to find the implied volatility for American option 
S_0 = S_am
N = 300
div = 0

for i in range(0, rowmax):
    t = data.iloc[i]['time_to_expir']
    strike = data.iloc[i]['strike']
    pri = data.iloc[i]['AmeriPri_bi']
    if t < 0.10:
        rate = r1    #1-month risk-free interest rate
    elif t < 0.20:
        rate = r2    #2-month risk-free interest rate
    elif t < 0.30:
        rate = r3    #3-month risk-free interest rate
    o = data.iloc[i]['option']
    
    # Bisection Method to find roots
    a = 0
    b = 5
    while (abs(a-b) > 10**(-6)):
        bi1 = Binomial(S_0, N, a, t, strike, rate, o, div)
        bi2 = Binomial(S_0, N, b, t, strike, rate, o, div)
        bi3 = Binomial(S_0, N, (a+b)/2, t, strike, rate, o, div)
        dif1 = bi1.BinomialEuro() - pri
        dif2 = bi2.BinomialEuro() - pri
        dif3 = bi3.BinomialEuro() - pri
        if dif1*dif3 < 0:
            b = (a+b)/2
        elif dif2*dif3 < 0:
            a = (a+b)/2
        else:
            break 
    data.at[i,'volImp_Ameri'] = a
    
data

# Comment: 
# For call options the implied volatilities obtained for American
# and European options are almost identical (differ by about 0.0000001).
# For put options, the implied volatilities obtained for American
# options appears to be slight higher (about 0.1%) than European options.


# In[33]:


# Problem 3 - Trinomial Tree

class Trinomial:  
    ## European vanilla options pricing
    def __init__(self, S0, N, sigma, T, K, r, opt, div):
        self.S = S0
        self.N = N
        self.sigma = sigma
        self.T = T
        self.K = K
        self.r = r
        self.dt = T/N
        self.div = div
        self.nu = r - div - (sigma**2)/2
        self.dxu = np.sqrt((self.nu*self.dt)**2 + (sigma**2)*self.dt)
        self.pu = 0.5 * (((sigma**2)*self.dt+(self.nu*self.dt)**2)/(self.dxu**2)                         + self.nu*self.dt/self.dxu)
        self.pd = 0.5 * (((sigma**2)*self.dt+(self.nu*self.dt)**2)/(self.dxu**2)                         - self.nu*self.dt/self.dxu)
        self.pm = 1 - self.pu - self.pd
        self.disc = np.exp(-r*self.dt)
        self.opt = opt
        try:
            self.dxu >= sigma*np.sqrt(3*self.dt)
        except:
            raise ValueError('dx seed must satisfy stability condition')

    def TrinomialEuro(self):
        S_t = np.repeat(np.nan, 2*self.N+1)
        opt = np.empty((self.N+1, 2*self.N+1))
        S_t[0] = self.S * np.exp(-self.N*self.dxu)
        for j in range(1, 2*self.N+1):
            S_t[j] = S_t[j-1] * np.exp(self.dxu)
        for j in range(0, 2*self.N+1):
            if self.opt == 'C':
                opt[self.N, j] = max(0, S_t[j] - self.K)
            if self.opt == 'P':
                opt[self.N, j] = max(0, self.K - S_t[j])
        for t in range(self.N, 0, -1):
            for i in range(0, 2*t-1):
                    opt[t-1, i] = self.disc*(self.pu*opt[t, i+2] + self.pm*opt[t, i+1] + self.pd*opt[t, i])
        return opt[0, 0]
    
    def TrinomialAmeri(self):
        S_t = np.repeat(np.nan, 2*self.N+1)
        opt = np.empty((self.N+1, 2*self.N+1))
        S_t[0] = self.S * np.exp(-self.N*self.dxu)
        for j in range(1, 2*self.N+1):
            S_t[j] = S_t[j-1] * np.exp(self.dxu)
        for j in range(0, 2*self.N+1):
            if self.opt == 'C':
                opt[self.N, j] = max(0, S_t[j] - self.K)
            if self.opt == 'P':
                opt[self.N, j] = max(0, self.K - S_t[j])
        for t in range(self.N, 0, -1):
            for i in range(0, 2*t-1):
                    opt[t-1, i] = self.disc*(self.pu*opt[t, i+2] + self.pm*opt[t, i+1] + self.pd*opt[t, i])
                    S_t[i] = S_t[i] * np.exp(self.dxu)
                    if self.opt == 'C':
                        opt[t-1, i] = max(opt[t-1, i], S_t[i] - self.K)
                    if self.opt == 'P':
                        opt[t-1, i] = max(opt[t-1, i], self.K - S_t[i])
        return opt[0, 0]
    


# In[34]:


# Check the trinomial tree method with the data
S0 = S_am
dv = 0

for i in range(0, rowmax): 
    t = data.iloc[i]['time_to_expir']
    strike = data.iloc[i]['strike']
    pri = data.iloc[i]['AmeriPri_bi']
    vol = data.iloc[i]['impVol']
    if t < 0.10:
        rate = r1    #1-month risk-free interest rate
    elif t < 0.20:
        rate = r2    #2-month risk-free interest rate
    elif t < 0.30:
        rate = r3    #3-month risk-free interest rate
    o = data.iloc[i]['option']

    # apply trinomial tree method
    N = 300
    pri = Trinomial(S0, N, vol, t, strike, rate, o, dv)
    data.at[i,'EuroPri_tri'] = pri.TrinomialEuro()
    data.at[i,'AmeriPri_tri'] = pri.TrinomialAmeri()
data


# In[35]:


# Problem 3 - b)
Tri_Put = Trinomial(S0 = 100, N = 300, sigma=0.2, T=1, K=100, r=0.05, opt='P', div=0.02)
Tri_EuroPut = Tri_Put.TrinomialEuro()
Tri_AmeriPut = Tri_Put.TrinomialAmeri()
print("Trinomial European Put:", Tri_EuroPut)
print("Trinomial American Put:", Tri_AmeriPut)

Tri_Call = Trinomial(S0 = 100, N = 300, sigma=0.2, T=1, K=100, r=0.05, opt='C', div=0.02)
Tri_EuroCall = Tri_Call.TrinomialEuro()
Tri_AmeriCall = Tri_Call.TrinomialAmeri()
print("Trinomial European Call:", Tri_EuroCall)
print("Trinomial American Call:", Tri_AmeriCall)


# In[114]:


# Show convergence
# Creat a dataframe to store results
df2 = pd.DataFrame(np.nan, index=range(15), columns=[])
step = [10, 20, 30, 40, 50, 100, 150, 200, 250, 300, 350, 400, 600, 800, 1000]

# Use European put    
for i in range(0, 15):
    p = Binomial(S0=100, N=step[i], sigma=0.2, T=1, K=100, r=0.05, opt='P', div=0.02)
    df2.at[i,'steps'] = step[i]
    df2.at[i,'BiTree_Euro'] = p.BinomialEuro()
    df2.at[i,'BS'] = BlackScholes(vol=0.2, S0=100, T=1, K=100, r=0.05, q=0.02).optPrice(o='P')
    df2.at[i,'error'] = abs(df2.at[i,'BiTree_Euro']-df2.at[i,'BS'])

df2['steps'] = df2['steps'].astype(int)
print(df2, '\n')
df2.plot(x='steps', y='error', kind='line')
# Comment: 
# The results obtained from the trinomial tree method converge to 
# the Black-Scholes model price as the number of steps increases.
# Compared to the binomial tree method, trinomial tree has a 
# much smoother trend line and converges faster.


# In[36]:


# Problem 4 - Barrier option
# Additive Binomial Tree
class BarrierTree:  
    def __init__(self, S0, N, sigma, T, K, r, div, H):
        self.S = S0
        self.N = N
        self.sigma = sigma
        self.T = T
        self.K = K
        self.r = r
        self.dt = T/N
        self.H = H
        self.nu = r-div-(sigma**2)/2
        self.dxu = np.sqrt((self.nu*self.dt)**2 + (sigma**2)*self.dt)
        self.dxd = -self.dxu
        self.pu = 0.5 + 0.5*self.nu*self.dt/self.dxu
        self.pd = 1 - self.pu
        self.disc = np.exp(-r*self.dt)

    def E_UpOutCall(self):
        S_t = np.repeat(np.nan, self.N+1)
        pri = np.repeat(np.nan, self.N+1)
        S_t[0] = self.S * np.exp(self.N*self.dxd)
        for j in range(1, self.N+1):
            S_t[j] = S_t[j-1] * np.exp(self.dxu-self.dxd)
        for j in range(0, self.N+1):
            if S_t[j] < self.H:
                pri[j] = max(0, S_t[j] - self.K)
            else:
                pri[j] = 0    
        for t in range(self.N, 0, -1):
            for i in range(0, t):
                S_t[i] = S_t[i] / np.exp(self.dxd)
                if S_t[i] < self.H:
                    pri[i] = self.disc*(self.pu*pri[i+1] + self.pd*pri[i])
                else:
                    pri[i] = 0
        return pri[0]
    
    def E_UpInCall(self):
        S_t = np.repeat(np.nan, self.N+1)
        pri = np.repeat(np.nan, self.N+1)
        S_t[0] = self.S * np.exp(self.N*self.dxd)
        for j in range(1, self.N+1):
            S_t[j] = S_t[j-1] * np.exp(self.dxu-self.dxd)
        for j in range(0, self.N+1):
            if S_t[j] > self.H:
                pri[j] = max(0, S_t[j] - self.K)
            else:
                pri[j] = 0    
        for t in range(self.N, 0, -1):
            for i in range(0, t):
                S_t[i] = S_t[i] / np.exp(self.dxd)
                if S_t[i] > self.H:
                    pri[i] = self.disc*(self.pu*pri[i+1] + self.pd*pri[i])
                else:
                    pri[i] = 0  
        return pri[0]
    
    def A_UpInPut(self):
        S_t = np.repeat(np.nan, self.N+1)
        pri = np.repeat(np.nan, self.N+1)
        S_t[0] = self.S * np.exp(self.N*self.dxd)
        for j in range(1, self.N+1):
            S_t[j] = S_t[j-1] * np.exp(self.dxu-self.dxd)
        for j in range(0, self.N+1):   
            if S_t[j] > self.H:
                pri[j] = max(0, self.K - S_t[j])
            else:
                pri[j] = 0
        for t in range(self.N, 0, -1):
            for i in range(0, t):
                S_t[i] = S_t[i] / np.exp(self.dxd)
                if S_t[i] > self.H:
                    pri[i] = self.disc*(self.pu*pri[i+1] + self.pd*pri[i])
                    pri[i] = max(pri[i], self.K - S_t[i])
                else:
                    pri[i] = 0
        return pri[0]


# In[37]:


price = BarrierTree(S0=10, N=3302, sigma=0.2, T=0.3, K=10, r=0.01, div=0, H=11)
p1 = price.E_UpOutCall()
print("European Up-and-Out Call:", p1)


# In[38]:


class BarrierBS:
    def __init__(self, vol, S0, T, K, r, q, H):
        self.S = S0
        self.vol = vol
        self.t = T
        self.K = K
        self.r = r
        self.q = q
        self.H = H
        self.d1 = (np.log(S0/K) + (r-q+(vol**2)/2)*T) / (vol*np.sqrt(T))
        self.d2 = self.d1 - vol*np.sqrt(T)
        self.nu = r-q-(vol**2)/2
        # dBS1 = d_BS(S, H)
        self.dBS1 = (np.log(S0/H) + self.nu*T) / (vol*np.sqrt(T))
        # dBS2 = d_BS(H, S)
        self.dBS2 = (np.log(H/S0) + self.nu*T) / (vol*np.sqrt(T))
    
    def UpOut_Call(self):
        # C1 = C_BS(S, K)
        C1 = BlackScholes(self.vol, self.S, self.t, self.K, self.r, self.q).optPrice(o="C")
        # C2 = C_BS(S, H)
        C2 = BlackScholes(self.vol, self.S, self.t, self.H, self.r, self.q).optPrice(o="C")
        # C3 = C_BS(H*H/S, K)
        C3 = BlackScholes(self.vol, (self.H**2)/self.S, self.t, self.K, self.r, self.q).optPrice(o="C")
        # C4 = C_BS(H*H/S, H)
        C4 = BlackScholes(self.vol, (self.H**2)/self.S, self.t, self.H, self.r, self.q).optPrice(o="C")
        UO = C1 - C2 - (self.H-self.K)*np.exp(-self.r*self.t)*ss.norm.cdf(self.dBS1)            - ((self.H/self.S)**(2*self.nu/(self.vol**2))) *                 (C3 - C4 - (self.H-self.K)*np.exp(-self.r*self.t)*ss.norm.cdf(self.dBS2))
        return UO
    
    def UpIn_Call(self):
        # C2 = C_BS(S, H)
        C2 = BlackScholes(self.vol, self.S, self.t, self.H, self.r, self.q).optPrice(o="C")
        # P3 = P_BS(H*H/S, K)
        P3 = BlackScholes(self.vol, (self.H**2)/self.S, self.t, self.K, self.r, self.q).optPrice(o="P")
        # P4 = P_BS(H*H/S, H)
        P4 = BlackScholes(self.vol, (self.H**2)/self.S, self.t, self.H, self.r, self.q).optPrice(o="P")
        UI_call = ((self.H/self.S)**(2*self.nu/(self.vol**2))) *             (P3 - P4 + (self.H-self.K)*np.exp(-self.r*self.t)*ss.norm.cdf(-self.dBS2))            + C2 + (self.H-self.K)*np.exp(-self.r*self.t)*ss.norm.cdf(self.dBS1)
        return UI_call


# In[39]:


B = BarrierBS(vol=0.2, S0=10, T=0.3, K=10, r=0.01, q=0, H=11)
UO_C = B.UpOut_Call()
UI_C = B.UpIn_Call()
print("Up-and-Out European Call:", UO_C)
print("Up-and-In European Call:", UI_C)


# In[40]:


# Use In-Out Parity
BS1 = BlackScholes(vol=0.2, S0=10, T=0.3, K=10, r=0.01, q=0)
C_BS = BS1.optPrice(o='C')
print("Up-and-In European Call (In-Out Parity):", C_BS - UO_C)


# In[41]:


# American option
aput = BarrierTree(S0=10, N=500, sigma=0.2, T=0.3, K=10, r=0.01, div=0, H=11)
ap = aput.A_UpInPut()
print("Up-and-In American Put:", ap)


# In[99]:


# Problem 5
# Exact code from HW1 BUT changed a small part of function1
def function1(x):
    if x == 0:
        return 1 # This value was set to 0 in HW1 while it should be 1
    else:
        return np.sin(x)/x

# Below was the same code I used for trapezoidal rule in HW1
a = 10**6
N = 10**7
h = 2*a/N
I = h/2*(function1(-a)+function1(a))

for n in range(1, N):
    x = -a + n*h
    I = I + h*function1(x)
    
print(I)


# In[109]:


# Trapezoidal Rule - write in formula 
def Trapz(a, N):
    I = h/2*(function1(-a)+function1(a))
    for n in range(1, N):
        x = -a + n*h
        I = I + h*function1(x)
    return I

# Simpon's Rule - didn't do in HW1
def Simp(a, N):
    h = 2*a/N
    f1 = 0
    f2 = 0
    for j in range(1, int(N/2)):
        f1 = f1 + 4 * function1(-a + 2*j*h)
    for j in range(1, int(N/2)+1):
        f2 = f2 + 2 * function1(-a + (2*j-1)*h)   
        I = (function1(-a) + f1 + f2 + function1(a)) * h/3
    return I
print(Simp(10**6, 10**7))


# In[116]:


# Step calculation - incomplete in HW1
k = 0
inc = 10**6
err = 0
while(err < 10**(-4)):
    I1 = Trapz(10**6, 10**6 + k*inc)
    I2 = Trapz(10**6, 10**6 + (k+1)*inc)
    err = abs(I1 - I2)
    k = k + 1
print("Trapezoidal rule steps:", k)

inc1 = 10**5
while(err < 10**(-4)):
    I1 = Simps(10**6, 10**6 + k*inc1)
    I2 = Simps(10**6, 10**6 + (k+1)*inc1)
    err = abs(I1 - I2)
    k = k + 1
print("Simpon's rule steps:", k)

# Comment:
# The number of steps taken to converge depends on the starting point
# and the increments in N. That means if the starting point is fairly
# large (h = 2*a/N is small), then it won't take too many steps to reach
# the condition of converge. However, if we picked a smaller increment 
# and a smaller starting point for N, it will take more steps to converge
# and longer to compute.
# In the code above, the steps a large increment of N was chosen and it
# took only 4 steps to reach converge at N=5*(10**7). However with the same
# amount of steps but smaller increments, the Simpon's rule converges at
# N = 6.5*(10**6), a much smaller number of steps.


# In[ ]:




