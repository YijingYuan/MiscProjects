#!/usr/bin/env python3
import yfinance as yf
import datetime as dt
import pandas as pd
import numpy as np
import scipy.stats as ss
import time

# Download AMZN & SPY stock data
am = yf.download("AMZN", start="2020-01-02", end="2020-01-03")
sp = yf.download("SPY", start="2020-01-02", end="2020-01-03")

# Record interest rate for day1 and day2
r1 = 1.53 / 100
r2 = 1.52 / 100

# Download AMZN & SPY option data
# Get all option chains
am_t = yf.Ticker("AMZN")
opt1_am = am_t.option_chain('2020-02-20')
opt2_am = am_t.option_chain('2020-02-27')
opt3_am = am_t.option_chain('2020-03-05')

sp_t = yf.Ticker("SPY")
opt1_sp = sp_t.option_chain('2020-02-17')
opt2_sp = sp_t.option_chain('2020-02-18')
opt3_sp = sp_t.option_chain('2020-02-20')
opt4_sp = sp_t.option_chain('2020-02-23')
opt5_sp = sp_t.option_chain('2020-02-25')
opt6_sp = sp_t.option_chain('2020-02-27')
opt7_sp = sp_t.option_chain('2020-03-01')
opt8_sp = sp_t.option_chain('2020-03-03')
opt9_sp = sp_t.option_chain('2020-03-05')

# Download VIX options data
vx_t = yf.Ticker("^VIX")
opt1_vx = vx_t.option_chain('2020-02-18')
opt2_vx = vx_t.option_chain('2020-02-25')
opt3_vx = vx_t.option_chain('2020-03-03')

# Record stock/ETF value at time of downloading
S_am = 2134.87
S_sp = 337.60

# Seperate put and call option tables
call1_am = opt1_am.calls
call2_am = opt2_am.calls
call3_am = opt3_am.calls
put1_am = opt1_am.puts
put2_am = opt2_am.puts
put3_am = opt3_am.puts

call1_sp = opt1_sp.calls
call2_sp = opt2_sp.calls
call3_sp = opt3_sp.calls
call4_sp = opt4_sp.calls
call5_sp = opt5_sp.calls
call6_sp = opt6_sp.calls
call7_sp = opt7_sp.calls
call8_sp = opt8_sp.calls
call9_sp = opt9_sp.calls
put1_sp = opt1_sp.puts
put2_sp = opt2_sp.puts
put3_sp = opt3_sp.puts
put4_sp = opt4_sp.puts
put5_sp = opt5_sp.puts
put6_sp = opt6_sp.puts
put7_sp = opt7_sp.puts
put8_sp = opt8_sp.puts
put9_sp = opt9_sp.puts

call1_vx = opt1_vx.calls
call2_vx = opt2_vx.calls
call3_vx = opt3_vx.calls
put1_vx = opt1_vx.puts
put2_vx = opt2_vx.puts
put3_vx = opt3_vx.puts

# Concatenate all dataframes
# Chose only the same maturities for AMZN and SPY
# (for easily comparison and shorter running time)
data = pd.concat([call1_am, call2_am, call3_am, put1_am, put2_am, put3_am, \
                  call3_sp, call6_sp, call9_sp, put3_sp, put6_sp, put9_sp], ignore_index=True)

# Add stock column
data['ticker'] = data.contractSymbol.str.extract(r'([a-zA-Z]+)\d+', expand=False)
# Add expiration date column
data['expir'] = data.contractSymbol.str.extract('(\d\d\d\d\d\d)', expand=False)
# Add options type column
data['option'] = data.contractSymbol.str.extract(r'\d+([A-Z]+)\d+', expand=False)

# Data cleaning
data['bid'] = data['bid'].fillna(0)
data['ask'] = data['ask'].fillna(0)


# Create a price column based on bid and ask price
# step1: create a function for the logic
def get_price(row):
    if row['bid'] == 0 and row['ask'] > 0:
        return row['ask']
    if row['bid'] > 0 and row['ask'] > 0:
        return (row['bid'] + row['ask']) / 2
    if row['bid'] > 0 and row['ask'] == 0:
        return row['bid']


# step2: apply formula
data['price'] = data.apply(lambda row: get_price(row), axis=1)
# step3: check prices with value 0
# data.loc[data['price'] == 0]


# %%

# Check duplicated data
dup_series = data[['strike', 'expir', 'ticker', 'option']].duplicated()
data[dup_series]


# %%

# Create an option pricing class using Black-Scholes formula
class BlackScholes:

    ## European vanilla options pricing
    def __init__(self, vol, S0, time, K, r):
        self.S = S0
        self.vol = vol
        self.t = time
        self.K = K
        self.r = r
        self.d1 = (np.log(S0 / K) + (r + (vol ** 2) / 2) * time) / (vol * np.sqrt(time))
        self.d2 = self.d1 - vol * np.sqrt(time)

    # Get option price based on call/put inputs
    def optPrice(self, s):
        self.s = s
        if self.s.lower() == "call" or self.s.upper() == "C":
            Price = self.S * ss.norm.cdf(self.d1) - self.K * np.exp(-self.r * self.t) * ss.norm.cdf(self.d2)
        elif self.s.lower() == "put" or self.s.upper() == "P":
            Price = self.K * np.exp(-self.r * self.t) * ss.norm.cdf(-self.d2) - self.S * ss.norm.cdf(-self.d1)
        else:
            print('error in input format')
        return Price

    # Greeks calculation
    def delta(self):
        return ss.norm.cdf(self.d1)

    def gamma(self):
        return (np.exp(-self.d1 ** self.d1 / 2) / np.sqrt(2 * np.pi)) / (self.S * self.vol * np.sqrt(self.t))

    def vega(self):
        return self.S * np.sqrt(self.t) * np.exp(-self.d1 ** self.d1 / 2) / np.sqrt(2 * np.pi)


# %%

# Get number of rows in the dataframe
rowmax = data.shape[0]
# Add new columns for approximated implied volatility
data['volImp_bi'] = np.nan
data['volImp_new'] = np.nan


# Add maturity column
def get_t(row):
    td = dt.datetime.strptime(row['expir'], '%y%m%d') - dt.datetime(2019, 12, 17)
    t = td.days / 365
    return t


data['time_to_expir'] = data.apply(lambda row: get_t(row), axis=1)

# Bisection method to find implied volatility
t1_bi = time.time()  # time recording start

for i in range(0, rowmax):
    if data.iloc[i]['ticker'] == 'AMZN':
        S01 = am.iloc[0]['Close']
    elif data.iloc[i]['ticker'] == 'SPY':
        S01 = sp.iloc[0]['Close']
    t = data.iloc[i]['time_to_expir']
    K = data.iloc[i]['strike']
    r = r1
    pri = data.iloc[i]['ask']  ## change later
    o = data.iloc[i]['option']

    # Using Black-Scholes formula
    a = 0
    b = 5
    while (abs(a - b) > 10 ** (-6)):
        BS1 = BlackScholes(a, S01, t, K, r)
        BS2 = BlackScholes(b, S01, t, K, r)
        BS3 = BlackScholes((a + b) / 2, S01, t, K, r)
        f1 = BS1.optPrice(o) - pri
        f2 = BS2.optPrice(o) - pri
        f3 = BS3.optPrice(o) - pri
        if f1 * f3 < 0:
            b = (a + b) / 2
        elif f2 * f3 < 0:
            a = (a + b) / 2
        else:
            break
    data.at[i, 'volImp_bi'] = a

t2_bi = time.time()  # time recording end
dt_bi = t2_bi - t1_bi
print(dt_bi, 'secs')

# %%

# Newton method to find implied volatility
# Using Black-Scholes formula
rowmax = data.shape[0]
t1_new = time.time()

for i in range(0, rowmax):
    if data.iloc[i]['ticker'] == 'AMZN':
        S01 = am.iloc[0]['Close']
    elif data.iloc[i]['ticker'] == 'SPY':
        S01 = sp.iloc[0]['Close']
    t = data.iloc[i]['time_to_expir']
    K = data.iloc[i]['strike']
    r = r1
    pri = data.iloc[i]['ask']  ## change later
    o = data.iloc[i]['option']

    x = 1
    x_old = 0
    count = 0
    while (abs(x - x_old) > 10 ** (-6)):
        count = count + 1
        x_old = x
        BS_new = BlackScholes(x, S01, t, K, r)
        f_new = BS_new.optPrice(o) - pri
        if BS_new.d1 < 10 ** (-2):
            BS_new.d1 = 0.0
        df_new = S01 * np.sqrt(t) * np.exp(-BS_new.d1 ** BS_new.d1 / 2) / np.sqrt(2 * np.pi)
        x = x - f_new / df_new
        # Stop calculation once reach 1000 steps for each root
        if count > 10 ** 3:
            x = np.nan
            break
    data.at[i, 'volImp_new'] = x

t2_new = time.time()
dt_new = t2_new - t1_new
print(dt_new, 'secs')

# %%

# Check erred out Newton method implied volatility value
data.loc[data['volImp_new'].isnull()]

# Check erred out bisection method implied volatility value
d1 = data.loc[data['volImp_new'].isnull()]
d1 = d1.loc[d1['volImp_bi'] != 0]

# %%

# Keep only relevant columns for aggregation
summ = data[['strike', 'ticker', 'time_to_expir', 'option', 'volImp_bi']]

# Average implied volatility by stock, maturity and option type
# Bisection volatility used
vol_avg = summ.groupby(['ticker', 'time_to_expir', 'option'])['volImp_bi'].mean()
print(vol_avg)

# %%

# Plot implied volatility vs strike
import matplotlib.pyplot as plt
#% matplotlib
#inline

# Plot implied volatility vs strike - expiration data (2020-02-21)
df_am = data[(data['expir'] == '200221') & (data['ticker'] == 'AMZN')][['strike', 'volImp_bi', 'option']]
df_sp = data[(data['expir'] == '200221') & (data['ticker'] == 'SPY')][['strike', 'volImp_bi', 'option']]

l1, index1 = np.unique(df_am["option"], return_inverse=True)
l2, index2 = np.unique(df_sp["option"], return_inverse=True)

fig1, ax1 = plt.subplots()
sc1 = ax1.scatter(df_am['strike'], df_am['volImp_bi'], c=index1)
ax1.legend(sc1.legend_elements()[0], l1)
ax1.set_title('AMZN - Implied Volatility vs Strike')
plt.show()

fig2, ax2 = plt.subplots()
sc2 = ax2.scatter(df_sp['strike'], df_sp['volImp_bi'], c=index2)
ax2.legend(sc2.legend_elements()[0], l2)
ax2.set_title('SPY - Implied Volatility vs Strike')
plt.show()

# Plot implied volatility vs strike (all three maturities)
df_amc = data[(data['option'] == 'C') & (data['ticker'] == 'AMZN')][['strike', 'volImp_bi', 'expir']]
df_amp = data[(data['option'] == 'P') & (data['ticker'] == 'AMZN')][['strike', 'volImp_bi', 'expir']]
df_spc = data[(data['option'] == 'C') & (data['ticker'] == 'SPY')][['strike', 'volImp_bi', 'expir']]
df_spp = data[(data['option'] == 'P') & (data['ticker'] == 'SPY')][['strike', 'volImp_bi', 'expir']]

l3, index3 = np.unique(df_amc["expir"], return_inverse=True)
l4, index4 = np.unique(df_spc["expir"], return_inverse=True)
l5, index5 = np.unique(df_amp["expir"], return_inverse=True)
l6, index6 = np.unique(df_spp["expir"], return_inverse=True)

fig3, ax3 = plt.subplots()
sc3 = ax3.scatter(df_amc['strike'], df_amc['volImp_bi'], c=index3)
ax3.legend(sc3.legend_elements()[0], l3)
ax3.set_title('AMZN Call - Implied Volatility vs Strike')
plt.show()

fig4, ax4 = plt.subplots()
sc4 = ax4.scatter(df_spc['strike'], df_spc['volImp_bi'], c=index4)
ax4.legend(sc4.legend_elements()[0], l4)
ax4.set_title('SPY Call - Implied Volatility vs Strike')
plt.show()

fig5, ax5 = plt.subplots()
sc5 = ax5.scatter(df_amp['strike'], df_amp['volImp_bi'], c=index5)
ax5.legend(sc5.legend_elements()[0], l5)
ax5.set_title('AMZN Put - Implied Volatility vs Strike')
plt.show()

fig6, ax6 = plt.subplots()
sc6 = ax6.scatter(df_spp['strike'], df_spp['volImp_bi'], c=index6)
ax6.legend(sc6.legend_elements()[0], l6)
ax6.set_title('SPY Put - Implied Volatility vs Strike')
plt.show()

# %%

# (BONUS) Make 3D plots of implied volatility vs strike vs maturity
from mpl_toolkits.mplot3d import Axes3D

df1 = data[data['ticker'] == 'AMZN'][['strike', 'volImp_bi', 'time_to_expir', 'option']]
df2 = data[data['ticker'] == 'SPY'][['strike', 'volImp_bi', 'time_to_expir', 'option']]

fig7 = plt.figure()
ax7 = fig7.add_subplot(projection='3d')
l7, index7 = np.unique(df1["option"], return_inverse=True)
sc7 = ax7.scatter(df1['strike'], df1['volImp_bi'], df1['time_to_expir'], c=index7)
ax7.legend(sc7.legend_elements()[0], l7)
ax7.set_title('AMZN - Implied Volatility vs Strike vs Maturity')
plt.show()

fig8 = plt.figure()
ax8 = fig8.add_subplot(projection='3d')
l8, index8 = np.unique(df2["option"], return_inverse=True)
sc8 = ax8.scatter(df2['strike'], df2['volImp_bi'], df2['time_to_expir'], c=index8)
ax8.legend(sc8.legend_elements()[0], l8)
ax8.set_title('SPY - Implied Volatility vs Strike vs Maturity')
plt.show()

# %%

# Put-Call Parity
# Define put-call parity formula
# Use recorded stock/ETF price today at the time of downloading
S_am = 2134.87
S_sp = 337.60


# Interest rate today is 1.59% (as of 2020-02-16)

def put_call_par_bid(row):
    pri = row['bid']
    if pri == 0:
        return np.nan
    else:
        td = dt.datetime.strptime(row['expir'], '%y%m%d') - dt.datetime(2020, 2, 16)
        t = td.days / 365
        r = 0.0159
        K = row['strike']
        if row['option'] == 'C':
            if row['ticker'] == 'AMZN':
                return pri + K * np.exp(-r * t) - S_am
            if row['ticker'] == 'SPY':
                return pri + K * np.exp(-r * t) - S_sp
        if row['option'] == 'P':
            if row['ticker'] == 'AMZN':
                return pri - K * np.exp(-r * t) + S_am
            if row['ticker'] == 'SPY':
                return pri - K * np.exp(-r * t) + S_sp


def put_call_par_ask(row):
    pri = row['ask']
    if pri == 0:
        return np.nan
    else:
        td = dt.datetime.strptime(row['expir'], '%y%m%d') - dt.datetime(2020, 2, 16)
        t = td.days / 365
        r = 0.0159
        K = row['strike']
        if row['option'] == 'C':
            if row['ticker'] == 'AMZN':
                return pri + K * np.exp(-r * t) - S_am
            if row['ticker'] == 'SPY':
                return pri + K * np.exp(-r * t) - S_sp
        if row['option'] == 'P':
            if row['ticker'] == 'AMZN':
                return pri - K * np.exp(-r * t) + S_am
            if row['ticker'] == 'SPY':
                return pri - K * np.exp(-r * t) + S_sp


# %%

# Add bid/ask price using put-call parity
data['par_bid'] = data.apply(lambda row: put_call_par_bid(row), axis=1)
data['par_ask'] = data.apply(lambda row: put_call_par_ask(row), axis=1)

# %%

# new = summ.pivot_table('volImp_bi', ['option', 'strike'], ['ticker','time_to_expir'])

# %%

# Create a new dataframe with only call options
calls = data.loc[data['option'] == 'C'].copy()


# Calculate Geeks
# Black-Scholes formula
def get_delta(row):
    if data.iloc[i]['ticker'] == 'AMZN':
        S0 = am.iloc[0]['Close']
    elif data.iloc[i]['ticker'] == 'SPY':
        S0 = sp.iloc[0]['Close']
    t = row['time_to_expir']
    r = r1
    K = row['strike']
    vol = row['volImp_bi']
    BS = BlackScholes(vol, S0, t, K, r)
    return BS.delta()


def get_gamma(row):
    if data.iloc[i]['ticker'] == 'AMZN':
        S0 = am.iloc[0]['Close']
    elif data.iloc[i]['ticker'] == 'SPY':
        S0 = sp.iloc[0]['Close']
    t = row['time_to_expir']
    r = r1
    K = row['strike']
    vol = row['volImp_bi']
    BS = BlackScholes(vol, S0, t, K, r)
    return BS.gamma()


def get_vega(row):
    if data.iloc[i]['ticker'] == 'AMZN':
        S0 = am.iloc[0]['Close']
    elif data.iloc[i]['ticker'] == 'SPY':
        S0 = sp.iloc[0]['Close']
    t = row['time_to_expir']
    r = r1
    K = row['strike']
    vol = row['volImp_bi']
    BS = BlackScholes(vol, S0, t, K, r)
    return BS.vega()


# Apply functions to dataframe
calls['Delta_BS'] = calls.apply(lambda row: get_delta(row), axis=1)
calls['Gamma_BS'] = calls.apply(lambda row: get_gamma(row), axis=1)
calls['Vega_BS'] = calls.apply(lambda row: get_vega(row), axis=1)


# %%

# Calculate Geeks
# Partial derivative approximation
# Delta
def delta_appr(row):
    if row['ticker'] == 'AMZN':
        S0 = am.iloc[0]['Close']
    elif row['ticker'] == 'SPY':
        S0 = sp.iloc[0]['Close']
    h = 0.01  # assuming stock/ETF go up by 1%
    ds = h * S0
    t = row['time_to_expir']
    r = r1  # use DATA1 interest rate
    K = row['strike']
    vol = row['volImp_bi']
    o = row['option']
    BS1 = BlackScholes(vol, S0, t, K, r)
    BS2 = BlackScholes(vol, S0 * (1 + h), t, K, r)
    return (BS2.optPrice(o) - BS1.optPrice(o)) / ds


def gamma_appr(row):
    if row['ticker'] == 'AMZN':
        S0 = am.iloc[0]['Close']
    elif row['ticker'] == 'SPY':
        S0 = sp.iloc[0]['Close']
    h = 0.01  # assuming stock/ETF go up by 1%
    ds = h * S0
    t = row['time_to_expir']
    r = r1  # use DATA1 interest rate
    K = row['strike']
    vol = row['volImp_bi']
    o = row['option']
    BS1 = BlackScholes(vol, S0, t, K, r)
    BS2 = BlackScholes(vol, S0 * (1 + h), t, K, r)
    BS3 = BlackScholes(vol, S0 * (1 - h), t, K, r)
    return (BS2.optPrice(o) + BS3.optPrice(o) - 2 * BS1.optPrice(o)) / (ds * ds)


def vega_appr(row):
    if row['ticker'] == 'AMZN':
        S0 = am.iloc[0]['Close']
    elif row['ticker'] == 'SPY':
        S0 = sp.iloc[0]['Close']
    dv = 0.01
    t = row['time_to_expir']
    r = r1  # use DATA1 interest rate
    K = row['strike']
    vol = row['volImp_bi']
    o = row['option']
    BS1 = BlackScholes(vol, S0, t, K, r)
    BS2 = BlackScholes(vol + dv, S0, t, K, r)
    return (BS2.optPrice(o) - BS1.optPrice(o)) / dv


# Apply approximation formulas
calls['Delta_PD'] = calls.apply(lambda row: delta_appr(row), axis=1)
calls['Gamma_PD'] = calls.apply(lambda row: gamma_appr(row), axis=1)
calls['Vega_PD'] = calls.apply(lambda row: vega_appr(row), axis=1)

calls


# %%

# Calculate day2 option price using DATA2
def get_price(row):
    if data.iloc[i]['ticker'] == 'AMZN':
        S0 = am.iloc[1]['Close']  # DATA2 underlying price
    elif data.iloc[i]['ticker'] == 'SPY':
        S0 = sp.iloc[1]['Close']
    t = row['time_to_expir']
    r = r2  # use DATA2 corresponding interest rate
    K = row['strike']
    vol = row['volImp_bi']
    o = row['option']
    BS = BlackScholes(vol, S0, t, K, r)
    return BS.optPrice(o)


data['price_day2'] = data.apply(lambda row: get_price(row), axis=1)


# %%

## PART 3.

def function1(x):
    if x == 0:
        return 0
    else:
        return np.sin(x) / x


a = 10 ** 6
N = 10 ** 8
h = 2 * a / N
I = h / 2 * (function1(-a) + function1(a))

for n in range(1, N):
    x = -a + n * h
    I = I + h * function1(x)


# %%

def Integral(a, N):
    h = 2 * a / N
    I = h / 2 * (function1(-a) + function1(a))
    for n in range(1, N):
        x = -a + n * h
        I = I + h * function1(x)
    return I


err1 = np.repeat(np.nan, 10)
for i in range(1, 11):
    a = 10 ** 6 * i
    N = 10 ** 7
    err1[i - 1] = abs(Integral(a, N) - np.pi)
    print(err1[i - 1])

# %%

err2 = np.repeat(np.nan, 10)
for i in range(1, 11):
    a = 10 ** 6
    N = 10 ** 7 * i
    err2[i - 1] = abs(Integral(a, N) - np.pi)
    print(err2[i - 1])

print(err2)

# %%

err2 = np.repeat(np.nan, 10)
for i in range(1, 11):
    a = 10 ** 6 * i
    N = 10 ** 7 * (i * 2)
    err2[i - 1] = abs(Integral(a, N) - np.pi)
    print(err2[i - 1])

print(err2)

# %%

a = 10 ** 6
N = 10 ** 7
I1 = 2
I2 = 1
k = 0
while (abs(I1 - I2) > 10 ** (-4)):
    k = k + 1
    I1 = Integral(a, N)
    N = N * 10
    I2 = Integral(a, N)
    print(N)
print(k)

# %%

## PART 4.
import scipy.integrate as si

I1 = si.dblquad(lambda y, x: x * y, 0, 1, lambda x: 0, lambda x: 3)
I2 = si.dblquad(lambda y, x: np.exp(x + y), 0, 1, lambda x: 0, lambda x: 3)
Ia_xy = I1[0]
Ia_exp = I2[0]
print(Ia_xy)
print(Ia_exp)

f_xy = lambda a, b: a * b
f_exp = lambda a, b: np.exp(a + b)


# Approximate the double integrals using the Trapezoidal Rule
def f_xy_integral(n, m):
    dx = 1 / n
    dy = 3 / m
    I_dbl = 0

    for i in range(0, n):
        for j in range(0, m):
            x1 = dx * i
            x2 = dx * (i + 1)
            y1 = dy * j
            y2 = dy * (j + 1)
            I_dbl = I_dbl + dx * dy / 16 * (f_xy(x1, y1) + f_xy(x1, y2) + f_xy(x2, y1) + f_xy(x2, y2) + \
                                            2 * (f_xy((x1 + x2) / 2, y1) + f_xy((x1 + x2) / 2, y2) + \
                                                 f_xy(x1, (y1 + y2) / 2) + f_xy(x2, (y1 + y2) / 2)) + \
                                            4 * f_xy((x1 + x2) / 2, (y1 + y2) / 2))
    return I_dbl


def f_exp_integral(n, m):
    dx = 1 / n
    dy = 3 / m
    I_dbl = 0

    for i in range(0, n):
        for j in range(0, m):
            x1 = dx * i
            x2 = dx * (i + 1)
            y1 = dy * j
            y2 = dy * (j + 1)
            I_dbl = I_dbl + dx * dy / 16 * (f_exp(x1, y1) + f_exp(x1, y2) + f_exp(x2, y1) + f_exp(x2, y2) + \
                                            2 * (f_exp((x1 + x2) / 2, y1) + f_exp((x1 + x2) / 2, y2) + \
                                                 f_exp(x1, (y1 + y2) / 2) + f_exp(x2, (y1 + y2) / 2)) + \
                                            4 * f_exp((x1 + x2) / 2, (y1 + y2) / 2))
    return I_dbl


print(f_xy_integral(10, 30), 'pair:', 1 / 10, 3 / 30)
print(f_xy_integral(100, 30), 'pair:', 1 / 100, 3 / 30)
print(f_xy_integral(10, 300), 'pair:', 1 / 10, 3 / 300)
print(f_xy_integral(100, 300), 'pair:', 1 / 100, 3 / 300)

# %%

print(f_exp_integral(10, 30), 'pair:', 1 / 10, 3 / 30)
print(f_exp_integral(10, 300), 'pair:', 1 / 10, 3 / 300)
print(f_exp_integral(100, 30), 'pair:', 1 / 100, 3 / 30)
print(f_exp_integral(100, 300), 'pair:', 1 / 100, 3 / 300)

# %%


