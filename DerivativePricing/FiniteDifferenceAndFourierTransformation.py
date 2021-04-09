#!/usr/bin/env python3
# coding: utf-8

# In[2]:


# Problem 1 - (a)(b)(c)

import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import pandas as pd
import scipy.fftpack as sft
from scipy.interpolate import interp1d
from scipy import signal

class FiniteDiff:  
    ## European vanilla options pricing
    def __init__(self, S0, sigma, Nj, dx, dt, T, K, r, div):
        self.S = S0
        self.Nj = Nj
        self.sigma = sigma
        self.T = T
        self.K = K
        self.r = r
        self.dt = dt
        self.N = int(T/dt)
        self.dx = dx
        self.nu = r - div - (sigma**2)/2
        self.edx = np.exp(dx)
        self.disc = np.exp(-r*self.dt)
        self.St = np.repeat(np.nan, 2*self.Nj+1)
        self.St[0] = S0*np.exp(-Nj*dx)
        # Get the EFD European call values to calculate the greeks
        self.EFD_c = np.empty((self.N+1, 2*self.Nj+1))
        for j in range(1, 2*self.Nj+1):
            self.St[j] = self.St[j-1]*np.exp(dx)
        # Check boundaries
        #print("Upper:", S0*np.exp(Nj*dx))
        #print("Lower:", S0*np.exp(-Nj*dx))

    def Explicit(self, opt):
        # Check convergence condition
        if (self.dx < self.sigma*np.sqrt(3*self.dt)):
            print("condition:", self.sigma*np.sqrt(3*self.dt))
            print("dx:", dx)
            raise ValueError('dx seed must satisfy stability condition')
        pri = np.empty((self.N+1, 2*self.Nj+1))
        pu = 0.5*self.dt*((self.sigma/self.dx)**2 + self.nu/self.dx)
        pd = 0.5*self.dt*((self.sigma/self.dx)**2 - self.nu/self.dx)
        pm = 1 - self.dt*((self.sigma/self.dx)**2) - self.r*self.dt
        for j in range(0, 2*self.Nj+1):
            if opt == "C" or opt == "Call":
                pri[self.N, j] = max(0, self.St[j] - self.K)
            if opt == "P" or opt == "Put":
                pri[self.N, j] = max(0, self.K - self.St[j])
        for i in range(self.N-1, -1, -1):
            for j in range(1, 2*self.Nj):
                pri[i, j] = pu*pri[i+1, j+1] + pm*pri[i+1, j] + pd*pri[i+1, j-1]
            # boundary conditions
            if opt == "C" or opt == "Call":                                                                      
                pri[i, 0] = pri[i, 1]
                pri[i, 2*self.Nj] = pri[i, 2*self.Nj-1] +                                     (self.St[2*self.Nj] - self.St[2*self.Nj-1])
            if opt == "P" or opt == "Put": 
                pri[i, 2*self.Nj] = pri[i, 2*self.Nj-1]
                pri[i, 0] = pri[i, 1] + (self.St[1] - self.St[0]) 
        if opt == "C" or opt == "Call":
            self.EFD_c = pri
        return pri[0, self.Nj]
        
    def Implicit(self, opt):
        pu = -0.5*self.dt*((self.sigma/self.dx)**2 + self.nu/self.dx)
        pd = -0.5*self.dt*((self.sigma/self.dx)**2 - self.nu/self.dx)
        pm = 1 + self.dt*((self.sigma/self.dx)**2) + self.r*self.dt
        pri = np.empty((self.N+1, 2*self.Nj+1))
        for j in range(0, 2*self.Nj+1):
            if opt == "C" or opt == "Call":
                pri[self.N, j] = max(0, self.St[j] - self.K)
            if opt == "P" or opt == "Put":
                pri[self.N, j] = max(0, self.K - self.St[j])
        # set the price boundary
        if opt == "P" or opt == "Put":
            L = -1 * (self.St[1] - self.St[0])
            U = 0.0
        if opt == "C" or opt == "Call": 
            U = self.St[2*self.Nj] - self.St[2*self.Nj-1]
            L = 0.0
        # solve the implicit tridiagonal system
        pmp = np.repeat(np.nan, 2*self.Nj)
        pp = np.repeat(np.nan, 2*self.Nj)
        for i in range(self.N-1, -1, -1):
            # boundary condition
            pmp[1] = pm + pd
            pp[1] = pri[i+1, 1] + pd*L
            # non-boundary
            for j in range(2, 2*self.Nj):
                pmp[j] = pm - pu*pd/pmp[j-1]
                pp[j] = pri[i+1, j] - pp[j-1]*pd/pmp[j-1]
            # boundary condition
            pri[i, 2*self.Nj] = (pp[2*self.Nj-1] + pmp[2*self.Nj-1]*U)/(pu + pmp[2*self.Nj-1])
            pri[i, 2*self.Nj-1] = pri[i, 2*self.Nj] - U
            # back-substitution
            for j in range(2*self.Nj-2, 0, -1):
                pri[i, j] = (pp[j] - pu*pri[i, j+1])/pmp[j]
                #pri[i,j] = max(pri[i,j], self.K - self.St[j])
            pri[i, 0] = pri[i, 1] - L
            #pri[i,0] = max(pri[i,0], self.K - self.St[0])
        return pri[0, self.Nj]
    
    def Crank(self, opt):
        pu = -0.25*self.dt*((self.sigma/self.dx)**2 + self.nu/self.dx)
        pd = -0.25*self.dt*((self.sigma/self.dx)**2 - self.nu/self.dx)
        pm = 1 + 0.5*self.dt*((self.sigma/self.dx)**2) + 0.5*self.r*self.dt
        pri = np.empty((self.N+1, 2*self.Nj+1))
        for j in range(0, 2*self.Nj+1):
            if opt == "C" or opt == "Call":
                pri[self.N, j] = max(0, self.St[j] - self.K)
            if opt == "P" or opt == "Put":
                pri[self.N, j] = max(0, self.K - self.St[j])
        # set the price boundary
        if opt == "P" or opt == "Put":
            L = -1 * (self.St[1] - self.St[0])
            U = 0.0
        if opt == "C" or opt == "Call": 
            U = self.St[2*self.Nj] - self.St[2*self.Nj-1]
            L = 0.0
        # solve the implicit tridiagonal system
        pmp = np.repeat(np.nan, 2*self.Nj)
        pp = np.repeat(np.nan, 2*self.Nj)
        for i in range(self.N-1, -1, -1):
            # boundary condition
            pmp[1] = pm + pd
            pp[1] = -pu*pri[i+1, 2] - (pm-2)*pri[i+1, 1] - pd*pri[i+1, 0] + pd*L
            # non-boundary
            for j in range(2, 2*self.Nj):
                pmp[j] = pm - pu*pd/pmp[j-1]
                pp[j] = -pu*pri[i+1, j+1] - (pm-2)*pri[i+1, j] - pd*pri[i+1, j-1] - pp[j-1]*pd/pmp[j-1]
            # boundary condition
            pri[i, 2*self.Nj] = (pp[2*self.Nj-1] + pmp[2*self.Nj-1]*U)/(pu + pmp[2*self.Nj-1])
            pri[i, 2*self.Nj-1] = pri[i, 2*self.Nj] - U
            # back-substitution
            for j in range(2*self.Nj-2, 0, -1):
                pri[i, j] = (pp[j] - pu*pri[i, j+1])/pmp[j]
                #pri[i,j] = max(pri[i,j], self.K - self.St[j])
            pri[i, 0] = pri[i, 1] - L
            #pri[i,0] = max(pri[i,0], self.K - self.St[0])
        return pri[0, self.Nj]


# In[4]:


# Problem 1 - (d)(e)

# Set up dt, dx scheme
size = 10
Nl = np.repeat(np.nan, size)
dtl = np.repeat(np.nan, size)
dxl = np.repeat(np.nan, size)
dxl2 = np.repeat(np.nan, size)
up = np.repeat(np.nan, size)
low = np.repeat(np.nan, size)
sigma = 0.25
Nj_choose = 200
S0 = 100
for i in range(0, size):
    Nl[i] = 100*(i+1)
    dtl[i] = 1/Nl[i]
    dxl[i] = sigma*np.sqrt(3*dtl[i])
    up[i] = S0*np.exp(Nj_choose*dxl[i])
    low[i] = S0*np.exp(-Nj_choose*dxl[i])
    
# Check boundary values to varify Nj is sufficiently large
#print(up, low) 
#print(dxl, dxl2)


# In[5]:


# Output results
# EFD, IFD, CNFD all follows the same (dt, dx) schemes

v1 = 0
v2 = 1
k = 0
while (abs(v2-v1) > 0.001):
    FD1 = FiniteDiff(S0=100, sigma=0.25, dt=dtl[k], Nj=Nj_choose, dx=dxl[k], T=1, K=100, r=0.06, div=0.03)
    v1 = FD1.Explicit(opt='Call')
    FD2 = FiniteDiff(S0=100, sigma=0.25, dt=dtl[k+1], Nj=Nj_choose, dx=dxl[k+1], T=1, K=100, r=0.06, div=0.03)
    v2 = FD2.Explicit(opt='Call')
    k = k + 1
    if k >= 20:
        break
print("Explicit Call Number of Steps:", k+1)
print("Explicit Call Convergence Value:", v1, v2)
print("Convergence N, dt, dx:", int(Nl[k]), dtl[k], dxl[k], '\n')

v1 = 0
v2 = 1
k = 0
while (abs(v2-v1) > 0.001):
    FD1 = FiniteDiff(S0=100, sigma=0.25, dt=dtl[k], Nj=Nj_choose, dx=dxl[k], T=1, K=100, r=0.06, div=0.03)
    v1 = FD1.Explicit(opt='Put')
    FD2 = FiniteDiff(S0=100, sigma=0.25, dt=dtl[k+1], Nj=Nj_choose, dx=dxl[k+1], T=1, K=100, r=0.06, div=0.03)
    v2 = FD2.Explicit(opt='Put')
    k = k + 1
    if k >= 20:
        break
print("Explicit Put Number of Steps:", k+1)
print("Explicit Put Convergence Value:", v1, v2)
print("Convergence N, dt, dx:", int(Nl[k]), dtl[k], dxl[k], '\n')

v1 = 0
v2 = 1
k = 0
while (abs(v2-v1) > 0.001):
    FD1 = FiniteDiff(S0=100, sigma=0.25, dt=dtl[k], Nj=Nj_choose, dx=dxl[k], T=1, K=100, r=0.06, div=0.03)
    v1 = FD1.Implicit(opt='Call')
    FD2 = FiniteDiff(S0=100, sigma=0.25, dt=dtl[k+1], Nj=Nj_choose, dx=dxl[k+1], T=1, K=100, r=0.06, div=0.03)
    v2 = FD2.Implicit(opt='Call')
    k = k + 1
    if k >= 20:
        break
print("Implicit Call Number of Steps:", k+1)
print("Implicit Call Convergence Value:", v1, v2)
print("Convergence N, dt, dx:", int(Nl[k]), dtl[k], dxl[k], '\n')

v1 = 0
v2 = 1
k = 0
while (abs(v2-v1) > 0.001):
    FD1 = FiniteDiff(S0=100, sigma=0.25, dt=dtl[k], Nj=Nj_choose, dx=dxl[k], T=1, K=100, r=0.06, div=0.03)
    v1 = FD1.Implicit(opt='Put')
    FD2 = FiniteDiff(S0=100, sigma=0.25, dt=dtl[k+1], Nj=Nj_choose, dx=dxl[k+1], T=1, K=100, r=0.06, div=0.03)
    v2 = FD2.Implicit(opt='Put')
    k = k + 1
    if k >= 20:
        break
print("Implicit Put Number of Steps:", k+1)
print("Implicit Put Convergence Value:", v1, v2)
print("Convergence N, dt, dx:", int(Nl[k]), dtl[k], dxl[k], '\n')

v1 = 0
v2 = 1
k = 0
while (abs(v2-v1) > 0.001):
    FD1 = FiniteDiff(S0=100, sigma=0.25, dt=dtl[k], Nj=Nj_choose, dx=dxl[k], T=1, K=100, r=0.06, div=0.03)
    v1 = FD1.Crank(opt='Call')
    FD2 = FiniteDiff(S0=100, sigma=0.25, dt=dtl[k+1], Nj=Nj_choose, dx=dxl[k+1], T=1, K=100, r=0.06, div=0.03)
    v2 = FD2.Crank(opt='Call')
    k = k + 1
    if k >= 20:
        break
print("Crank-Nicolson Call Number of Steps:", k+1)
print("Crank-Nicolson Call Convergence Value:", v1, v2)
print("Convergence N, dt, dx:", int(Nl[k]), dtl[k], dxl[k], '\n')

v1 = 0
v2 = 1
k = 0
while (abs(v2-v1) > 0.001):
    FD1 = FiniteDiff(S0=100, sigma=0.25, dt=dtl[k], Nj=Nj_choose, dx=dxl[k], T=1, K=100, r=0.06, div=0.03)
    v1 = FD1.Crank(opt='Put')
    FD2 = FiniteDiff(S0=100, sigma=0.25, dt=dtl[k+1], Nj=Nj_choose, dx=dxl[k+1], T=1, K=100, r=0.06, div=0.03)
    v2 = FD2.Crank(opt='Put')
    k = k + 1
    if k >= 20:
        break
print("Crank-Nicolson Put Number of Steps:", k+1)
print("Crank-Nicolson Put Convergence Value:", v1, v2)
print("Convergence N, dt, dx:", int(Nl[k]), dtl[k], dxl[k])


# In[6]:


# Output results
# IFD, CNFD follows a different dx scheme with higher reduction rate

size = 20
Nl = np.repeat(np.nan, size)
dtl = np.repeat(np.nan, size)
dxl = np.repeat(np.nan, size)
dxl2 = np.repeat(np.nan, size)
up = np.repeat(np.nan, size)
low = np.repeat(np.nan, size)
sigma = 0.25
Nj_choose = 300
S0 = 100
for i in range(0, size):
    Nl[i] = 50*(i+1)
    dtl[i] = 1/Nl[i]
    dxl[i] = sigma*np.sqrt(3*dtl[i])*2
    dxl2[i] = dxl[i]*0.2

v1 = 0
v2 = 1
k = 0
while (abs(v2-v1) > 0.001):
    FD1 = FiniteDiff(S0=100, sigma=0.25, dt=dtl[k], Nj=Nj_choose, dx=dxl[k], T=1, K=100, r=0.06, div=0.03)
    v1 = FD1.Explicit(opt='Call')
    FD2 = FiniteDiff(S0=100, sigma=0.25, dt=dtl[k+1], Nj=Nj_choose, dx=dxl[k+1], T=1, K=100, r=0.06, div=0.03)
    v2 = FD2.Explicit(opt='Call')
    k = k + 1
    if k >= 20:
        break
print("Explicit Call Number of Steps:", k+1)
print("Explicit Call Convergence Value:", v1, v2)
print("Convergence N, dt, dx:", int(Nl[k]), dtl[k], dxl[k], '\n')

v1 = 0
v2 = 1
k = 0
while (abs(v2-v1) > 0.001):
    FD1 = FiniteDiff(S0=100, sigma=0.25, dt=dtl[k], Nj=Nj_choose, dx=dxl[k], T=1, K=100, r=0.06, div=0.03)
    v1 = FD1.Explicit(opt='Put')
    FD2 = FiniteDiff(S0=100, sigma=0.25, dt=dtl[k+1], Nj=Nj_choose, dx=dxl[k+1], T=1, K=100, r=0.06, div=0.03)
    v2 = FD2.Explicit(opt='Put')
    k = k + 1
    if k >= 20:
        break
print("Explicit Put Number of Steps:", k+1)
print("Explicit Put Convergence Value:", v1, v2)
print("Convergence N, dt, dx:", int(Nl[k]), dtl[k], dxl[k], '\n')

v1 = 0
v2 = 1
k = 0
while (abs(v2-v1) > 0.001):
    FD1 = FiniteDiff(S0=100, sigma=0.25, dt=dtl[k], Nj=Nj_choose, dx=dxl2[k], T=1, K=100, r=0.06, div=0.03)
    v1 = FD1.Implicit(opt='Call')
    FD2 = FiniteDiff(S0=100, sigma=0.25, dt=dtl[k+1], Nj=Nj_choose, dx=dxl2[k+1], T=1, K=100, r=0.06, div=0.03)
    v2 = FD2.Implicit(opt='Call')
    k = k + 1
    if k >= 20:
        break
print("Implicit Call Number of Steps:", k+1)
print("Implicit Call Convergence Value:", v1, v2)
print("Convergence N, dt, dx:", int(Nl[k]), dtl[k], dxl2[k], '\n')

v1 = 0
v2 = 1
k = 0
while (abs(v2-v1) > 0.001):
    FD1 = FiniteDiff(S0=100, sigma=0.25, dt=dtl[k], Nj=Nj_choose, dx=dxl2[k], T=1, K=100, r=0.06, div=0.03)
    v1 = FD1.Implicit(opt='Put')
    FD2 = FiniteDiff(S0=100, sigma=0.25, dt=dtl[k+1], Nj=Nj_choose, dx=dxl2[k+1], T=1, K=100, r=0.06, div=0.03)
    v2 = FD2.Implicit(opt='Put')
    k = k + 1
    if k >= 20:
        break
print("Implicit Put Number of Steps:", k+1)
print("Implicit Put Convergence Value:", v1, v2)
print("Convergence N, dt, dx:", int(Nl[k]), dtl[k], dxl2[k], '\n')

v1 = 0
v2 = 1
k = 0
while (abs(v2-v1) > 0.001):
    FD1 = FiniteDiff(S0=100, sigma=0.25, dt=dtl[k], Nj=Nj_choose, dx=dxl2[k], T=1, K=100, r=0.06, div=0.03)
    v1 = FD1.Crank(opt='Call')
    FD2 = FiniteDiff(S0=100, sigma=0.25, dt=dtl[k+1], Nj=Nj_choose, dx=dxl2[k+1], T=1, K=100, r=0.06, div=0.03)
    v2 = FD2.Crank(opt='Call')
    k = k + 1
    if k >= 20:
        break
print("Crank-Nicolson Call Number of Steps:", k+1)
print("Crank-Nicolson Call Convergence Value:", v1, v2)
print("Convergence N, dt, dx:", int(Nl[k]), dtl[k], dxl2[k], '\n')

v1 = 0
v2 = 1
k = 0
while (abs(v2-v1) > 0.001):
    FD1 = FiniteDiff(S0=100, sigma=0.25, dt=dtl[k], Nj=Nj_choose, dx=dxl2[k], T=1, K=100, r=0.06, div=0.03)
    v1 = FD1.Crank(opt='Put')
    FD2 = FiniteDiff(S0=100, sigma=0.25, dt=dtl[k+1], Nj=Nj_choose, dx=dxl2[k+1], T=1, K=100, r=0.06, div=0.03)
    v2 = FD2.Crank(opt='Put')
    k = k + 1
    if k >= 20:
        break
print("Crank-Nicolson Put Number of Steps:", k+1)
print("Crank-Nicolson Put Convergence Value:", v1, v2)
print("Convergence N, dt, dx:", int(Nl[k]), dtl[k], dxl2[k])

## Comment:
# 1. When EFD, IFD and CNFD follows the same (dt, dx) scheme,
#   IFD and CNFD doesn't have the clear speed advantage over EFD.
#   In fact, both IFD and CNFD are slower, while CNFD is still
#   slightly faster than IFD.
# 2. However, when IFD and EFD follows a different dx scheme than,
#   EFD, we can see that the IFD converges faster than EFD, and
#   CNFD converges even faster than IFD. 
# 3. The above observation can be explained that, IFD and CNFD are
#   not restricted to the convergence and stability condition, thus
#   dx can be reduced at a faster rate and we can achieve convergence
#   way faster than EFD. Also, CNFD is a refinement of IFD so it will 
#   be faster than both EFD and IFD.


# In[7]:


# Problem 1 - (f)
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


# In[11]:


# Problem 1 - (f) & (h)

# Calculate BS option prices
BS = BlackScholes(vol=0.25, S0=100, T=1, K=100, r=0.06, q=0.03)
BS_call = BS.optPrice(o='Call')
BS_put = BS.optPrice(o='put')

# Set up dt, dx scheme
size = 40
Nl = np.repeat(np.nan, size)
dtl = np.repeat(np.nan, size)
dxl = np.repeat(np.nan, size)
dxl2 = np.repeat(np.nan, size)
up = np.repeat(np.nan, size)
low = np.repeat(np.nan, size)
sigma = 0.25
Nj_choose = 300
S0 = 100
for i in range(0, size):
    Nl[i] = 50*(i+1)
    dtl[i] = 1/Nl[i]
    dxl[i] = sigma*np.sqrt(3*dtl[i])*2
    dxl2[i] = dxl[i]*0.2

# Create a dataframe to store the data
df1 = pd.DataFrame(columns=['Number of Steps', 'Price'],
                  index=["EFD_call", "EFD_put", "IFD_call", "IFD_put", "CNFD_call", "CNFD_put"])

k = 0
v1 = 0
while (abs(BS_call-v1) > 0.001):
    FD1 = FiniteDiff(S0=100, sigma=0.25, dt=dtl[k], Nj=Nj_choose, dx=dxl[k], T=1, K=100, r=0.06, div=0.03)
    v1 = FD1.Explicit(opt='Call')
    k = k + 1
    if k >= 20:
        break
df1.at["EFD_call", "Number of Steps"] = k
df1.at["EFD_call", "Price"] = v1
df1.at["EFD_call", "N"] = Nl[k-1]
df1.at["EFD_call", "dt"] = FD1.dt
df1.at["EFD_call", "dx"] = FD1.dx
df1.at["EFD_call", "diff"] = abs(v1 - BS_call)

k = 0
v1 = 0
while (abs(BS_put-v1) > 0.001):
    FD1 = FiniteDiff(S0=100, sigma=0.25, dt=dtl[k], Nj=Nj_choose, dx=dxl[k], T=1, K=100, r=0.06, div=0.03)
    v1 = FD1.Explicit(opt='Put')
    k = k + 1
    if k >= 20:
        break
df1.at["EFD_put", "Number of Steps"] = k
df1.at["EFD_put", "Price"] = v1
df1.at["EFD_put", "N"] = Nl[k-1]
df1.at["EFD_put", "dt"] = FD1.dt
df1.at["EFD_put", "dx"] = FD1.dx
df1.at["EFD_put", "diff"] = abs(v1 - BS_put)

k = 0
v1 = 0
while (abs(BS_call-v1) > 0.001):
    FD1 = FiniteDiff(S0=100, sigma=0.25, dt=dtl[k], Nj=Nj_choose, dx=dxl2[k], T=1, K=100, r=0.06, div=0.03)
    v1 = FD1.Implicit(opt='Call')
    k = k + 1
    if k >= 20:
        break
df1.at["IFD_call", "Number of Steps"] = k
df1.at["IFD_call", "Price"] = v1
df1.at["IFD_call", "N"] = Nl[k-1]
df1.at["IFD_call", "dt"] = FD1.dt
df1.at["IFD_call", "dx"] = FD1.dx
df1.at["IFD_call", "diff"] = abs(v1 - BS_call)

k = 0
v1 = 0
while (abs(BS_put-v1) > 0.001):
    FD1 = FiniteDiff(S0=100, sigma=0.25, dt=dtl[k], Nj=Nj_choose, dx=dxl2[k], T=1, K=100, r=0.06, div=0.03)
    v1 = FD1.Implicit(opt='Put')
    k = k + 1
    if k >= 20:
        break
df1.at["IFD_put", "Number of Steps"] = k
df1.at["IFD_put", "Price"] = v1
df1.at["IFD_put", "N"] = Nl[k-1]
df1.at["IFD_put", "dt"] = FD1.dt
df1.at["IFD_put", "dx"] = FD1.dx
df1.at["IFD_put", "diff"] = abs(v1 - BS_put)

k = 0
v1 = 0
while (abs(BS_call-v1) > 0.001):
    FD1 = FiniteDiff(S0=100, sigma=0.25, dt=dtl[k], Nj=Nj_choose, dx=dxl2[k], T=1, K=100, r=0.06, div=0.03)
    v1 = FD1.Crank(opt='Call')
    k = k + 1
    if k >= 20:
        break
df1.at["CNFD_call", "Number of Steps"] = k
df1.at["CNFD_call", "Price"] = v1
df1.at["CNFD_call", "N"] = Nl[k-1]
df1.at["CNFD_call", "dt"] = FD1.dt
df1.at["CNFD_call", "dx"] = FD1.dx
df1.at["CNFD_call", "diff"] = abs(v1 - BS_call)

k = 0
v1 = 0
while (abs(BS_put-v1) > 0.001):
    FD1 = FiniteDiff(S0=100, sigma=0.25, dt=dtl[k], Nj=Nj_choose, dx=dxl2[k], T=1, K=100, r=0.06, div=0.03)
    v1 = FD1.Crank(opt='Put')
    k = k + 1
    if k >= 20:
        break
df1.at["CNFD_put", "Number of Steps"] = k
df1.at["CNFD_put", "Price"] = v1
df1.at["CNFD_put", "N"] = Nl[k-1]
df1.at["CNFD_put", "dt"] = FD1.dt
df1.at["CNFD_put", "dx"] = FD1.dx
df1.at["CNFD_put", "diff"] = abs(v1 - BS_put)

print(df1)

## Comment:
# 1. Again, CNFD converges much faster than IFD and EFD
# 2. Note that, it takes longer for all methods to reach
#    the true convergence to BS price rather than the convergence
#    discussed above
# 3. Due to the dt, dx set up, IFD does not appear to be
#    faster than EFD. However, in other experiments (not shown
#    here), IFD can be slightly faster than EFD. Also, based
#    on the difference to BS price, IFD is slightly closer to 
#    BS under the same number of steps. This supports the fact
#    that IFD should converge faster than EFD. 
# 4. Another thing worth mentioning is the accuracy to BS price
#    when convergence is reached, CNFD appears to be performing 
#    the best amoung all.
# 5. The explanation of convergence speed above still holds.
#    (IFD and CNFD are not restricted to the convergence and 
#     stability condition, thus achieve convergence faster)


# In[15]:


# Problem 1 - (g)

dx = 0.005
dt = 0.001
r = 0.06
div = 0.03
sig = np.repeat(np.nan, 12)
pu = np.repeat(np.nan, 12)
pm = np.repeat(np.nan, 12)
pd = np.repeat(np.nan, 12)
for i in range(1, 13):
    sigma = 0.05*i
    nu = r - div - (sigma**2)/2
    sig[i-1] = sigma
    pu[i-1] = -0.5*dt*((sigma/dx)**2 + nu/dx)
    pd[i-1] = -0.5*dt*((sigma/dx)**2 - nu/dx)
    pm[i-1] = 1 + dt*((sigma/dx)**2) + r*dt

plt.plot(sig, pu, 'r--')
plt.plot(sig, pm, 'bs')
plt.plot(sig, pd, 'g^')
#plt.plot(sig, pd-pu)

plt.show()

# Comments:
# 1. pd and pu are negative and pm is larger than 1
# 2. We can no longer interpret pu, pm, pd as probabilities
# 3. pd and pu have small differences that's invisible on graph


# In[17]:


# Problem 1 - (i)
# Calculate Greeks - .EFD European Call

FD1 = FiniteDiff(S0=100, sigma=0.25, dt=10**(-4), Nj=300, dx=0.0045, T=1, K=100, r=0.06, div=0.03)
v1 = FD1.Explicit(opt='Call')
N = FD1.Nj
Delta = (FD1.EFD_c[0, N+1] - FD1.EFD_c[0, N-1]) / (FD1.St[N+1] - FD1.St[N-1])
Gamma = ((FD1.EFD_c[0, N+1] - FD1.EFD_c[0, N])/(FD1.St[N+1] - FD1.St[N]) -          (FD1.EFD_c[0, N] - FD1.EFD_c[0, N-1])/(FD1.St[N] - FD1.St[N-1])) / (0.5*(FD1.St[N+1] - FD1.St[N-1]))
Theta = (FD1.EFD_c[1, N] - FD1.EFD_c[0, N]) / FD1.dt
FD_u = FiniteDiff(S0=100, sigma=0.25*(1+0.001), dt=10**(-4), Nj=250, dx=0.006, T=1, K=100, r=0.06, div=0.03)
v_u = FD_u.Explicit(opt='Call')
FD_d = FiniteDiff(S0=100, sigma=0.25*(1-0.001), dt=10**(-4), Nj=250, dx=0.006, T=1, K=100, r=0.06, div=0.03)
v_d = FD_d.Explicit(opt='Call')
Vega = (v_u - v_d) / (FD_u.sigma - FD_d.sigma)

print("Delta:", Delta)
print("Gamma:", Gamma)
print("Theta:", Theta)
print("Vega:", Vega)


# In[14]:


# Problem 2. - Carr-Madan Fast Fourier Transform


# BS lognormal characteristic function
def BS_phi(u, sigma, ttm, r, q, lnS):
    nu = r - q - 0.5*(sigma**2)
    charafun = np.exp(1j*u*(lnS + nu*ttm) - 0.5*(sigma*sigma*u*u*ttm))
    return charafun

def FFT(alpha, eta, N, T, r, sig, S0, K, div):
    lnS0 = np.log(S0)
    # Spacing unit for log-strike grid
    lambda1 = (2*np.pi) / (N*eta)
    # Log-strike range(-b, b)
    b = 0.5*lambda1*N
    disc = np.exp(-r*T)
    
    Vj = (np.linspace(1, N, N) - 1)*eta
    # Seires of log-strike prices range from -b to b with spacing of lambda
    Ku = -b + lambda1*(np.linspace(1, N, N) - 1)
    
    # Fourier transformation of modified call price - formula (6)
    psi = disc * BS_phi(u=Vj-(alpha+1)*1j, sigma=sig, ttm=T, r=r, q=div, lnS=lnS0)           / (alpha**2 + alpha - Vj**2 + 1j*(2*alpha+1)*Vj)
    # FFT for computing the sum - formula (16) & (24)
    x =  np.exp(1j * Vj * b) * psi * (eta/3) * (3 + np.power(-1, np.linspace(1, N, N)) - ((np.linspace(1, N, N)-1)==0))
    CallPrices = np.real(np.exp(-alpha*Ku)/np.pi*sft.fft(x))
    
    # Use interpolation to estimate price at log(K)
    index = int(np.floor((np.log(K) + b)/lambda1 + 1))
    # log(K) fall into the range(Ku[index], Ku[index-1])
    xp = [Ku[index], Ku[index-1]]
    yp = [CallPrices[index], CallPrices[index-1]]
    interp_f = interp1d(xp, yp)
    Call = float(interp_f(np.log(K)))
    print("Carr-Madan FFT European Call:", Call)
    return Call

# Used eta and N as suggested in the paper
FFT(alpha=5, eta=0.45, N=4096, T=1, r=0.06, sig=0.25, S0=100, K=100, div=0.03)
# Compared with BS formula price
BS = BlackScholes(vol=0.25, S0=100, T=1, K=100, r=0.06, q=0.03)
f = BS.optPrice(o='Call')
print("Black Scholes price:", f)


# Sources Used:
# 1. Matsuda, K., Introduction to Option Pricing with Fourier Transform: 
#    Option Pricing with Exponential Lévy Models, December 2004, pp.111,
#    Graduate School and University Center of the City University of New York,
#    "http://www.maxmatsuda.com/Papers/2004/Matsuda%20Intro%20FT%20Pricing.pdf".
# 2. Kienitz, J., Wetterau, D., Financial Modelling Theory, Implementation 
#    and Practice (with Matlab source), John Wiley Sons Ltd, 2012, pp.202-210

