#!/usr/bin/env python
import numpy as np;

## Vasicek Model Tree Construction

# Short Rate Tree - branch reducing method
def exp_rate(r_base, k, theta, sigma, dt):
    r = r_base + k*(theta - r_base)*dt
    return r

def step_rate(r_base, k, theta, sigma, dt, direction):
    if direction.lower() == 'up' or direction.upper() == 'U':
        r = r_base + k*(theta - r_base)*dt + sigma*np.sqrt(dt)
    if direction.lower() == 'down' or direction.upper() == 'D':
        r = r_base + k*(theta - r_base)*dt - sigma*np.sqrt(dt)
    return r

def p_up(sd, r_ex, r1):
    p1 = 1/((sd**2 - (r_ex-r1)**2)/((r_ex-r1)**2) + 2)
    return p1
    
def p_down(sd, r_ex, r1):
    p2 = 1 - 1/((sd**2 - (r_ex-r1)**2)/((r_ex-r1)**2) + 2)
    return p2

def VasicekTree(r0, k, theta, sigma, dt, step):
    rates = np.zeros((step+1, step+1))
    exR = np.repeat(np.nan, step+1)
    rates[0, 0] = r0
    rates[1, 0] = r0 + k*(theta - r0)*dt + sigma*np.sqrt(dt)
    rates[1, 1] = r0 + k*(theta - r0)*dt - sigma*np.sqrt(dt)
    sd = sigma*np.sqrt(dt)
    exR[0] = r0
    for x in range(1, step+1):
        exR[x] = exp_rate(exR[x-1], k, theta, sigma, dt)
        if x%2 == 0:
            rates[x, int(x/2)] = exR[x]
            if x!= step:
                rates[x+1, int(x/2)] = step_rate(rates[x, int(x/2)], k, theta, sigma, dt, "U")
                rates[x+1, int(x/2)+1] = step_rate(rates[x, int(x/2)], k, theta, sigma, dt, 'D')
    for i in range(2, step+1):
        for j in range(0, i+1):
            if rates[i, j] == 0:
                if j < i//2:
                    ex_r_step = exp_rate(rates[i-1, j], k, theta, sigma, dt)
                    p = p_up(sd, ex_r_step, rates[i, j+1])
                    rates[i, j] = (ex_r_step - rates[i, j+1])/p + rates[i, j+1]
                if j > i//2:
                    ex_r_step = exp_rate(rates[i-1, j-1], k, theta, sigma, dt)
                    p = p_down(sd, ex_r_step, rates[i, j-1])
                    rates[i, j] = (ex_r_step - rates[i, j-1])/p + rates[i, j-1]
    rates = np.around(rates, decimals=6)
    return rates

rt = VasicekTree(r0=5.121/100, k=0.025, theta=15.339/100, sigma=126/(10**4), dt=1/12, step=30)
#print(np.transpose(rt))
print("Rate outputs in step #30:", rt[30:], '\n')

# Monte Carlo Simulation Method - model formula
def MonteCarloRate(steps, r0, k, theta, sigma, dt, rep):
    rates_MC = np.repeat(np.nan, steps+1)
    rates_MC[0] = r0
    r_MC = 0
    for j in range(0, rep):
        for i in range(1, steps+1):
            rates_MC[i] = rates_MC[i-1] + k*(theta - rates_MC[i-1])*dt - sigma*np.sqrt(dt)*np.random.normal(0, 1)
        r_MC = r_MC + rates_MC[steps]
    return r_MC/rep

rMC = MonteCarloRate(steps=30, r0=5.121/100, k=0.025, theta=15.339/100, sigma=126/(10**4), dt=1/12, rep=100)
print("Monte Carlo output:", rMC)

