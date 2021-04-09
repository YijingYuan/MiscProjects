#!/usr/bin/python3
import json
import numpy as np
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
from sklearn.linear_model import LinearRegression
import sys
import os
import requests

API_KEY = 'PASTE_GLASSNODE_API_KEY_HERE';
paths   = {
    'active': 'addresses/active_count',
    'price':  'market/price_usd',
}
baseURL = 'https://api.glassnode.com/v1/metrics/'
zoomDay = 730; # half of the zoom window in days

def fetch(metric, coin):
    result = {}
    response = requests.get('{}{}'.format(baseURL, paths[metric]),
                params= {'a': coin, 'api_key': API_KEY})
    data = json.loads(response.text)
    mint = data[0]['t']
    maxt = data[-1]['t']
    for point in data:
        result[point['t']] = point['v']
    return [result, mint, maxt]

def ingest(filename):
    result = {}
    with open(filename) as jsonfile:
        data = json.load(jsonfile)
        mint = data[0]['t']
        maxt = data[-1]['t']
        for point in data:
            result[point['t']] = point['v']
    return [result, mint, maxt]

def getThresholdEpoch(active, epoch, end, threshold=10000):
    while epoch <= end:
        epoch += 86400
        if not(epoch in active) or active[epoch] is None:
            continue
        if active[epoch] >= threshold:
            return epoch
    return end

def getDate(epoch):
    return os.popen("date --date='@{}' +%D".format(epoch)).read().rstrip()

def getEpoch(date):
    return int(os.popen("date --date='{}' +%s".format(date)).read().rstrip())



#[activeE, minE, maxE] = fetch('active', 'ETH');
#[activeB, minB, maxB] = fetch('active', 'BTC');
[activeE, minE, maxE] = ingest('eth/active.json')
[activeB, minB, maxB] = ingest('btc/active.json')
# calculate diff
threshEpochB = getThresholdEpoch(activeB, minB, maxB)
threshEpochE = getThresholdEpoch(activeE, minE, maxE)
#diff  = 152928000
diff = threshEpochE - threshEpochB

print("Bitcoin  range: {}-{} [{}-{}]".format(getDate(minB),getDate(maxB), minB, maxB))
print("Ethereum range: {}-{} [{}-{}]".format(getDate(minE),getDate(maxE), minE, maxE))
print("10k active addresses Bitcoin:  {} [{}] -24hr[{}]".format(threshEpochB,
        activeB[threshEpochB], activeB[threshEpochB-86400]))
print("10k active addresses Ethereum: {} [{}] -24hr[{}]".format(threshEpochE,
        activeE[threshEpochE], activeE[threshEpochE-86400]))
print("difference: {}".format(diff))


XE = np.array([])
YE = np.array([])
XB = np.array([])
YB = np.array([])

# btc[t-diff] = eth[t]
epoch = max(minE, minB)
end   = min(maxE, maxB)

#epoch = 1458432000
# line up the data
idx = 0
zLeftBound = 0
while epoch <= end:
    # only count valid data that we have for both coins
    if not(epoch in activeE) or not(epoch in activeB):
        epoch += 86400
        continue
    if (activeE[epoch] is None) or (activeB[epoch] is None):
        epoch += 86400
        continue
    avgE = 0.0
    avgB = 0.0
    days = 0
    # calculate 7 day rolling average of active addresses
    for i in range(-3,4,1):
        edx = epoch + 86400 * i
        if edx in activeE and not(activeE[edx] is None):
            if edx in activeB and not(activeB[edx] is None):
                days += 1
                avgE += activeE[edx]
                avgB += activeB[edx-diff]
    avgE /= days
    avgB /= days
    # X = time
    XE = np.append(XE, epoch)
    XB = np.append(XB, epoch)
    # Y = active
    YE = np.append(YE, avgE)
    YB = np.append(YB, avgB)
    epoch += 86400
    idx   += 1
    if zLeftBound == 0 and epoch >= (end - zoomDay * 86400):
        zLeftBound = idx

# include the remaining "future" bitcoin data
epoch -= diff
while epoch < end:
    if not(epoch in activeB):
        epoch += 86400
        continue
    if activeB[epoch] is None:
        epoch += 86400
        continue
    avgB = 0.0
    days = 0
    for i in range(-3,4,1):
        idx = epoch + 86400 * i
        if idx in activeB and not(activeB[idx] is None):
            days += 1
            avgB += activeB[idx]
    avgB /= days
    XB = np.append(XB, epoch+diff)
    YB = np.append(YB, avgB)
    epoch += 86400

# y = e^b * x^m

# create zoom plot arrays
#ZXB = XB[zLeftBound:min(zLeftBound + 2*zoomDay, end)]
#ZXE = XE[zLeftBound:]
ZXB = np.arange(-zoomDay, zoomDay+1, step=1)
ZXE = np.arange(-zoomDay, 1, step=1)
ZYB = YB[zLeftBound:min(zLeftBound + 2*zoomDay+1, end)]
ZYE = YE[zLeftBound:]

#fig, (ax1, ax2) = plt.subplots(1,2)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig.suptitle('BTC & ETH Active Addresses with BTC curve shifted to sync with ETH @10k')

ax1.scatter(XB.reshape(-1,1), YB.reshape(-1,1), color='red', label='BTC')
ax1.scatter(XE.reshape(-1,1), YE.reshape(-1,1), label='ETH')
ax2.scatter(XB.reshape(-1,1), YB.reshape(-1,1), color='red', label='BTC')
ax2.scatter(XE.reshape(-1,1), YE.reshape(-1,1), label='ETH')
ax2.set_yscale('log')
ax1.legend()
ax2.legend()
ax2.grid(True, which='both', axis='y', ls='-')
ax1.set_xlabel('Date')
ax2.set_xlabel('Date')
ax1.set_ylabel('Active Addresses')
ax2.set_ylabel('Active Addresses')

dates = [
    '01/01/2016',
    '01/01/2017',
    '01/01/2018',
    '01/01/2019',
    '01/01/2020',
    '01/01/2021',
    '01/01/2022',
    '01/01/2023',
    '01/01/2024',
    '01/01/2025',
    '01/01/2026'
];
xticks = np.array(list(map(getEpoch, dates)))

ax1.set_xticks(xticks)
ax1.set_xticklabels(dates, rotation=20, ha='right')
ax2.set_xticks(xticks)
ax2.set_xticklabels(dates, rotation=20, ha='right')


ax3.scatter(ZXB.reshape(-1,1), ZYB.reshape(-1,1), color='red', label='BTC')
ax3.scatter(ZXE.reshape(-1,1), ZYE.reshape(-1,1), label='ETH')
ax4.scatter(ZXB.reshape(-1,1), ZYB.reshape(-1,1), color='red', label='BTC')
ax4.scatter(ZXE.reshape(-1,1), ZYE.reshape(-1,1), label='ETH')
ax4.set_yscale('log')
ax3.legend()
ax4.legend()
ax4.grid(True, which='both', axis='y', ls='-')
ax3.set_xlabel('Time[days]')
ax4.set_xlabel('Time[days]')
ax3.set_ylabel('Active Addresses')
ax4.set_ylabel('Active Addresses')

ax3.set_xticks(np.arange(-zoomDay, zoomDay, step=zoomDay/4))
ax4.set_xticks(np.arange(-zoomDay, zoomDay, step=zoomDay/4))

plt.show()

#ax1.scatter(X.reshape(-1,1), Y.reshape(-1,1))
#ax2.scatter(X.reshape(-1,1), Y.reshape(-1,1))
#ax2.set_xscale('log')
#ax2.set_yscale('log')
#ax2.grid(True, which='both', ls='-')
#fig.suptitle('Active Addresses vs. Price [{}]'.format(coin.upper()))
#ax1.set_xlabel('Active Addresses')
#ax1.set_ylabel('Price')
#ax2.set_xlabel('Active Addresses')
#ax2.set_ylabel('Price')
#ax2.set_ylim(0.1,)
#ax1.text(1000,maxprice*.95,'y = e^{:.6} * x^{:.6}'.format(lr.intercept_[0], lr.coef_[0][0]), color='black', bbox=dict(facecolor='white', edgecolor='black', pad=10.0))
#ax2.text(1000,maxprice*.95,'y = {:.6}*x + {:.6}'.format(lr.coef_[0][0], lr.intercept_[0]), color='black', bbox=dict(facecolor='white', edgecolor='black', pad=10.0) )
#ax1.plot(X.reshape(-1,1), fit, color='red')
#ax2.plot(X.reshape(-1,1), fit, color='red')
#
#b = lr.intercept_[0]
#m = lr.coef_[0][0]
#
#ePred = np.array([np.exp(b)*np.float_power(active[ei],m) for ei in E]).reshape(-1,1)
#
#
#ax3.scatter(E.reshape(-1,1), Y.reshape(-1,1))
#ax3.set_yscale('log')
#ax3.set_ylim(0.1,)
##ax3.set_xbound(150000000,)
#ax3.set_ylabel('Price')
#ax3.set_xlabel('Time[Unix Epoch]')
#ax3.grid(True, which='both', axis='y', ls='-')
#
##ax3.vlines(1438992000, 0.1, 100000)
#
#ax3.plot(E.reshape(-1,1), ePred.reshape(-1,1), color='red')
#
#
#
#
#plt.show()



