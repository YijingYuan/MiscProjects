#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt

callmean = pd.read_csv('CallProfitMean.csv')
callmax = pd.read_csv('CallProfitMax.csv')
putmean = pd.read_csv('PutProfitMean.csv')
putmax = pd.read_csv('PutProfitMax.csv')

# call option
plt.figure(figsize = (14, 7))
plt.plot(callmean['Date'], callmean['profit'], label='mean profit of call')
plt.plot(callmax['Date'], callmax['profit'], label='max profit of call')
plt.xlabel('Expiration Date')
plt.ylabel('profit')
plt.title('Net Profit of Call Option at Expiration Date')
plt.legend()
for x,y in zip(callmean['Date'],callmean['profit']):

    label = "{:.2f}".format(y)

    plt.annotate(label, # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
for a,b in zip(callmax['Date'],callmax['profit']):

    label = "{:.2f}".format(b)

    plt.annotate(label, # this is the text
                 (a,b), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center    

plt.show()

# put option
plt.figure(figsize = (14, 7))
plt.plot(putmean['Date'], putmean['profit'], label='mean profit of put')
plt.plot(putmax['Date'], putmax['profit'], label='max profit of put')
plt.xlabel('Expiration Date')
plt.ylabel('profit')
plt.title('Net Profit of Put Option at Expiration Date')
plt.legend()
for x,y in zip(putmean['Date'],putmean['profit']):

    label = "{:.2f}".format(y)

    plt.annotate(label, # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
for a,b in zip(putmax['Date'],putmax['profit']):

    label = "{:.2f}".format(b)

    plt.annotate(label, # this is the text
                 (a,b), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center    

plt.show()

