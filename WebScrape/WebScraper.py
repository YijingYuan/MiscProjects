#!/usr/bin/env python

import requests
from bs4 import BeautifulSoup  
import numpy as np
import pandas as pd


def getWeatherURL(latitude,longitude):
    URL = "https://forecast.weather.gov/MapClick.php?lat={0:0.4f}&lon={1:0.4f}"
    return URL.format(latitude, longitude)

def fetchPage(URL):
    page = requests.get(URL)
    if (page.status_code > 200):
        print("Error retrieving page, code %s\n" % (page.status_code))
        exit()
    return page

# Print an error and return false if query fails
def selectOrFail(tree, pattern):
    result = tree.select(pattern)
    if (result == []):
        print("Error: no results returned for pattern '{}'".format(pattern))
        return False
    return result[0]

# Parses the current conditions detail table
# Make a dictionary using headings as keys and text as values
def getTableDict(table):
    result = {}
    for row in table.select('tr'):
        data = row.select('td')
        if (len(data) < 2): # error in data
            return result
        result[data[0].select('b')[0].get_text()] = data[1].get_text()
    return result

def parseData(pageContent):
    result = {
        'location':     np.nan,
        'forecast':     np.nan,
        'temperature':  np.nan,
        'humidity':     np.nan,
        'windspeed':    np.nan,
        'lastupdate':   np.nan
    }
    soup = BeautifulSoup(pageContent, 'html.parser')

    conditions = selectOrFail(soup, 'div#current-conditions')
    if (not conditions):
        return result

    location = selectOrFail(conditions, 'div.panel-heading div h2')
    if (location):
        result['location'] = location.get_text()

    body = selectOrFail(conditions, 'div#current-conditions-body')
    if (not body):
        return result

    forecast = selectOrFail(body, 'div#current_conditions-summary p.myforecast-current')
    if (forecast and forecast.get_text() != 'NA'):
        result['forecast'] = forecast.get_text()

    temperature = selectOrFail(body, 'div#current_conditions-summary p.myforecast-current-lrg')
    if (temperature):
        result['temperature'] = temperature.get_text().replace('°F', '')

    table = selectOrFail(body, 'div#current_conditions_detail table') # table tbody')

    if (not table):
        return result

    data = getTableDict(table)
    if (data['Humidity'] != 'NA'):
        result['humidity']  = data['Humidity']
    if (data['Wind Speed'] != 'NA'):
        result['windspeed'] = data['Wind Speed']
    result['lastupdate'] = data['Last update'].strip() # remove extra white space

    return result


def queryWeather(latitude, longitude):
    page = fetchPage(getWeatherURL(latitude, longitude))
    return parseData(page.content)
    
#print(queryWeather(40.746, -74.0321))

def appendWeather(coordinates):
    result = queryWeather(coordinates[0], coordinates[1])
    return coordinates.append(pd.Series(
        [result['location'], result['forecast'], result['temperature'],
         result['humidity'], result['windspeed'], result['lastupdate'] ],
        index=['location', 'forecast', 'temperature',
               'humidity', 'windspeed', 'lastupdate']))


# Import randomly selected 1000 rows from data
skip = np.random.randint(2,33145,33144-1000)
skip = np.sort(skip)
Locations = pd.read_csv('locations.csv', skiprows = skip).iloc[:, 1:]
# Only takes the top 100 rows to print output due to lengthy running time
Locations = Locations.head(100)

weatherData = Locations.apply(appendWeather, axis=1)
weatherData.dropna(inplace = True)
weatherData.drop_duplicates(subset='location', keep='first', inplace= True)
weatherData = weatherData.sort_values('temperature')

print(weatherData.head(5), '\n')
print(weatherData.tail(5))


