#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 13:44:40 2020

@author: nrajendrapandian
"""

#conda install -c conda-forge fbprophet

import pandas as pd
import numpy as np

import requests
import json
from datetime import datetime, timedelta

#import pickle
#import math
import scipy.optimize as op
import matplotlib.pyplot as plt

from fbprophet import Prophet

# Define logistic funcion with the coefficients to estimate
def logistic_func(t, a, b, c):
    return c / (1 + a * np.exp(-b*t))

#####
# Step 1: Read the covid-19 daily data from REST API. I've decided to analyze the below countries.
#####

countries = ['Brazil', 'Canada', 'Germany', 'India', 'Iran', 'Italy', 'Spain', 'Thailand', 'United_States_of_America']

# Collect and validate the incoming data for one country
country_x = {'country': 'China'} # or {'code': 'DE'}
covid_rest_api = 'https://api.statworx.com/covid'
response = requests.post(url = covid_rest_api, data = json.dumps(country_x))

# Store it in a dataframe
covid_data = pd.DataFrame.from_dict(json.loads(response.text))

'''
covid_data.dtypes
covid_data.index
covid_data.shape
covid_data.head()
'''

# Collect other countries
for x in countries:
    country_x = {'country': x}
    covid_rest_api = 'https://api.statworx.com/covid'
    response = requests.post(url = covid_rest_api, data = json.dumps(country_x))
    
    covid_tmp = pd.DataFrame.from_dict(json.loads(response.text))
    covid_data = covid_data.append(covid_tmp, ignore_index = True)

#####
# Exporting the data to a csv for future reference
#####

covid_data.to_csv('/Users/nrajendrapandian/Documents/Python/covid-19/covid19_json_data_'\
                  + datetime.today().strftime('%Y%m%d') + '.csv')

#covid_data[covid_data.country == 'India']

covid_data.pivot(index = 'date', columns = 'country', values = 'cases_cum').plot()

country_excl = ['China', 'Germany', 'Iran', 'Italy', 'Spain', 'United_States_of_America']

covid_data[~covid_data.country.isin(country_excl)]\
    .pivot(index = 'date', columns = 'country', values = 'cases_cum').plot()

#####
# Introduce the days sequence field to start from the first case reporting date.
# This way I'm setting all the countries in the same scale for comparison.
#####

covid_cases = covid_data[covid_data.cases_cum > 0].iloc[:, [0, 6, 8, 9, 10]]
covid_cases['days'] = covid_cases.groupby(['country']).cumcount() + 1
covid_cases.reset_index(drop = True, inplace = True)

covid_cases.pivot(index = 'days', columns = 'country', values = 'cases_cum').plot()

covid_cases[~covid_cases.country.isin(country_excl)]\
    .pivot(index = 'days', columns = 'country', values = 'cases_cum').plot()

#####
# Step 2: Estiamte a, b, and c using Scipy Curve Fit for Nonlinear Least Squares Estimation.
#####

# Randomly initialize the coefficients
p0 = np.random.exponential(size = 3)
p0

# Set min bound 0 on all coefficients, and set different max bounds for each coefficient
# Upper bound of c could be the country's population size itself.
bounds = (0, [100000., 2., 1300000000.])

# Convert pd.Series to np.Array and use Scipy's curve fit to find the best Nonlinear Least Squares coefficients

country_fc = 'India'

# For close to a month, India had only 3 max cases.
# I'm skipping those idle period.
covid_logistic = covid_cases[covid_cases.country == country_fc][['date', 'cases_cum']].iloc[31:, ]
covid_logistic.reset_index(drop = True, inplace = True)
covid_logistic['days'] = covid_logistic.index + 1

x = np.array(covid_logistic['days'])
y = np.array(covid_logistic['cases_cum'])

(a,b,c), cov = op.curve_fit(logistic_func, x, y, bounds = bounds, p0 = p0, maxfev = 1000000)
a, b, c

def my_logistic(t):
    return c / (1 + a * np.exp(-b*t))

plt.scatter(x, y)
plt.plot(x, my_logistic(x))
plt.title('Logisitc Model vs. Actuals of ' + country_fc + ' COVID-19')
plt.legend(['Logistic Model', 'Actual Data'])
plt.xlabel('Time')
plt.ylabel('Infections')

# The time at which the growth is peak
t_fastest = np.log(a) / b
i_fastest = logistic_func(t_fastest, a, b, c)

t_fastest, i_fastest

res_df = covid_logistic.copy()
res_df['fastest_grow_day'] = t_fastest
res_df['fastest_grow_value'] = i_fastest
res_df['growth_stabilized'] = t_fastest <= x[-1]
res_df['timestep'] = x
res_df['res_func_logistic'] = logistic_func(x, a, b, c)

if t_fastest <= x[-1]:
    print('\n\nGrowth stabilized: ', country_fc, '| Fastest grow day:', t_fastest, '| Infections:', i_fastest, '\n\n')
    res_df['cap'] = logistic_func(x[-1] + 10, a, b, c)
else:
    print('\n\nGrowth increasing: ', country_fc, '| Fastest grow day:', t_fastest, '| Infections:', i_fastest, '\n\n')
    res_df['cap'] = logistic_func(i_fastest + 10, a, b, c)

#####
# Step 3: Forecast the logistic growth using Prophet.
#####

covid_fc = covid_logistic[['date', 'cases_cum']].copy()
covid_fc['infection_cap'] = res_df['cap'].iloc[0]
covid_fc.rename(columns={"date": "ds", "cases_cum": "y", "infection_cap": "cap"}, inplace = True)
covid_fc.reset_index(drop = True, inplace = True)

m = Prophet(growth = 'logistic')
m.fit(covid_fc)

future = m.make_future_dataframe(periods = 30)
future['cap'] = res_df['cap'].iloc[0]
#future.head()

forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

# Fastest growth day data
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].iloc[t_fastest.astype(int), ]

fig1 = m.plot(forecast)
fig1.set_size_inches(9, 5)
datenow = forecast[['ds']].iloc[t_fastest.astype(int), ][0]
dateend = datenow + timedelta(days=50)
datestart = dateend - timedelta(days=121)
plt.xlim([datestart, dateend])
plt.title("COVID19 forecast: " + country_fc, fontsize=20)
plt.xlabel("Day", fontsize=20)
plt.ylabel("Infections", fontsize=20)
plt.axvline(datenow, color="k", linestyle=":")
plt.show()




