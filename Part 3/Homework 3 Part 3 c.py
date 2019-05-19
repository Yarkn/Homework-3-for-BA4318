from math import sqrt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.api import SimpleExpSmoothing, Holt #,ExponentialSmoothing,  
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt 
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
import random


name = 0
alpha = 0.2
slope=0.1


dfx = pd.read_csv("Data.txt", sep='\t')
seriesname = 0
may_series_name = 0

trendtype = 'additive'

dff = []
dfk =[]
#function for adding values to the list
def add_value(value):
    dff.append(value)
#Selects random values. 11020 is highest value in range can divided by 60. That is why limit of loop is 11020
def random_select():
    i=0
    while (i<11020):
        
        dfal = dfx.iloc[i:i+60]
        
        dfqw = dfal.sample(n=1, random_state=1)
        
        index = dfqw.index.values.astype(int)[0]
        
        value = dfqw.get_value(index,'Close').astype(float)
        add_value(value)
        
        i += 60
random_select()
# Writes 60 times same variable. 184 is range.
def create():
    a = 0
    while a <184:
        y = dff[a]
        a += 1
        i=0
        while i <60:
            dfk.append(y)
            i +=1
create()
dfz = dfk

df = pd.DataFrame(dfk)
series = df[seriesname]

def decomp(frame,name,f,mod='Additive'):
    series = frame[name]
    array = np.asarray(series, dtype=float)
    result = sm.tsa.seasonal_decompose(array,freq=f,model=mod,two_sided=False)
    result.plot()
    plt.show() 
    return result

#I will test the stationarity in this function
def stationarity_test(timeseries):

    rolmean = pd.Series(timeseries).rolling(window=60).mean()
    rolstd = pd.Series(timeseries).rolling(window=60).std()

    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')

    plt.title('Rolling Mean & Standard Deviation')
    plt.savefig('part3all.png')
    plt.show(block=False)
    plt.savefig('part3figure1.png')

    print("Results of Dickey-Fuller Test:")
    array = np.asarray(timeseries, dtype='float')
    np.nan_to_num(array,copy=False)
    dftest = adfuller(array, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)



def prediction(df, seriesname):
    numbers = np.asarray(df[seriesname])
    model = ExponentialSmoothing(numbers, trend=trendtype)
    
    fit = model.fit(optimized=True, remove_bias=True)
    estimate = fit.predict(11037)[0]
    print('The estimation for 15:57 is: ')
    print(estimate)
    return estimate

predict = prediction(df,name)

def prediction2(df, seriesname):
    numbers = np.asarray(df[seriesname])
    model = ExponentialSmoothing(numbers, trend=trendtype)
    
    fit = model.fit(optimized=True, remove_bias=True)
    estimate2 = fit.predict(11038)[0]
    print('The estimation for 15:58 is: ')
    print(estimate2)
    return estimate2

predict2 = prediction2(df,name)

def prediction3(df, seriesname):
    numbers = np.asarray(df[seriesname])
    model = ExponentialSmoothing(numbers, trend=trendtype)
    
    fit = model.fit(optimized=True, remove_bias=True)
    estimate3 = fit.predict(12480)[0]
    print('The estimation for 16:00 7th of May is: ')
    print(estimate3)
    return estimate3

predict3 = prediction3(df,name)

def prediction4(df, seriesname):
    numbers = np.asarray(df[seriesname])
    model = ExponentialSmoothing(numbers, trend=trendtype)
    
    fit = model.fit(optimized=True, remove_bias=True)
    estimate4 = fit.predict(11039)[0]
    print('The estimation for 15:59 is: ')
    print(estimate4)
    return estimate4

predict4 = prediction4(df,name)

def prediction5(df, seriesname):
    numbers = np.asarray(df[seriesname])
    model = ExponentialSmoothing(numbers, trend=trendtype)
    
    fit = model.fit(optimized=True, remove_bias=True)
    estimate5 = fit.predict(11040)[0]
    print('The estimation for 16:00 is: ')
    print(estimate5)
    return estimate5

predict5 = prediction5(df,name)

#Function for RMSE values of 15.57 and 15.58
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

rmse_val = rmse(predict,6.0185)
print('RMSE Value of 15.57 is: ')
print(rmse_val)

rmse_val2 = rmse(predict2,6.01933)
print('RMSE Value of 15.58 is: ')
print(rmse_val2)



stationarity_test(series)

