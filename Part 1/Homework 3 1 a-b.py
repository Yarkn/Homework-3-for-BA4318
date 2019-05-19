from math import sqrt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.api import SimpleExpSmoothing, Holt #,ExponentialSmoothing,  
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt 
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
name = 'Close'
alpha = 0.2
slope=0.1

df = pd.read_csv("Data.txt", sep='\t')
seriesname = 'Close' 
may_series_name = 'Close'
may_six_series = df.iloc[10080:11038,5:6]
trendtype = 'additive'

def decomp(frame,name,f,mod='Additive'):
    series = frame[name]
    array = np.asarray(series, dtype=float)
    result = sm.tsa.seasonal_decompose(array,freq=f,model=mod,two_sided=False)
    result.plot()
    plt.show() 
    return result

#I will test the stationarity in this function
def test_stationarity(timeseries):

    rollingmean = pd.Series(timeseries).rolling(window=60).mean()
    rollingstd = pd.Series(timeseries).rolling(window=60).std()

    orig = plt.plot(timeseries, color='red',label='Original')
    mean = plt.plot(rollingmean, color='black', label='Rolling Mean')
    std = plt.plot(rollingstd, color='blue', label = 'Rolling Std')

    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    plt.savefig('Graph1.png')

    print("Results of Dickey-Fuller Test:")
    array = np.asarray(timeseries, dtype='float')
    np.nan_to_num(array,copy=False)
    dftest = adfuller(array, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    
#I will test the stationarity for sixth of may
def test_stationarity_may_six(timeseries):

    rollingmean = pd.Series(timeseries).rolling(window=60).mean()
    rollingstd = pd.Series(timeseries).rolling(window=60).std()

    orig = plt.plot(timeseries, color='purple',label='Original')
    mean = plt.plot(rollingmean, color='gray', label='Rolling Mean')
    std = plt.plot(rollingstd, color='brown', label = 'Rolling Std')

    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)


    print("The result for Dickey-Fuller test is: ")
    array = np.asarray(timeseries, dtype='float')
    np.nan_to_num(array,copy=False)
    dftest = adfuller(array, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

#This is the function for predicting 15.57
def holt_predict(df, seriesname):
    numbers = np.asarray(df[seriesname])
    model = ExponentialSmoothing(numbers, trend=trendtype)
    
    fit = model.fit(optimized=True, remove_bias=True)
    estimate = fit.predict(11037)[0]
    print('Estimation for 15:57 is: ')
    print(estimate)
    return estimate


#This is the function for predicting 15.58
def holt_predict2(df, seriesname):
    numbers = np.asarray(df[seriesname])
    model = ExponentialSmoothing(numbers, trend=trendtype)
    
    fit = model.fit(optimized=True, remove_bias=True)
    estimate2 = fit.predict(11038)[0]
    print('Estimation for 15:58 is: ')
    print(estimate2)
    return estimate2


#This is the function for predicting 15.59
def holt_predict3(df, seriesname):
    numbers = np.asarray(df[seriesname])
    model = ExponentialSmoothing(numbers, trend=trendtype)
    
    fit = model.fit(optimized=True, remove_bias=True)
    estimate3 = fit.predict(11039)[0]
    print('Estimation of 15:59 is: ')
    print(estimate3)
    return estimate3



#This is the function for predicting 16.00
def holt_predict4(df, seriesname):
    numbers = np.asarray(df[seriesname])
    model = ExponentialSmoothing(numbers, trend=trendtype)
    
    fit = model.fit(optimized=True, remove_bias=True)
    estimate4 = fit.predict(11040)[0]
    print('Estimation of 16:00 is: ')
    print(estimate4)
    return estimate4



def holt_predict5(df, seriesname):
    numbers = np.asarray(df[seriesname])
    model = ExponentialSmoothing(numbers, trend=trendtype)
    
    fit = model.fit(optimized=True, remove_bias=True)
    estimate5 = fit.predict(12480)[0]
    print('Estimation of 16:00, 7th of May is: ')
    print(estimate5)
    return estimate5

#Now, I'll call all the prediction functions

predict = holt_predict(df,name)

predict2 = holt_predict2(df,name)

predict3 = holt_predict3(df,name)

predict4 = holt_predict4(df,name)

predict5 = holt_predict5(df,name)

# This function is for calculating the RMSE Values for 15.57 and 15.58
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

rmse_val = rmse(predict,6.0185)
print('The RMSE Value of 15.57 is: ')
print(rmse_val)

rmse_val2 = rmse(predict2,6.01933)
print('The RMSE Value of 15.58 is: ')
print(rmse_val2)


series = df[seriesname]
test_stationarity(series)

may_six = may_six_series[may_series_name]
test_stationarity_may_six(may_six)

