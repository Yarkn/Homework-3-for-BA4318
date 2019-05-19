from math import sqrt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.api import SimpleExpSmoothing, Holt #,ExponentialSmoothing,  
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
name = 'Close'
alpha = 0.2
slope=0.1

dfx = pd.read_csv("Data.txt", sep='\t')
seriesname = 'Close' 
may_series_name = 'Close'
may_six_series = dfx.iloc[336:368,5:6]
trendtype = 'additive'

df = dfx[dfx['Volume'] != 0]

def decomp(frame,name,f,mod='Additive'):
    series = frame[name]
    array = np.asarray(series, dtype=float)
    result = sm.tsa.seasonal_decompose(array,freq=f,model=mod,two_sided=False)
    result.plot()
    plt.show() 
    return result

#I will test the stationarity in this function
def stationarity_test(timeseries):

    rollingmean = pd.Series(timeseries).rolling(window=60).mean()
    rollingstd = pd.Series(timeseries).rolling(window=60).std()

    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rollingmean, color='red', label='Rolling Mean')
    std = plt.plot(rollingstd, color='black', label = 'Rolling Std')

    plt.title('Rolling Mean & Standard Deviation')
    plt.savefig('part2-c.png')
    plt.show(block=False)
    

    print("The Results for Dickey-Fuller Test:")
    array = np.asarray(timeseries, dtype='float')
    np.nan_to_num(array,copy=False)
    dftest = adfuller(array, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
   
#I will test the stationarity for sixth of may   
def stationarity_test_may_six(timeseries):

    rollingmean = pd.Series(timeseries).rolling(window=60).mean()
    rollingstd = pd.Series(timeseries).rolling(window=60).std()

    orig = plt.plot(timeseries, color='gray',label='Original')
    mean = plt.plot(rollingmean, color='brown', label='Rolling Mean')
    std = plt.plot(rollingstd, color='purple', label = 'Rolling Std')

    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)


    print("The Results for Dickey-Fuller Test:")
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
    estimate = fit.predict(365)[0]
    print(estimate)
    return estimate

predict = holt_predict(df,name)

#This is the function for predicting 15.58
def holt_predict2(df, seriesname):
    numbers = np.asarray(df[seriesname])
    model = ExponentialSmoothing(numbers, trend=trendtype)
    
    fit = model.fit(optimized=True, remove_bias=True)
    estimate2 = fit.predict(366)[0]
    print(estimate2)
    return estimate2

predict2 = holt_predict2(df,name)

# This function is for calculating the RMSE Values for 15.57 and 15.58
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

rmse_val = rmse(predict,6.0185)
print(rmse_val)

rmse_val2 = rmse(predict2,6.01933)
print(rmse_val2)


series = df[seriesname]
stationarity_test(series)

may_six = may_six_series[may_series_name]
stationarity_test_may_six(may_six)
