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
may_six_series = dfx.iloc[10080:11038,5:6]
trendtype = 'additive'

df = dfx[dfx['Volume'] != 0]

def decomp(frame,name,f,mod='Additive'):
    series = frame[name]
    array = np.asarray(series, dtype=float)
    result = sm.tsa.seasonal_decompose(array,freq=f,model=mod,two_sided=False)
    result.plot()
    plt.show() 
    return result


#The function which is testing for stationarity
def test_stationarity(timeseries):

    rollingmean = pd.Series(timeseries).rolling(window=60).mean()
    rollingstd = pd.Series(timeseries).rolling(window=60).std()

    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rollingmean, color='red', label='Rolling Mean')
    std = plt.plot(rollingstd, color='black', label = 'Rolling Std')

    plt.title('Rolling Mean & Standard Deviation')
    plt.savefig('GraphC.png')
    plt.show(block=False)
   

    print("The Results for Dickey-Fuller Test:")
    array = np.asarray(timeseries, dtype='float')
    np.nan_to_num(array,copy=False)
    dftest = adfuller(array, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    
#The function which is testing for the stationarity of sixth of May and then printing the results of dickey fuller test
def test_stationarity_may_six(timeseries):

    rollingmean = pd.Series(timeseries).rolling(window=60).mean()
    rollingstd = pd.Series(timeseries).rolling(window=60).std()

    orig = plt.plot(timeseries, color='gray',label='Original')
    mean = plt.plot(rollingmean, color='brown', label='Rolling Mean')
    std = plt.plot(rollingstd, color='purple', label = 'Rolling Std')

    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)


    print("The results for Dickey-Fuller Test is: ")
    array = np.asarray(timeseries, dtype='float')
    np.nan_to_num(array,copy=False)
    dftest = adfuller(array, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

#Function predicting for 15:57
def prediction(df, seriesname):
    numbers = np.asarray(df[seriesname])
    model = ExponentialSmoothing(numbers, trend=trendtype)
    
    fit = model.fit(optimized=True, remove_bias=True)
    estimate = fit.predict(11037)[0]
    print('Estimation of 15:57 is: ')
    print(estimate)
    return estimate

predict = prediction(df,name)

#Function predicting for 15.58
def prediction2(df, seriesname):
    numbers = np.asarray(df[seriesname])
    model = ExponentialSmoothing(numbers, trend=trendtype)
    
    fit = model.fit(optimized=True, remove_bias=True)
    estimate2 = fit.predict(11038)[0]
    print('Estimation of 15:58 is: ')
    print(estimate2)
    return estimate2

predict2 = prediction2(df,name)

# Function which calculates RMSE values for 15.57 and 15.58
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

rmse_val = rmse(predict,6.0185)
print('RMSE Value of 15.57 is: ')
print(rmse_val)

rmse_val2 = rmse(predict2,6.01933)
print('RMSE Value of 15.58 is: ')
print(rmse_val2)

def prediction3(df, seriesname):
    numbers = np.asarray(df[seriesname])
    model = ExponentialSmoothing(numbers, trend=trendtype)
    
    fit = model.fit(optimized=True, remove_bias=True)
    estimate3 = fit.predict(11039)[0]
    print('Estimation of 15:59 is: ')
    print(estimate3)
    return estimate3

predict3 = prediction3(df,name)

def prediction4(df, seriesname):
    numbers = np.asarray(df[seriesname])
    model = ExponentialSmoothing(numbers, trend=trendtype)
    
    fit = model.fit(optimized=True, remove_bias=True)
    estimate4 = fit.predict(11040)[0]
    print('Estimation of 16:00 is: ')
    print(estimate4)
    return estimate4


predict4 = prediction4(df,name)

def prediction5(df, seriesname):
    numbers = np.asarray(df[seriesname])
    model = ExponentialSmoothing(numbers, trend=trendtype)
    
    fit = model.fit(optimized=True, remove_bias=True)
    estimation5 = fit.predict(12480)[0]
    print('Estimation of 16:00 for 7th of May is: ')
    print(estimation5)
    return estimation5


series = df[seriesname]
test_stationarity(series)

may_six = may_six_series[may_series_name]
test_stationarity_may_six(may_six)

