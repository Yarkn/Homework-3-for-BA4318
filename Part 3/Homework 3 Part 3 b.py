from math import sqrt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.api import SimpleExpSmoothing, Holt #,ExponentialSmoothing,  
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt 
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error

dfx = pd.read_csv("Data.txt", sep='\t')
dfa = dfx[dfx['Volume'] != 0]
seriesname = 'Close' 
series = dfa[seriesname]
name = 'Close'
alpha = 0.2
slope=0.1
trendtype = 'additive'
dfu = dfa["Close"]

dfz = dfu.values.tolist()
#df = pd.Series(series).rolling(window=60).apply(dfa)

liste = []
liste2 = []

total = 1830
#adding values to list2
def value_x(x):
    liste2.append(x)


#adding values to list
def add_value(x):
    
    liste.append(x)

#create a list for weights. Sum of 1+2+3...+60 = 1830. That is where 1830 coming from.
def weights():
    i=0
    while i <60:
        x = i/1830
        add_value(x)
        
        
        i+=1
weights()

#Makes calculations
def calculate():
    i=60
    while i< 8127:
        
        dfo = dfu.iloc[i-60:i]
        value = dfo.values.tolist()
        t = 0
        while t<60:
            x = value[t] * liste[t]
            x += x
            t += 1
        value_x(x)
        i += 1    
calculate()  
    
df = pd.DataFrame(liste2)

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
    plt.savefig('part3all.png')
    plt.show(block=False)
    plt.savefig('part3figure1.png')

    print("The results for Dickey-Fuller Test:")
    array = np.asarray(timeseries, dtype='float')
    np.nan_to_num(array,copy=False)
    dftest = adfuller(array, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)


#These are the functions for estimations

def prediction(df, seriesname):
    numbers = np.asarray(dfa[seriesname])
    model = ExponentialSmoothing(numbers, trend=trendtype)
    
    fit = model.fit(optimized=True, remove_bias=True)
    estimate = fit.predict(11037)[0]
    print(' The estimation for 15:57 is: ')
    print(estimate)
    return estimate

predict = prediction(df,name)

def prediction2(df, seriesname):
    numbers = np.asarray(dfa[seriesname])
    model = ExponentialSmoothing(numbers, trend=trendtype)
    
    fit = model.fit(optimized=True, remove_bias=True)
    estimate2 = fit.predict(11038)[0]
    print('The stimation for 15:58 is: ')
    print(estimate2)
    return estimate2

predict2 = prediction2(df,name)

def prediction3(df, seriesname):
    numbers = np.asarray(dfa[seriesname])
    model = ExponentialSmoothing(numbers, trend=trendtype)
    
    fit = model.fit(optimized=True, remove_bias=True)
    estimate3 = fit.predict(12480)[0]
    print('The estimation for 16:00 for 7th of May is: ')
    print(estimate3)
    return estimate3

predict3 = prediction3(df,name)

def prediction4(df, seriesname):
    numbers = np.asarray(dfa[seriesname])
    model = ExponentialSmoothing(numbers, trend=trendtype)
    
    fit = model.fit(optimized=True, remove_bias=True)
    estimate4 = fit.predict(11039)[0]
    print('The estimation for 15:59 is: ')
    print(estimate4)
    return estimate4

predict4 = prediction4(df,name)

def prediction5(df, seriesname):
    numbers = np.asarray(dfa[seriesname])
    model = ExponentialSmoothing(numbers, trend=trendtype)
    
    fit = model.fit(optimized=True, remove_bias=True)
    estimate5 = fit.predict(11040)[0]
    print('The estimation for 16:00 is: ')
    print(estimate5)
    return estimate5

predict5 = prediction5(df,name)

#The function for RMSE values of 15.57 and 15.58
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

rmse_val = rmse(predict,6.0185)
print('The RMSE Value for 15.57 is: ')
print(rmse_val)

rmse_val2 = rmse(predict2,6.01933)
print('The RMSE Value for 15.58 is: ')
print(rmse_val2)



stationarity_test(series)

