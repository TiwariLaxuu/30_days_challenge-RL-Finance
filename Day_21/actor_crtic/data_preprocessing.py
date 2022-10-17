import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import quandl
from scipy import signal



def data_preprocess():
    apl_stock=quandl.get('WIKI/AAPL', start_date="2014-01-01", end_date="2018-08-20" )
    msf_stock=quandl.get('WIKI/MSFT', start_date="2014-01-01", end_date="2018-08-20")

    #data_preprocessing 
    apl_stock['Close'].loc[:'2014-06-09'].iloc[:-1] /= 7 
    apl_stock['Open'].loc[:'2014-06-09'].iloc[:-1] /= 7 
    msf_stock['Close'].loc[:'2014-06-09'].iloc[:-1] /= 7 
    msf_stock['Open'].loc[:'2014-06-09'].iloc[:-1] /= 7 

    apl_open = apl_stock["Open"].values
    apl_close = apl_stock["Close"].values
    msf_open = msf_stock["Open"].values
    msf_close = msf_stock["Close"].values 

    apl_stock.to_csv('aapl.csv')
    apl_stock.to_csv('msf.csv')
    
    #apl_open_close fig 
    apl_open = signal.detrend(apl_open)
    apl_close = signal.detrend(apl_close)
    plt.plot(range(0,len(apl_open)), apl_open)
    plt.savefig('apl_open.png')
    plt.plot(range(0,len(apl_close)), apl_close)
    plt.savefig('apl_close.png')
    #msf_open_close fig 
    msf_open = signal.detrend(msf_open)
    msf_close = signal.detrend(msf_close)
    plt.plot(range(0,len(msf_open)), msf_open)
    plt.savefig('msf_open.png')
    plt.plot(range(0,len(msf_close)), msf_close)
    plt.savefig('msf_close.png')

    ''' 
    It doesn't make sense to buy a share at a negative value. 
    But since the model will learn to maximize reward, we can just shift it up by a 
    constant number so it's always positive. 
    '''

    print(apl_open.min())
    print(apl_close.min())
    print(msf_open .min())
    print(msf_close.min()) 

    apl_open += 35.
    apl_close += 35.
    msf_open  += 35.
    msf_close += 35. 

    return apl_stock, apl_open, apl_close, msf_stock, msf_open, msf_close

