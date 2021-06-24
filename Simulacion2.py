#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 15:41:10 2021

@author: usuario
"""
import numpy as np
import pandas as pd
import pandas_datareader.data as wb

tickers=['FB','^GSPC']

data=pd.DataFrame()
#print(data)
for t in tickers:
     data[t]= wb.DataReader(t,'yahoo','2016-1-1')['Adj Close']
log_returns =np.log(1+data.pct_change())
#print(log_returns)
cov= log_returns.cov()*252
#print(cov)
cov_market =cov.iloc[0,1]
print(cov_market)
market_var=log_returns['^GSPC'].var()*252

stock_beta=cov_market/market_var

rf=0.0
riskpremium=(log_returns['^GSPC'].mean()*252) -rf   

stock_capm_return=rf+stock_beta*riskpremium
sharpe=(stock_capm_return-rf)/(log_returns['FB'].std()*252**0.5)
     
print('La Beta de'+str(tickers)+'es de:' +str(round(stock_beta,3)))     
print('El retorno CAPM de ' + str(tickers)+' es de' +str(round(stock_capm_return*100,3)
)+'%')
print('El ratio Sharpe de'+ str(tickers)+' es de'+str(round(sharpe,3)))