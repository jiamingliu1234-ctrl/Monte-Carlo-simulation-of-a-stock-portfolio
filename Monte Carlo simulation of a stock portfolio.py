import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from pandas_datareader import data as pdr

#import data
def get_data(stocks, start, ends):
    import yfinance as yf
    stockData = yf.download(stocks, start=start, end=ends, auto_adjust=True)
    stockData = stockData['Close']
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix

stocklist = ['NVDA','PLTR','TSLA','AAPL']
stocks = stocklist
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=300)

meanReturns, covMatrix = get_data(stocks, startDate, endDate)

weights = np.random.random(len(meanReturns))
weights /= np.sum(weights)

# Monte Carlo Method
# number of stimulation
mc_sims = 100
T = 100 #time frame in days

meanM = np.full(shape=(T, len(weights)), fill_value=meanReturns)
meanM = meanM.T

profolios_sims = np.full(shape=(T, mc_sims), fill_value=0.0)

initialPorfolios = 10000
for m in range(mc_sims):
# MC Loops
    Z = np.random.normal(size=(T, len(weights)))
    L = np.linalg.cholesky(covMatrix)
    dailyreturns = meanM + np.inner(L,Z)
    profolios_sims[:,m] = np.cumprod(np.inner(weights, dailyreturns.T)+1)*initialPorfolios

# Graph sims
# plt.plot(profolios_sims)
# plt.ylabel('Portfolio value ($)')
# plt.xlabel('Date')
# plt.title(' MC simulation of a stock portfolio')
# plt.show()

#mean return
paths = profolios_sims              # (T, mc_sims)
V0 = float(initialPorfolios)
terminal = paths[-1, :]             # (mc_sims,)
exp_terminal = terminal.mean()
print("Expected return:", exp_terminal)

#medium return
median_terminal = np.median(terminal)
print("Median return:", median_terminal)

# prob that VT < V0
loss_prob = np.mean(terminal < V0)
print("Loss probability:", loss_prob)