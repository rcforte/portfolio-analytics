import datetime
import scipy.optimize as scopt
import pandas_datareader as data
import pandas as pd
import numpy as np

def get_historical_closes(ticker, start_date, end_date):
    f = True
    for t in ticker:
        pnl = data.DataReader(t, 'yahoo', start_date, end_date)
        pnl['minor'] = t
        tmp = pnl[['minor', 'Adj Close']].reset_index()
        tmp.rename(columns={'minor':'Ticker', 'Adj Close':'Close'}, inplace=True)
        if f:
            df = tmp
            f = False
        else:
            df = pd.concat([df,tmp])
    piv = df.pivot(index='Date', columns='Ticker')
    piv.columns = piv.columns.droplevel(0)
    return piv

def calc_daily_returns(closes):
    return np.log(closes/closes.shift(1))

def calc_annual_returns(daily_returns):
    return np.exp(daily_returns.groupby(lambda date: date.year).sum())-1

def calc_portfolio_var(returns, weights=None):
    n = returns.columns.size
    if weights is None:
        weights = np.ones(n)/n
    sigma = np.cov(returns.T, ddof=0)
    return (weights * sigma * weights.T).sum()

def sharpe_ratio(returns, weights=None, risk_free_rate=0.015):
    n = returns.columns.size
    if weights is None:
        weights = np.ones(n)/n
    var = calc_portfolio_var(returns, weights)
    means = returns.mean()
    return (means.dot(weights) - risk_free_rate/np.sqrt(var))

def y_f(x):
    return 2+x**2

if __name__ == '__main__':
    #scopt.fmin(y_f,1000)
    closes = get_historical_closes(['MSFT','AAPL','KO'], '2010-01-01', '2014-12-31')
    print(closes)

    day_rets = calc_daily_returns(closes)
    print(day_rets)

    ann_rets = calc_annual_returns(day_rets)
    print(ann_rets)

    ptf_var = calc_portfolio_var(ann_rets)
    print(ptf_var)

    sr = sharpe_ratio(ann_rets)
    print(sr)

    x = data.DataReader('MSFT', 'yahoo', '2010-01-01', '2014-12-31')
    print(x)


