import pandas as pd
import numpy as np
import pandas_datareader as web
from datetime import datetime

import scipy as sp
import scipy.optimize as scopt
import scipy.stats as spstats

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

#%matplotlib inline

pd.set_option('display.notebook_repr_html', False)
pd.set_option('display.max_columns', 7)
pd.set_option('display.max_rows', 10)
pd.set_option('display.width', 82)
pd.set_option('precision', 3)

def get_historical_closes(ticker, start_date, end_date):
    first_time = True

    # For each ticker
    for t in ticker:
        # get the data from yahoo finance
        pnl = web.DataReader(t, 'yahoo', start_date, end_date)

        # add a column for the ticker and call it minor
        pnl['minor'] = t

        # extract a new dataframe only with
        # the minor and adj close columns
        tmp = pnl[['minor', 'Adj Close']].reset_index()

        # rename the columns to tiker and close
        tmp.rename(
            columns={'minor':'Ticker', 'Adj Close':'Close'},
            inplace=True
        )

        if first_time:
            # if first ticker, just assign df
            df = tmp
            first_time = False
        else:
            # if has tickers already, concat
            # to the dataframe
            df = pd.concat([df,tmp])

    # pivot the dataframe by ticker
    piv = df.pivot(index='Date', columns='Ticker')

    # pivoting creates a multi-index with
    # 2 levels since. We need to make sure
    # we remove the first level, which is
    # close price
    piv.columns = piv.columns.droplevel(0)
    return piv

def calc_daily_returns(closes):
    # calculates the returns, and converts it
    # to log space so that it is easier to
    # annualize
    return np.log(closes/closes.shift(1))

def calc_annual_returns(daily_returns):
    # takes all daily returns of a given year
    # sums them up and converts back to arithmetic
    # space. Notice we can add daily returns because
    # they are in log space
    return np.exp(daily_returns.groupby(lambda date: date.year).sum())-1

def calc_portfolio_var(returns, weights=None):
    n = returns.columns.size
    if weights is None:
        # equally distributes the weights
        weights = np.ones(n)/n

    # Takes the cov matrix of the returns
    # ddof returns a simple average
    sigma = np.cov(returns.T, ddof=0)

    # applies the portfolio weights to the
    # cov matrix so that it becomes the portfolio
    # cov matrix instead of stock returns cov matrix
    return (weights * sigma * weights.T).sum()

def sharpe_ratio(returns, weights=None, risk_free_rate=0.015):
    n = returns.columns.size
    if weights is None:
        # equally distributes the weights
        weights = np.ones(n)/n

    # calculate the cov matrix
    var = calc_portfolio_var(returns, weights)

    # calculate the returns mean
    means = returns.mean()

    # calculate the portfolio mean returns
    # subtracts from the risk free rate and
    # divides by the portfolio std
    return (means.dot(weights) - risk_free_rate)/np.sqrt(var)

def negative_sharpe_ratio_n_minus_1_stock(weights, returns, risk_free_rate):
    weights2 = sp.append(weights, 1-np.sum(weights))
    return -sharpe_ratio(returns, weights2, risk_free_rate)

def optimize_portfolio(returns, risk_free_rate):
    w0 = np.ones(returns.columns.size - 1, dtype=float) * 1. / returns.columns.size
    w1 = scopt.fmin(negative_sharpe_ratio_n_minus_1_stock, w0, args=(returns, risk_free_rate))
    final_w = sp.append(w1, 1-np.sum(w1))
    final_sharpe = sharpe_ratio(returns, final_w, risk_free_rate)
    return (final_w, final_sharpe)

def objfun(W, R, target_ret):
    stock_mean = np.mean(R,axis=0)
    port_mean = np.dot(W,stock_mean)
    cov = np.cov(R.T)
    port_var = np.dot(np.dot(W, cov), W.T)
    penalty = 2000*abs(port_mean - target_ret)
    return np.sqrt(port_var) + penalty

def calc_efficient_frontier(returns):
    result_means = []
    result_stds = []
    result_weights = []

    means = returns.mean()
    min_mean, max_mean = means.min(), means.max()

    nstocks = returns.columns.size

    for r in np.linspace(min_mean, max_mean, 100):
        weights = np.ones(nstocks)/nstocks
        bounds = [(0,1) for i in np.arange(nstocks)]
        constraints = ({'type':'eq', 'fun':lambda W: np.sum(W)-1})
        results = scopt.minimize(
            objfun,
            weights,
            (returns,r),
            method='SLSQP',
            constraints=constraints,
            bounds=bounds
        )
        if not results.success:
            raise Exception(result.message)
        result_means.append(np.round(r,4))
        std_ = np.round(np.std(np.sum(returns*results.x, axis=1)), 6)
        result_stds.append(std_)

        result_weights.append(np.round(results.x, 5))
    return {'Means':result_means, 'Stds':result_stds, 'Weights':result_weights}

def plot_efficient_frontier(ef_data):
    plt.figure(figsize=(12,8))
    plt.title('Efficient Frontier')
    plt.xlabel('Sandard Deviation of the portfolio (Risk)')
    plt.ylabel('Return of the portfolio')
    plt.plot(ef_data['Stds'], ef_data['Means'], '--')
    plt.show()

def y_f(x):
    return 2+x**2

if __name__ == '__main__':
    #scopt.fmin(y_f,1000)

    closes = get_historical_closes(['MSFT','AAPL','KO'], '2010-01-01', '2014-12-31')
    daily_returns = calc_daily_returns(closes)
    annual_returns = calc_annual_returns(daily_returns)

    portfolio_var = calc_portfolio_var(annual_returns)
    print(np.sqrt(portfolio_var))
    sr = sharpe_ratio(annual_returns)

    optimized_result = optimize_portfolio(annual_returns, 0.0003)
    frontier_data = calc_efficient_frontier(annual_returns)
    plot_efficient_frontier(frontier_data)

