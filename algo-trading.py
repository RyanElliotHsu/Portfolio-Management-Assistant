import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from dateutil.relativedelta import relativedelta
import talib

# Step 1: Data Preparation

# year_ago = 1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
# interval = 1d,1wk,1mo 
def get_adjusted_price(symbols, interval="1mo",year_ago="5y"):
    sym_string = " ".join(symbols)
    df = yf.download(sym_string,group_by = 'ticker',interval=interval,period=year_ago)
    # extract the second-level (Adj Close) and drop the level with symbol name left
    df = df.xs("Adj Close",axis =1, level=1, drop_level=True)
    
    return (df,df.columns)

price_data = get_adjusted_price(["AAPL","TSLA"],interval="1d",year_ago="5y")[0]["AAPL"]


# Step 2: Strategy Implementation
# Define strategy parameters
short_window = 20  # Short moving average window
long_window = 50  # Long moving average window

# Define trading signals
def generate_signals_MA(data):
    signals = pd.DataFrame(index=data.index)
    signals['signal'] = 0.0

    # Generate buy signals
    signals['short_mavg'] = data.rolling(window=short_window, min_periods=1, center=False).mean()
    signals['long_mavg'] = data.rolling(window=long_window, min_periods=1, center=False).mean()
    signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1.0, 0.0)   

    # Generate sell signals
    signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] < signals['long_mavg'][short_window:], -1.0, signals['signal'][short_window:])
    return signals

window = 50  # Bollinger Bands window
deviation = 3  # Bollinger Bands deviation
rsi_period = 6  # RSI period
macd_fast_period = 12  # MACD fast period
macd_slow_period = 26  # MACD slow period
macd_signal_period = 9  # MACD signal period

# Define trading signals
def generate_signals_M_Reversion(data):
    signals = pd.DataFrame(index=data.index)
    signals['signal'] = 0

    # Calculate Bollinger Bands
    signals['upper'], signals['middle'], signals['lower'] = talib.BBANDS(data, timeperiod=window, nbdevup=deviation, nbdevdn=deviation)

    # Calculate RSI
    signals['rsi'] = talib.RSI(data, timeperiod=rsi_period)

    # Calculate MACD
    # signals['macd'], signals['signal_line'], _ = talib.MACD(data, fastperiod=macd_fast_period, slowperiod=macd_slow_period, signalperiod=macd_signal_period)

    # Generate buy signals
    signals['signal'] = np.where((data < signals['lower']) & (signals['rsi'] < 30), 1.0, signals['signal'])

    # Generate sell signals
    signals['signal'] = np.where((data> signals['upper']) & (signals['rsi'] > 70) , -1.0, signals['signal'])
    
    clean_sign = []
    # first transaction need to be buy
    
    current = -1
    for i in signals['signal']:
        if i == 0:
            clean_sign.append(0)
        elif i != current:
            clean_sign.append(i)
            current = i
        else:
            clean_sign.append(0)
    signals['signal'] = clean_sign  
    
    return signals

# Step 3: Portfolio Management
# Define initial portfolio equity and position size
initial_equity = 100000.0
position_size = 10000.0

# Step 4: Backtesting Loop
signals = generate_signals_M_Reversion(price_data)

def make_portfolio(signals,price_data):
    portfolio = pd.DataFrame(index=signals.index)
    portfolio['signal'] = signals['signal']
    portfolio['price'] = price_data

    # Step 5: Performance Metrics
    portfolio['holdings'] = (portfolio['signal'].cumsum()) * position_size * portfolio['price']
    portfolio['cash'] = initial_equity - ((portfolio['signal'] * portfolio['price']).cumsum() * position_size)
    portfolio['total_equity'] = portfolio['cash'] + portfolio['holdings'] 
    portfolio['returns'] = portfolio['total_equity'].pct_change()
    portfolio['cumulative_returns'] = (1.0 + portfolio['returns']).cumprod() - 1.0
    portfolio['drawdown'] = (portfolio['cumulative_returns'] - portfolio['cumulative_returns'].cummax())

    return portfolio

portfolio = make_portfolio(signals,price_data)

# Step 6: Visualization
# Plot portfolio equity curve and drawdown
import matplotlib.pyplot as plt

def visualize(portfolio):
    buy = portfolio[portfolio["signal"]==1].index
    sell = portfolio[portfolio["signal"]==-1].index

    fig, ax1 = plt.subplots()
    ax1.plot(portfolio['cumulative_returns'])
    ax1.scatter(buy,portfolio['cumulative_returns'][buy],Label="Buy")
    ax1.scatter(sell,portfolio['cumulative_returns'][sell],Label="Sell")
    ax1.legend()
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Cumulative Returns', color='b')
    '''
    ax2 = ax1.twinx()
    ax2.plot(portfolio['drawdown'], 'r')
    ax2.set_ylabel('Drawdown', color='r') 
    '''
    plt.title('Backtesting Results')
    plt.show()


    plt.plot(price_data,linewidth=1)
    plt.scatter(buy,portfolio['price'][buy],label="Buy",s=2,c="g",Label="Buy")
    plt.scatter(sell,portfolio['price'][sell],label="Sell",s=2,c="r",Label="Sell")
    plt.legend()
    plt.show()     

