import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
from scipy.optimize import minimize, LinearConstraint

def get_adjusted_price(symbols, interval="1mo",year_ago="5y"):
    sym_string = " ".join(symbols)
    df = yf.download(sym_string,group_by = 'ticker',interval=interval,period=year_ago)
    # extract the second-level (Adj Close) and drop the level with symbol name left
    df = df.xs("Adj Close",axis =1, level=1, drop_level=True)
    return (df,df.columns)
    
def get_risk_premium(df):
     df=(df.reset_index(drop=True).pct_change())
     return df

def objective_SR (weight,rtn,cov_mat):
     rtn = np.dot(weight,rtn)
     std = np.sqrt(np.matmul(np.matmul(weight,cov_mat),weight))
     sr = rtn/std
     return -sr # minimize negative Sharpe == maximize sharpe

class portfolio:
    def __init__(self,symbol_list,interval="1mo",year_ago="5y",short_sale_constrain = True, ind_weight_constrain = None, invested_cash = 100000):

        annualize = {"1d":256,"1wk":52,"1mo":12}

        self.price_df,self.symbol_list = get_adjusted_price(symbol_list,interval=interval,year_ago=year_ago)
        self.short_sale_constrain = short_sale_constrain
        self.ind_weight_constrain = ind_weight_constrain
        self.num_stock = len(self.symbol_list)
        self.cov_mat = get_risk_premium(self.price_df).cov() * (annualize[interval]) 
        self.ann_rtn_mat = get_risk_premium(self.price_df).mean() * (annualize[interval]) 
        self.weight_mat = np.ones(self.num_stock)/self.num_stock
        self.sharpe = 0
        self.ann_rtn = 0 
        self.std = 0
        self.invest_cash = invested_cash

    # show stock price time series plot
    def show_price_plot(self):
        for i in self.price_df.columns:
          plt.plot(self.price_df[[i]],label=i)
        plt.legend()
        plt.show()
       
    # print sharpe, return and standard deviation
    def visualize_stat(self):
        print("Sharpe Ratio:",round(self.sharpe,4))
        print("Annul return:",round(self.ann_rtn,4)*100,"%")
        print("Standard deviation:", round(self.std,4)*100,"%")

    # calculate num of stocks to buy
    def num_stock_to_buy(self,current_price,equal=False):
        if equal:
          equal_weight = (np.ones(self.num_stock)/self.num_stock)
          return ( equal_weight * self.invest_cash/ current_price).astype(int).rename(index=lambda x:"number of stock")
        else:
          return (self.weight_mat * self.invest_cash/ current_price).astype(int).rename(index=lambda x:"number of stock")

  
    def max_SR(self):

        # the max weight range is from -300% to 300% if there is no constraint

        # ind weight constraint
        upper = (self.ind_weight_constrain) if (self.ind_weight_constrain) else 3
        # short sale constrain
        lower = 0 if self.short_sale_constrain else -3
        bound = [(lower,upper)] * self.num_stock
        #sum of weight should be 1
        con = LinearConstraint(np.ones(self.num_stock), lb=1, ub=1)
        # minimize neg Sharpe = max Sharpe
        result = minimize(objective_SR,x0=self.weight_mat,args=(self.ann_rtn_mat,self.cov_mat),bounds=bound,constraints=con)

        # update stat 
        self.sharpe = -(result["fun"])
        self.weight_mat = result["x"]
        self.ann_rtn = np.dot(self.weight_mat,self.ann_rtn_mat)
        self.std = np.sqrt(np.matmul(np.matmul(self.weight_mat,self.cov_mat),self.weight_mat))

        # visualize stat
        self.visualize_stat()

        # visualize weight table 
        display_df = pd.DataFrame({"Symbol":self.symbol_list,"Weight":list(map(lambda x:str(round(x,3)*100)+"%", self.weight_mat)),"# stock":self.num_stock_to_buy(self.price_df[-1:]).loc["number of stock"].values})
        
        print(display_df)

    def back_test(self):
      # daily price
      daily_price,sym = get_adjusted_price(self.symbol_list,interval="1d",year_ago="5y")
      # spy
      spy = yf.download("SPY",interval="1d",period="5y")["Adj Close"]

      price_len = len(daily_price)
      # splice total stock data into 1 and 3 years 
      three_yr = daily_price.iloc[price_len-255*3:][self.symbol_list]
      three_yr_spy= spy[price_len-255*3:]
      one_yr = daily_price.iloc[price_len-255:][self.symbol_list]
      one_yr_spy= spy[price_len-255:]

      # calculate the starting porfolio and the trend 
      three_yr_start = three_yr.iloc[0]

      three_yr_port = self.num_stock_to_buy(three_yr_start)
      three_yr_num_sp = int(self.invest_cash/three_yr_spy[0])
      three_yr_equal = self.num_stock_to_buy(three_yr_start,equal=True)
      
      port_trend = np.sum(three_yr * list(three_yr_port),axis=1)
      spy_trend = three_yr_spy * three_yr_num_sp
      equal_trend = np.sum(three_yr * list(three_yr_equal),axis=1)
      plt.plot(port_trend,label="Portfolio")
      plt.plot(spy_trend,label="SPY")
      plt.plot(equal_trend,label="Equal Weighted")
      plt.title("Three-year trend SPY vs Porfolio vs Equal Weight")
      plt.axhline(self.invest_cash,linestyle = '--')
      plt.legend()
      plt.show()
      # calculate the starting porfolio and the trend 
      one_yr_start = one_yr.iloc[0]

      one_yr_port = self.num_stock_to_buy(one_yr_start)
      one_yr_equal = self.num_stock_to_buy(one_yr_start,equal=True)
      one_yr_num_sp = int(self.invest_cash/one_yr_spy[0])

      port_trend = np.sum(one_yr * list(one_yr_port),axis=1)
      spy_trend = one_yr_spy * one_yr_num_sp
      equal_trend = np.sum(one_yr * list(one_yr_equal),axis=1)
      plt.plot(port_trend,label="Portfolio")
      plt.plot(spy_trend,label="SPY")
      plt.plot(equal_trend,label="Equal Weighted")
      plt.title("One-year trend SPY vs Porfolio vs Equal Weight")
      
      plt.legend()
      plt.axhline(self.invest_cash,linestyle = '--')
      plt.show()


client1= portfolio(["AAPL","KO","IBM","GOOG"],year_ago="5y",short_sale_constrain = False, ind_weight_constrain = None)
client1.show_price_plot()
client1.max_SR()
client1.back_test()