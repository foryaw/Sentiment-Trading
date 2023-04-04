#%%
import os
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import yfinance.shared as shared
import matplotlib.ticker as ticker
from datetime import datetime, timedelta

#%%
START = '2016-01-01'
END = '2021-05-19'
YEAR = int(START[:4])

#%%
# Define a function that calculates the return of a given portfolio over a specified time period
def portfolio_return(portfolio, weights, start_date, end_date):
    # Initialize p_return variable to None
    p_returns = None
    # Extract tickers from the portfolio
    tickers = portfolio[0] + portfolio[1]
    # Convert start and end date strings to datetime objects if necessary
    if not isinstance(start_date, datetime):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    if not isinstance(end_date, datetime):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    # If no weights are specified, assign equal weights to each ticker
    if weights is None:
        weights = pd.Series([1 / portfolio.shape[0]] * portfolio.shape[0])
    # Loop through each ticker in the portfolio
    for ticker in tickers:
        # Load historical price data for the ticker from a CSV file
        hist_data = pd.read_csv(f"./historical prices/{ticker}_hist_price.csv")
        # Convert the 'date' column to datetime objects
        hist_data['Date'] = pd.to_datetime(hist_data['Date'])
        # Filter the data to exclude dates after the end_date
        hist_data = hist_data[hist_data['Date'] <= end_date]
        # Calculate the daily return for each date
        hist_data['Daily Return'] = hist_data['Adj Close'].pct_change()
        # Remove any rows with missing data
        hist_data.dropna(inplace=True)
        # Filter the data to only include dates on or after the start_date
        hist_data = hist_data[hist_data['Date'] >= start_date]
        if hist_data.empty: return None
        # Multiply the daily return for each date by the weight of the corresponding ticker
        if p_returns is None:
            p_returns = hist_data.loc[:, ['Date', 'Daily Return']]
            p_returns['Daily Return'] = p_returns['Daily Return'] * (weights[ticker])
        else:
            return_vector = hist_data['Daily Return'] * (weights[ticker])
            p_returns['Daily Return'] = p_returns['Daily Return'].add(return_vector, fill_value=0)
    p_returns = p_returns[p_returns['Date'] >= start_date]
    p_returns = p_returns[p_returns['Date'] <= end_date]
    # Calculate the cumulative return for the period
    p_returns['Cumulative Return'] = (1 + p_returns['Daily Return']).cumprod() - 1
    # Return the final cumulative return value
    return p_returns['Cumulative Return'].iloc[len(p_returns) - 1], p_returns[['Date', 'Daily Return']]

# Define a function that returns a dictionary of weights for each ticker in a portfolio
def get_weights(portfolio, MCap=None):
    # If no market cap data is specified, assign equal weights to each ticker
    if MCap is None:
        total = len(portfolio[0] + portfolio[1])
        w_long = {ticker: 1 / total for ticker in portfolio[0]}
        w_short = {ticker: -1 / total for ticker in portfolio[1]}
    # Otherwise, calculate the weights based on the market cap of each ticker
    else:
        total = sum(list(map(lambda long: MCap[long] ,portfolio[0]))) - sum(list(map(lambda short: MCap[short] ,portfolio[1])))
        w_long = dict(map(lambda key: (key, MCap[key] / total), portfolio[0]))
        w_short = dict(map(lambda key: (key, -1 * MCap[key] / total), portfolio[1]))
    return w_long | w_short

def select_portfolio(dynamic_df):
    # Sort the DataFrame by 'PN' in descending order
    dynamic_df = dynamic_df.sort_values(by=['PN'], ascending=False)
    # Calculate the 90th percentile of 'PN'
    top_10_percentile = dynamic_df['PN'].quantile(0.9)
    # Calculate the 10th percentile of 'PN'
    bottom_10_percentile = dynamic_df['PN'].quantile(0.1)
    # Get the list of tickers in the top 10th percentile of 'PN'
    long = dynamic_df[dynamic_df['PN'] > top_10_percentile]['ticker'].to_list()
    # Get the list of tickers in the bottom 10th percentile of 'PN'
    short = dynamic_df[dynamic_df['PN'] < bottom_10_percentile]['ticker'].to_list()
    # Create a list of the long and short tickers
    portfolio = [long, short]
    # Return the portfolio
    return portfolio

def download_hist_prices(tickers: pd.Series):
    # Create a directory to store the historical price CSV files
    dir = './historical prices'
    if not os.path.exists(dir):
        os.makedirs(dir)
    # Loop through each ticker in the input Series
    for ticker in tickers:
        # Download the historical price data for the ticker using Yahoo Finance
        hist_price = yf.download(ticker, start='2015-01-01', end='2022-12-31')
        # Create a file path to save the CSV file
        path = f"./{dir}/{ticker}_hist_price.csv"
        # Save the CSV file to the specified path
        if not os.path.exists(path):
            hist_price.to_csv(path)
        # Check if the ticker is invalid, and if so, remove it from the list of tickers and delete its CSV file
        invalid_ticker = None if (len(list(shared._ERRORS.keys())) == 0) else list(shared._ERRORS.keys())[0]
        if invalid_ticker:
            tickers = tickers[tickers != invalid_ticker]
            os.remove(f"./{dir}/{invalid_ticker}_hist_price.csv")
            print(f"removed {invalid_ticker}")
    # Return the list of tickers
    return tickers

def get_measurement(ticker, measurement):
    # Make a GET request to the Yahoo Finance page for the specified ticker
    response = requests.get(f'https://finance.yahoo.com/quote/{ticker}?p={ticker}')
    # Parse the HTML content of the page using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')
    # Find all tables on the page
    stock_data = soup.find_all("table")
    # Loop through each table and scrape the desired measurement
    for table in stock_data:
        trs = table.find_all('tr')
        for tr in trs:
            tds = tr.find_all('td')
            # If the measurement is found in the first column of a table row, return the value in the second column
            if len(tds) >= 2:
                if measurement.lower() in tds[0].get_text().lower():
                    return(tds[1].get_text())
                
def calculate_market_cap(df):
    # iterate over each row of the DataFrame
    for index, row in df.iterrows():
        # get the ticker symbol and date for the current row
        tickerSymbol = row['ticker']
        date = row['date']
        
        # retrieve the historical stock data for the tickerSymbol and date
        stockData = yf.Ticker(tickerSymbol)
        stockPrice = stockData.history(start=date, end=date)['Adj Close'][0]
        
        # get the stock split data and calculate the split factor up to the current date
        splitData = stockData.splits
        splitFactor = 1.0
        for i in range(len(splitData)):
            splitdate = splitData.index[i].strftime('%Y-%m-%d')
            if splitdate <= date:
                splitFactor *= splitData[i]

        # calculate the market capitalization for the current row
        # by multiplying the PN by the split factor and the stock price
        marketCap = row['PN'] * splitFactor * stockPrice
        
        # update the DataFrame with the calculated market capitalization
        df.loc[index, 'Market Cap'] = marketCap

#%%
# read data from Excel file
df = pd.read_csv("./sp100.csv", header=0)

#%%
# get unique tickers and drop unnecessary columns
tickers = df['ticker'].unique()
# remove redundent column
df = df.drop(['File', 'positive', 'neutral', 'negative'], axis=1)
# convert the 'date' column to datetime format, specifying the format and the dayfirst parameter
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y', dayfirst=True)

#%%
# valid tickers
# tickers = download_hist_prices(tickers)
tickers = ['ABT', 'ABBV', 'ACN', 'CAN', 'ADBE', 'AMD', 'AMZN', 'AIG', 'AMT', 'AMGN', 'AAPL',
        'BK', 'BLK', 'AVGO', 'COF', 'C', 'CL', 'CMCSA', 'XOM', 'NFLX', 'NVDA', 'PYPL', 'PEP',
        'PFE', 'PM', 'PG', 'QCOM', 'SPG', 'SO', 'SBUX', 'TMUS', 'TGT', 'TSLA', 'UNP', 'UPS',
        'UNH', 'USB', 'VZ', 'V', 'WBA', 'WMT', 'WFC', 'BAC', 'AVGO ', 'MMM', 'GOOGL', 'MO',
        'T', 'BKNG', 'BMY', 'CAT', 'CHTR', 'CVX', 'CSCO', 'KO', 'COP', 'COST', 'CVS', 'DHR',
        'DOW', 'DUK', 'LLY', 'EMR', 'EXC', 'FDX', 'F', 'GD', 'GE', 'GM', 'GILD', 'GS', 'HD',
        'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KHC', 'LIN', 'LMT', 'LOW', 'MA', 'MCD', 'MDT',
        'MRK', 'META', 'MSFT', 'MDLZ', 'MS', 'NEE', 'NKE', 'ORCL', 'RTX', 'CRM', 'TXN',
        'TMO', 'DIS']

#%%
# filter dataFrame to only include valid tickers
df = df[df['ticker'].isin(tickers)]
# filter dataFrame to only include dates after {YEAR-1}-09-01
df = df[df['date'] > f'{YEAR-1}-06-01']

#%%
# Sort the original dataframe by date
dynamic_df = df.sort_values('date')
# filter DataFrame to only include dates before START
dynamic_df = dynamic_df[dynamic_df['date'] < START]
# Group the sorted dataframe by ticker
grouped_df = dynamic_df.groupby('ticker')
# Select the row corresponding to the earliest date for each ticker
dynamic_df = grouped_df.first()
# Reset the index of the resulting dataframe
dynamic_df = dynamic_df.reset_index()
# Reorder columns and sort by date
dynamic_df = dynamic_df[['date', 'ticker', 'PN']].sort_values('date')
# Reset the index of the resulting dataframe
dynamic_df.reset_index(drop=True, inplace=True)
# Remove any rows with missing values
dynamic_df.dropna(inplace=True)

#%%
# Select the portfolio based on initial universe
portfolio = select_portfolio(dynamic_df)
# Initialize the start date for the analysis
start_date = datetime.strptime(START, '%Y-%m-%d') + timedelta(days=-1)
# Initialize the end date for the analysis
end_date = datetime.strptime(START, '%Y-%m-%d') + timedelta(days=-1)
# Filter the dataframe to only include entries after start date
df = df[df['date'] >= start_date]
# Sort df by date
df.sort_values(by='date', inplace=True)
# Initialize the overall return variable
overall_return = 1
strategy = pd.DataFrame(columns=['Date', 'Daily Return'])
print(df)

#%%
# Loop through each row in the DataFrame
for index, row in df.iterrows():
    # Set the start date to be the end date of the previous row
    start_date = end_date + timedelta(days=1)
    # Set the end date to be the date of the current row
    end_date = row['date']
    # If the start date is not the same as the end date, calculate the portfolio return
    # Get the weights for the current portfolio
    weights = get_weights(portfolio)
    # Calculate the return for the current portfolio
    p_returns = portfolio_return(portfolio, weights, start_date, end_date)
    if p_returns is None: continue
    overall_return *= 1 + p_returns[0]
    print(f'overall return up to {end_date} = {overall_return - 1:.4%}')
    strategy = pd.concat([strategy,p_returns[1]],ignore_index=True)

    # If the ticker is already in the dynamic_df, update the PN for that ticker
    if row['ticker'] in dynamic_df['ticker'].tolist():
        dynamic_df.loc[dynamic_df['ticker'] == row['ticker'], 'date'] = row['date']
        dynamic_df.loc[dynamic_df['ticker'] == row['ticker'], 'PN'] = row['PN']
    # Otherwise, add a new row to the dynamic_df
    else:
        dynamic_df.loc[dynamic_df.index.max() + 1] = row
    
    # Select the portfolio using the updated dynamic_df
    portfolio = select_portfolio(dynamic_df)
overall_return -= 1
# Print the overall return for the analysis period   
print(f"overall return over period from {START} to {END}\n= {overall_return:.4%}")
strategy.to_csv('./strategy_dailyReturn.csv')

#%%
sp500 = yf.download(tickers='^GSPC', start=START, end=END, interval='1d')
sp500['Daily Return'] = sp500['Adj Close'].pct_change()
sp500['Cumulative Return'] = (1 + sp500['Daily Return']).cumprod() - 1
sp500.reset_index(inplace=True)
strategy = pd.read_csv('./strategy_dailyReturn.csv')
strategy['Cumulative Return'] = (1 + strategy['Daily Return']).cumprod() - 1

sp500['Date'] = pd.to_datetime(sp500['Date'])
strategy['Date'] = pd.to_datetime(strategy['Date'])

#%%
start_date = datetime.strptime(START, '%Y-%m-%d')
end_date = datetime.strptime(END, '%Y-%m-%d')
years = (end_date - start_date).days / 365.25
annual_return = ((1 + overall_return) ** (1 / years)) - 1
annual_vol = np.sqrt(252) * strategy['Daily Return'].std()
print(f'annual return = {annual_return}')
print(f'annual vol = {annual_vol}')
print(f'sharpe ratio = {annual_return/annual_vol}')

#%%
plt.plot(strategy['Date'], strategy['Cumulative Return'], label='Strategy')
plt.plot(sp500['Date'], sp500['Cumulative Return'], label='SP500')

plt.xlim(datetime.strptime(START, '%Y-%m-%d'), datetime.strptime(END, '%Y-%m-%d'))
# automatically adjust y-axis limits
plt.autoscale(enable=True, axis='y')
plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter())

plt.legend()
plt.show()