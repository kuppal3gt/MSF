

# Please change the following to your own PAPER api key and secret
# or set them as environment variables (ALPACA_API_KEY, ALPACA_SECRET_KEY).
# You can get them from https://alpaca.markets/
## API Credentials
#api_key <- "AKFHWSO5B9N004386UVD"
api_key = "AKFHWSO5B9N004386UVD"
secret_key = "8yNoeYYdaYNBPUrcRYZSpxwIEuFgOCtklf3HrK7i"
API_KEY=api_key
SECRET_KEY=secret_key
#### We use paper environment for this example ####
paper = False # Please do not modify this. This example is for paper trading only.
####

# Below are the variables for development this documents
# Please do not change these variables
trade_api_url = None
trade_api_wss = None
data_api_url = None
stream_data_wss = None
import ta

from ta.momentum import RSIIndicator, ROCIndicator, StochasticOscillator
import os
import pytz  # For timezone conversion
import pandas as pd
import pandas as pd
import nest_asyncio
import alpaca
import requests
from bs4 import BeautifulSoup
import time
from datetime import date
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from textblob import TextBlob
import pandas as pd
#import xgboost as xgb
import numpy as np
import nltk
import re
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()


if api_key is None:
    api_key = os.environ.get('ALPACA_API_KEY')

if secret_key is None:
    secret_key = os.environ.get('ALPACA_SECRET_KEY')
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
#from call_alpaca_dev_03262025 import *
warnings.filterwarnings('ignore')
    
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo

from alpaca.trading.client import TradingClient
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.historical.corporate_actions import CorporateActionsClient
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.trading.stream import TradingStream
from alpaca.data.live.stock import StockDataStream
from alpaca.data.historical import StockHistoricalDataClient, NewsClient
from alpaca.data.historical.screener import ScreenerClient
from alpaca.trading.requests import GetOptionContractsRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.enums import DataFeed

import alpaca.data
from alpaca.data.requests import (
    CorporateActionsRequest,
    StockBarsRequest,
    StockQuotesRequest,
    StockTradesRequest,
    MostActivesRequest,
    StockLatestTradeRequest,
    MarketMoversRequest,
    NewsRequest
)
from alpaca.trading.requests import (
    ClosePositionRequest,
    GetAssetsRequest,
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    StopLimitOrderRequest,
    StopLossRequest,
    StopOrderRequest,
    TakeProfitRequest,
    TrailingStopOrderRequest,
    ReplaceOrderRequest,
    GetPortfolioHistoryRequest
)

from alpaca.data.requests import (
    OptionBarsRequest,
    OptionTradesRequest,
    OptionLatestQuoteRequest,
    OptionLatestTradeRequest,
    OptionSnapshotRequest,
    OptionChainRequest    
)

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import OptionChainRequest
from alpaca.data.requests import (
    CryptoBarsRequest,
    CryptoQuoteRequest,
    CryptoTradesRequest,
    CryptoLatestQuoteRequest
    )

from alpaca.trading.models import (
    PortfolioHistory
)
from alpaca.trading.enums import (
    AssetExchange,
    AssetStatus,
    OrderClass,
    OrderSide,
    OrderType,
    QueryOrderStatus,
    TimeInForce,
)    



historical_client = StockHistoricalDataClient(api_key, secret_key)
trade_client = TradingClient(api_key=api_key, secret_key=secret_key, paper=paper, url_override=trade_api_url)
news_client = NewsClient(api_key, secret_key)
# Initialize Crypto Data Client
crypto_client = CryptoHistoricalDataClient()

# Define EST timezone
EST = pytz.timezone("US/Eastern")
# Risk Management Parameters
RISK_TOLERANCE = 0.01  # 1% max risk per trade
REWARD_THRESHOLD = 0.01  # 5% minimum profit per trade

POSITION_SIZE = 250  # dollars
PROFIT_TARGET = 0.05  # 0.5%
STOP_LOSS = 0.1     # 0.7%
MAX_HOLD_TIME_MIN = 60*24*60 #120
CANCEL_ORDER_AFTER_MIN = 1
TRAIL_PERCENT = 0.01 #2 / 100  # 0.15%

UNDERLYING_STOCKS = ["NVDA", "AVGO", "AMD", "QCOM","TXN",
"MRVL","LRCX","ADI","KLAC","TSM","AMAT","ASML","NXPI",
"MU","MPWR","MCHP","INTC","ON","TER","ENTG","SWKS","ONTO"]

# Stock Weights as of latest index data (percentages)
STOCK_WEIGHTS = {
  
    'TXN': 8.21, 'NVDA': 7.99, 'AVGO': 7.58, 'AMD': 7.36, 'QCOM': 6.95,
    'INTC': 4.63, 'KLAC': 4.30, 'AMAT': 4.20, 'LRCX': 4.17, 'NXPI': 3.91,
    'MU': 3.89, 'MPWR': 3.89, 'ADI': 3.86, 'ASML': 3.82, 'TSM': 3.79,
    'MCHP': 3.39, 'MRVL': 2.97, 'ON': 2.48, 'TER': 1.97, 'ENTG': 1.79,
    'SWKS': 1.51, 'LSCC': 0.98, 'QRVO': 0.94, 'ONTO': 0.90, 'OLED': 0.86,
    'ASX': 0.84, 'STM': 0.81, 'UMC': 0.75, 'MKSI': 0.72, 'ARM': 0.57
}


SOXL = "SOXL"
SOXS = "SOXS"

SCORING_THRESHOLDS = {
    'strong_buy': 6,
    'buy': 5,
    'neutral': 3,
    'sell': 1
}

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')
import requests
from io import StringIO
from bs4 import BeautifulSoup

def parse_symbols_of_interest(filename):
    df = pd.read_csv(filename) 
    # 1. Large Cap stocks (Market Cap > 10 billion)
    large_cap = df[df['Market Capitalization'] > 1e10]
    large_cap_symbols = large_cap['Symbol'].tolist()
    
    # 2. Stocks with Dividends (non-null Dividend Pay Date)
    stocks_with_dividends = df[(df['Dividend Pay Date'].notna()) & (df['Dividend Rate']>=1)]
    dividend_symbols = stocks_with_dividends['Symbol'].tolist()
    
    # 3. Stocks with High P/E Ratio (P/E Ratio > 20)
    high_pe_ratio = df[(df['P/E Ratio'] > 5) & (df['Earnings/Share']>0)]
    high_pe_symbols = high_pe_ratio['Symbol'].tolist()
    
    # 4. Stocks with Analyst Rating of "Buy" (assuming the rating is stored as a string containing "Buy")
    stocks_with_buy_rating = df[df['Ave. Analyst Rating'].str.contains('Buy', case=False, na=False)]
    buy_rating_symbols = stocks_with_buy_rating['Symbol'].tolist()
    
    # Create an aggregated list of all unique symbols across the categories
    aggregated_symbols = list(set(large_cap_symbols + dividend_symbols + high_pe_symbols + buy_rating_symbols))
    
    # Display the aggregated list of symbols
    #print("Aggregated List of Symbols:")
    #print(aggregated_symbols)
    
    # Compute the intersection of the four symbol lists
    intersection_symbols = list(set(large_cap_symbols) & set(dividend_symbols) & set(high_pe_symbols) & set(buy_rating_symbols))
    print("\nIntersection of Symbols Across All Categories:")
    print(intersection_symbols)
    return aggregated_symbols, intersection_symbols
  
def prepare_stock_symbols(source_file):
    # List of top 50 S&P 500 stock symbols (a sample of well-known companies as placeholders)
    sp500_symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK.B', 'UNH', 'V',
        'JNJ', 'PG', 'HD', 'MA', 'DIS', 'VZ', 'PYPL', 'NFLX', 'INTC', 'CSCO', 'KO',
        'PFE', 'XOM', 'MRK', 'ABT', 'PEP', 'NKE', 'ORCL', 'MCD', 'INTU', 'AMD',
        'WMT', 'CRM', 'BA', 'GE', 'T', 'CVX', 'BA', 'LLY', 'WFC', 'GS', 'IBM',
        'MDT', 'C', 'CAT', 'HCA', 'AMT', 'MU', 'SPGI', 'DUK', 'CL', 'UPS', 'NEE'
    ]
    
    all_symbols, highdividend_symbols_of_interest=parse_symbols_of_interest("~/Downloads/allquotes_yahoo.csv")
    
    # List of top 10 ETFs (Exchange Traded Funds) and additional symbols
    etf_symbols = [
        'SPY', 'IVV', 'VOO', 'VTI', 'QQQ', 'IWM', 'DIA', 'EEM', 'GLD', 'XLF',
        'XLRE', 'SOXS', 'SOXL', 'NVDX', 'CWEB', 'HAO', 'CRSH', 'SCHD', 'UNG'
    ]
    
    # Simulate daily gainers list (for the sake of this example, just use random symbols)
    # In real-world use, you would fetch this data from an external source.
    daily_gainers = ["VVPR","STFS","IMTE","POAI","MLGO","ADTX",
         "OSRH","PTPI","VVPR","DMN","CTAS","LYT",
         "GTEC","DRMA","CISO",
        "PRTG","LXRX","TWG","BOWN",
        "CTOR", "ICCT", "NWTG", "MYSZ", "TGL",  "DOMH", "MULN", "CDT",  "HMST", "CORT", "COOP", "HCWC", "LNZA",
        "BIAF","RSLS","SATX","GATE","GRI","IBO","ICCT","IBG","DGLY","CNTM","SBFM","GLXG"] #['TSLA', 'AMD', 'NVDA', 'AMZN', 'AAPL', 'MSFT', 'META', 'NKE', 'BA', 'PEP']
    
    # Date range from Sep 1, 2024, to April 8, 2025
    date_range = pd.date_range(start='2024-09-01', end='2025-04-08', freq='B')  # 'B' is for business days
    
    # Combine the lists to create the DataFrame with stock symbols in Column A and date in Column B
    symbols = sp500_symbols + etf_symbols + highdividend_symbols_of_interest + daily_gainers
    
    # Repeat the symbols for each date in the date range
    data = {
        'Symbol': np.tile(symbols, len(date_range)),
        'Date': np.repeat(date_range, len(symbols))
    }
    
    # Create the DataFrame
    df = pd.DataFrame(data)
    
    df.to_csv(source_file, index=False)

def update_stock_symbols(source_file):
    # Load the existing DataFrame (from a file, database, or memory)
    df = pd.read_csv(source_file)  # Example of loading from CSV
    
    # Get today's date
    today = datetime.today().date()
    
    # Check if today is a valid trading day (skip weekends and holidays)
    if today.weekday() < 5:  # 0 = Monday, 4 = Friday
        # Define the new row of symbols (top 50 SP500 stocks, ETFs, and daily gainers)
        symbols = sp500_symbols + etf_symbols + daily_gainers
    
        # Create a new DataFrame for today's date
        new_data = {
            'Symbol': symbols,
            'Date': [today] * len(symbols)
        }
    
        new_df = pd.DataFrame(new_data)
    
        # Append the new data to the existing DataFrame
        df = pd.concat([df, new_df], ignore_index=True)
    
        # Optionally, save the updated DataFrame back to a CSV or database
        df.to_csv(source_file, index=False)

def daily_fund_holdings1():
    url = "https://www.direxion.com/product/daily-semiconductor-bull-bear-3x-etfs"

    # Fetch HTML content
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Find the table containing the holdings
    table = soup.find("table", {"id": "holdings-table"})
    
    # Extract headers and rows
    data = []
    if table:
        rows = table.find_all("tr")
        for row in rows[1:]:  # skip header
            cols = row.find_all("td")
            if len(cols) >= 2:
                symbol = cols[2].get_text(strip=True)
                weight_text = cols[9].get_text(strip=True).replace("%", "")
                try:
                    weight = float(weight_text)
                    data.append((symbol, weight))
                except ValueError:
                    continue
    
    # Convert to dictionary
    holdings_dict = dict(data)
    
    # Print or use the holdings dictionary
    print(holdings_dict)

def daily_fund_holdings(url="https://www.direxion.com/holdings/SOXL.csv"):
    # URL to the daily holdings CSV file for SOXL
    holdings_url = url #'https://www.direxion.com/product/daily-semiconductor-bull-bear-3x-etfs#show-daily-holdings'
    
    # Send a GET request to fetch the CSV data
    response = requests.get(holdings_url)
    response.raise_for_status()  # Raise an exception for HTTP errors
    csv_content = StringIO(response.text)
    holdings_df= pd.read_csv(csv_content, skiprows=5)
    #print(holdings_df)
    # Parse the CSV data into a pandas DataFrame
    #holdings_df = pd.read_csv(StringIO(response.text))
    
    # Extract the relevant columns: 'Symbol' and 'Weight (%)'
    holdings_df = holdings_df[['StockTicker', 'HoldingsPercent']]
    
    # Convert the 'Weight (%)' column to numeric, handling any non-numeric values
    holdings_df['HoldingsPercent'] = pd.to_numeric(holdings_df['HoldingsPercent'], errors='coerce')
    
    # Normalize weights so they sum to 100
    total_weight = holdings_df['HoldingsPercent'].sum()
    holdings_df['Weight'] = (holdings_df['HoldingsPercent'] / total_weight) * 100
    # Drop rows with missing values
    holdings_df.dropna(inplace=True)
    
    # Create a dictionary mapping stock symbols to their weights
    holdings_dict = dict(zip(holdings_df['StockTicker'], holdings_df['HoldingsPercent']))
    
    # Output the holdings dictionary
    #print(holdings_dict)
    return(holdings_dict)
    
def etf_weights(filename):
    # Load the CSV file
    file_path = filename  # Update path if needed
    df = pd.read_csv(file_path)
    
    # Extract symbols (Column C) and weights (Column J)
    df_cleaned = df.iloc[:, [2, 9]].dropna()
    df_cleaned.columns = ["Symbol", "Weight"]
    
    # Convert to dictionary
    stock_weight_dict = dict(zip(df_cleaned["Symbol"], df_cleaned["Weight"]))
    
    # Optional: Filter stocks with weight >= 1%
    filtered_stock_weights = {symbol: round(weight, 2) for symbol, weight in stock_weight_dict.items() if weight >= 1}
    
    # Print or use
    print(filtered_stock_weights)
    
def score_stock(indicators):
    score = 0
    if indicators['close'] > indicators['l5_MA']: score += 1
    if indicators['close'] > indicators['l20_MA']: score += 1
    if indicators['lRSI_14'] and indicators['lRSI_14'] > 40 and indicators['lRSI_14'] <80: score += 1
    if indicators['lPercentChange'] and indicators['lPercentChange'] > 0: score += 1
    if indicators['lfastK'] and indicators['lfastK'] > 40 and indicators['lfastK']<70: score += 1
    if indicators['lfastD'] and indicators['lfastD'] > 20 and indicators['lfastD']<70: score += 1
    return score


  
def get_stock_score(stock,timeframe="DAY"):
    start_date=date.today()
    #print(start_date)
    indicators, prev = calculate_indicators_hist_date(stock, start_date,timeframe)
    #print(indicators)
    score1 = score_stock(indicators)
    score2 = score_stock(prev)
    return score1, score2
  
def analyze_sector(start_date,timeframe="Minute"):
    scores = {}
    STOCK_WEIGHTS=daily_fund_holdings()

    UNDERLYING_STOCKS=STOCK_WEIGHTS.keys()
    for stock in UNDERLYING_STOCKS:
        try:
            
            indicators, prev = calculate_indicators_hist_date(stock, start_date,timeframe)
            #print(indicators)
            score = score_stock(indicators)
            #score=6
            weight = STOCK_WEIGHTS.get(stock, 0)
            weighted_score = score * (weight / 100)
            #total_weight += weight
            scores[stock] = weighted_score #/len(UNDERLYING_STOCKS)
        except Exception as e:
            print(f"Error processing {stock}: {e}")
            scores[stock] = 0
    return scores

def soxl_soxs_compare(start_date=date.today()):
    
    print("Analyzing sector for long term trend...")
    stock_scores = analyze_sector(start_date,"DAY")
    recommendation1,etf_score1  = get_soxl_soxs_recommendation(stock_scores) #aggregate_score(stock_scores)

    #print("\n--- Underlying Stock Scores ---")
    #for stock, score in stock_scores.items():
    #    print(f"{stock}: {score}")

    #print("\n--- Strategy Recommendation (Long)---")
    #print(recommendation1)
    
    #print("Analyzing sector for daily trend...")
    stock_scores = analyze_sector(start_date,"Minute")
    recommendation2,etf_score2 = get_soxl_soxs_recommendation(stock_scores) #aggregate_score(stock_scores)

    if etf_score1<1 and etf_score2>4 and recommendation1==SOXS:
        recommendation1=recommendation1+":Entry"
        if etf_score2>5 and recommendation2 is not None:
          
            recommendation2=recommendation2+":Sell"
            
    elif etf_score1>5 and etf_score2<2 and recommendation1==SOXL:
        recommendation1=recommendation1+":Entry"
        if etf_score2<0.25 and recommendation2 is not None:
          
            recommendation2=recommendation2+":Sell"
    elif etf_score2>3 and etf_score2<4 and recommendation2 is not None:
        recommendation2=recommendation2+":Entry"
    #print("\n--- Underlying Stock Scores ---")
    #for stock, score in stock_scores.items():
    #    print(f"{stock}: {score}")

    #print("\n--- Strategy Recommendation (Daily)---")
    #print(recommendation2)etf_score1, ,etf_score2 
    return(recommendation1,recommendation2)
def get_soxl_soxs_recommendation(stock_scores):
    etf_score=round(sum(stock_scores.values()),2)
    print(etf_score)
    recommendation = SOXL if etf_score > SCORING_THRESHOLDS['neutral']+0.5 else SOXS if etf_score < SCORING_THRESHOLDS['neutral']-0.5 else None
    
   
    return recommendation,etf_score
    
def aggregate_score(stock_scores):
    total_score = sum(stock_scores.values())
    avg_score = total_score / len(stock_scores)
    if avg_score >= SCORING_THRESHOLDS['strong_buy']:
        return "STRONG BUY SOXL"
    elif avg_score >= SCORING_THRESHOLDS['buy']:
        return "BUY SOXL"
    elif avg_score >= SCORING_THRESHOLDS['neutral']:
        print(avg_score)
        return "NEUTRAL / HOLD"
    else:
        return "BUY SOXS"  

def backtest_strategy_sox():
    from datetime import date
    end_date = date.today() #datetime.now(timezone.utc) #datetime.now()
    start_date = end_date - timedelta(days=30)
    print(start_date)
    print(end_date)
    trading_days = pd.date_range(start=start_date, end=end_date, freq='B')

    soxl_data = get_stock_data_date("SOXL",end_date,timeframe="DAY")#get_historical_data(SOXL, start_date, end_date)
    soxs_data = get_stock_data_date("SOXS",end_date,timeframe="DAY") #get_historical_data(SOXS, start_date, end_date)
    
    #soxl_data['timestamp']= soxl_data['timestamp'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d')) #pd.to_datetime(soxl_data['timestamp'], format='%Y-%m-%d') #datetime.strptime(soxl_data['timestamp'], "%Y-%m-%d")
    #soxs_data['timestamp']= soxs_data['timestamp'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d')) #pd.to_datetime(soxs_data['timestamp'], format='%Y-%m-%d') #datetime.strptime(soxs_data['timestamp'], "%Y-%m-%d")
    soxl_data['timestamp']= pd.to_datetime(soxl_data['timestamp'], format='%Y-%m-%d').dt.date #datetime.strptime(soxl_data['timestamp'], "%Y-%m-%d")
    soxs_data['timestamp']= pd.to_datetime(soxs_data['timestamp'], format='%Y-%m-%d').dt.date #datetime.strptime(soxs_data['timestamp'], "%Y-%m-%d")
    
    print(soxl_data.tail())
    print(soxl_data.iloc[0:3,0])
    print(soxl_data.iloc[0:3,0:5])
    results = pd.DataFrame()

    print(trading_days)
    for date in trading_days:
        total_score = 0
        count = 0
        #print("date")
        #print(date)
        for stock in UNDERLYING_STOCKS:
            try:
                #df = get_stock_data_date(stock,start_date,timeframe="DAY") #get_historical_data(stock, start_date, date + timedelta(days=1))
                start_date1=date
                df, prev = calculate_indicators_hist_date(stock, start_date1,timeframe="DAY")
                #print(df)
                #if date in df.index:
                indicators = df #.loc[date]
                score= score_stock(indicators)
                #print(score)
                weight = STOCK_WEIGHTS.get(stock, 0)
                weighted_score = score * (weight / 100)
                #print(weighted_score)
                total_score +=weighted_score
                #total_weight += weight
                #scores[stock] = weighted_score
                count += 1
            except Exception as e:
                print(e)
                #continue

        #if count == 0:
        #    continue

        etf_score = total_score #/ count
        #print(avg_score)
        recommendation = SOXL if etf_score > SCORING_THRESHOLDS['neutral']+1 else SOXS if etf_score < SCORING_THRESHOLDS['neutral']-1 else None
        #print(recommendation)
            

        # Specify the date you want to match
        target_date = pd.to_datetime(date).date()
        #print(target_date)
        # Filter rows where the date part of the timestamp matches
        #filtered_df = df[df['timestamp'] == target_date]
        # Get a boolean mask where the date matches
        #mask = df['timestamp'].dt.date == target_date
        
        # Get the index of the matching rows
        #matching_indices = df[mask].index
        #indices = df.index[df.index== date].tolist()
        #print(indices)
        try:
            if recommendation == SOXL:
                # Get a boolean mask where the date matches
                mask = soxl_data['timestamp'] == target_date
                
                # Get the index of the matching rows
                matching_indices = soxl_data[mask].index
                
                today_price = soxl_data.iloc[matching_indices,4] #'close']
                next_day_price = soxl_data.shift(-1).iloc[matching_indices,2] #.loc[date]['close']
                daily_return = (next_day_price - today_price) / today_price
            elif recommendation == SOXS:
                mask = soxs_data['timestamp'] == target_date
                #print("mask")
                #print(mask)
                # Get the index of the matching rows
                matching_indices = soxs_data[mask].index
                #print("index")
                #print(matching_indices)
                
                today_price = soxs_data.iloc[matching_indices,4] #.loc[date]['close']
                next_day_price = soxs_data.shift(-1).iloc[matching_indices,2] #.shift(-1).loc[date]['close']
                daily_return = (next_day_price - today_price) / today_price
                
            else:
                daily_return = None
            #print(today_price)
            #print(next_day_price)
            
            #print("Daily return")
            #print(daily_return)
            #results.append({
            #    'Date': date.strftime('%Y-%m-%d'),
            #    'Recommendation': recommendation,
            #    'Actual Return (%)': round(daily_return * 100, 2)
            #})
            
            temp_df = pd.DataFrame({ 'Date': date.strftime('%Y-%m-%d'),
                'Recommendation': recommendation,
                'Actual Return (%)': round(daily_return * 100, 2)})
            results = pd.concat([results, temp_df], ignore_index=True)
        except Exception as e:
            print(e)
            continue
          
 
    #results=pd.DataFrame(results) #.tolist(), index = results.index)
    
    return results

# --- MAIN ---
def run_soxl_soxs_backtest():
    print("Running backtest using Alpaca data...")
    results_df = backtest_strategy()
    print("\n--- SOXL/SOXS Strategy Backtest ---")
    print(results_df)

    cumulative_return = results_df['Actual Return (%)'].sum()
    win_rate = (results_df['Actual Return (%)'] > 0).mean() * 100

    print("\nSummary:")
    print(f"Cumulative Return: {round(cumulative_return, 2)}%")
    print(f"Win Rate: {round(win_rate, 2)}%")

  
def check_sb3_buy_signal(df):
    #df=df.iloc[-1]
    try:
        return (
            25 < df["RSI_14"] < 55 and
            0 < df["fastK"] < 45 and
            0 < df["fastD"] < 35 and
            df["MACD_histogram"] > -0.03 and
            df["ROC_10"] > -0.5 and
            df["close"] > df["5_MA"] and
            df["close"] < df["20_MA"] and
            df["close"] < df["50_MA"] and
            df["close"] < df["R1"] and
            df["close"] > df["S1"]
        )
    except Exception as e:
        print(e)
        return False

#In a declining state (lower lows, weak RSI, under key moving averages)

#On a road to recovery (higher lows, bullish crossovers, increasing momentum)
def evaluate_stock_recovery(symbol):
    #start_date1=date.today()
    end_date = date.today() #datetime.now(timezone.utc) #datetime.now()
    start_date = end_date - timedelta(days=1)
    df=get_stock_data_window(symbol,start_date,end_date,timeframe="Minute",days=200) #(symbol,end_date,timeframe="DAY")
    df=calculate_indicators_hist_fromdf(df)
    df2 = prepare_features(df)
    latest = df.iloc[-1]
    prev_10 = df.iloc[-11:-1]

    declining = (
        (latest['close'] < latest['20_MA']) and
        (latest['close'] < latest['120_MA']) and
        (latest['close'] < latest['50_MA']) and
        (latest['RSI_14'] < 45) and
        (latest['momentum'] < 0) and
        (prev_10['close'].diff().mean() < 0)
    )

    recovering = (
        (latest['close'] > latest['20_MA']) and
        (latest['close'] > latest['120_MA']) and
        (latest['close'] > latest['50_MA']) and
        (latest['RSI_14'] > 50) and
        (latest['momentum'] > 0) and
        (prev_10['close'].diff().mean() > 0)
    )

    if declining:
        return "Declining"
    elif recovering:
        return "Recovering"
    else:
        return "Neutral/Uncertain"


# --- Place a bracket order ---
def place_buy_order(symbol: str, amount_usd: float,buy_limit=0.005,STOP_LOSS=0.1,PROFIT_TARGET=0.05):
    try: 
        latest_price = get_latest_price_alpaca(symbol) #["close"]
        qty = round(amount_usd / latest_price, 0)
        profit_price=max(latest_price * (1 + PROFIT_TARGET), latest_price+0.01)
        
        take_profit_price = round(profit_price, 2)
        
        buy_price=round(latest_price*(1-(buy_limit)),2)
        #if latest_price<0.5:
        stop_loss_price = min(round(buy_price * (1 - STOP_LOSS), 2),round((buy_price-0.01), 2))
       
    
        #response = trade_client.submit_order(order)
        #now = datetime.now(ZoneInfo("America/New_York")).time()
        #print(now)
        now = datetime.now(ZoneInfo("America/New_York")).time()
        market_time=datetime.strptime("09:30", "%H:%M").time()
        if market_time<=now:
            response=place_bracket_limit_order(symbol, qty,buy_price,stop_loss_price,take_profit_price)
        
        elif market_time > now:
            req = LimitOrderRequest(
            symbol = symbol,
            qty = qty,
            limit_price = buy_price,
            side = OrderSide.BUY,
            type = OrderType.LIMIT,
            time_in_force = TimeInForce.DAY,
            extended_hours=True,
            stop_loss=StopLossRequest(stop_price=stop_loss_price),
            take_profit=TakeProfitRequest(limit_price=take_profit_price)
            )
            response = trade_client.submit_order(req)
        #print(response)
        # Extract child orders (stop and limit)
        stop_order_id = None
        limit_order_id = None
        order=response
        if order.legs:
            for leg in order.legs:
                if leg.order_type == 'stop':
                    stop_order_id = leg.id
                elif leg.order_type == 'limit':
                    limit_order_id = leg.id
  
        time.sleep(2)  # wait for fill
        order_info = trade_client.get_order_by_id(response.id)
        now=datetime.now(timezone.utc)
        fill_time = order_info.filled_at or now
        return latest_price, fill_time, response.id,qty,stop_order_id
    except Exception as e:
      print(e)
      
# --- Cancel unfilled orders ---
def cancel_unfilled_orders(symbol: str, order_id: str, order_time: datetime):
    now=datetime.now(timezone.utc)
    if (now - order_time).total_seconds() / 60 >= CANCEL_ORDER_AFTER_MIN:
        order = trade_client.get_order_by_id(order_id)
        if order.filled_at is None:
            trade_client.cancel_order_by_id(order_id)
            print(f"??? Canceled unfilled order for {symbol} after 30 minutes")
            return True
    return False
# --- Close position manually ---
def close_position(symbol: str, entry_price: float,TRAIL_PERCENT):
    current_price = get_latest_price_alpaca(symbol)
    unrealized_loss = (entry_price - current_price) / entry_price

    positions = trade_client.get_all_positions()
    for pos in positions:
        if pos.symbol == symbol:
            qty = pos.qty_available
            if unrealized_loss > 0.01:
                print(f"Loss > 1%. Selling {symbol} at market...")
                order = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.GTC
                )
                trade_client.submit_order(order)
            else:
                print(f"Using trailing stop order for {symbol}...")
                order = TrailingStopOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.SELL,
                    trail_percent=TRAIL_PERCENT,
                    time_in_force=TimeInForce.GTC
                )
                trade_client.submit_order(order)
            break

# --- Update stop-loss if price between entry and target ---
def update_stop_to_current_profit(symbol: str, entry_price: float, order_id: str,PROFIT_TARGET):
    current_price = get_latest_price_alpaca(symbol) #fetch_indicators(symbol)["close"]
    profit_price = entry_price * (1 + PROFIT_TARGET)
    current_profit=100*(current_price-entry_price)/entry_price
    if (entry_price < current_price < profit_price) and current_profit>0.15:
        stop_price = round(current_price, 2)
        new_profit_price = current_price * (1 + PROFIT_TARGET)
        try:
            replace_req = ReplaceOrderRequest(stop_loss={"stop_price": stop_price},take_profit={"limit_price": new_profit_price})
            trade_client.replace_order_by_id(order_id, replace_req)
            print(f"🔧 Updated stop price for {symbol} to {stop_price} and limit price to {new_profit_price}")
        except Exception as e:
            print(f"⚠️ Failed to update stop for {symbol}: {e}")

#8 or higher
def generate_buy_score_v2(df):
    df['buy_score'] = 0
    df['buy_score'] += ((df['close'] < df['l50_MA'])).astype(int)
    df['buy_score'] += ((df['close'] < df['l5_MA'])).astype(int)

    df['buy_score'] += ((df['lmomentum_1'] < 0)).astype(int)
    df['buy_score'] += ((df['lmomentum_3'] < 0)).astype(int)
    df['buy_score'] += ((df['lmomentum_5'] < 0)).astype(int)
    df['buy_score'] += ((df['lROC_5'] < 0)).astype(int)
    df['buy_score'] += ((df['lRSI_14'] < 30)).astype(int)
    df['buy_score'] += ((df['lfastK'] < 20)).astype(int)
    df['buy_score'] += ((df['lvolume_change'] >= 75)).astype(int)
    df['buy_score'] += ((df['lvolume_change_ratio'] > 1)).astype(int)

    df['buy_score'] += ((df['close'] > df['lS1']) & (df['close'] < df['lR1']) ).astype(int)

        
    return df
  
# Fetch historical data and compute indicators
def fetch_and_compute_indicators(symbol,start_date,end_date,timeframe="Day",scorethresh=7):

    df=get_stock_data_window(symbol,start_date,end_date,timeframe=timeframe,days=200) 
    df=calculate_indicators_hist_fromdf_v2(df)
    df=generate_buy_score_v2(df)
    
    # Strategy: Buy if RSI < 30 and FastK < 20 and price > MA50 and Momentum > 0  & df["lvolume_change"]>100
    buy_conditions1 = (
      ((df["lRSI_14"]<30) & (df["lfastK"]<20) & (df["buy_score"]>scorethresh))
    )
    
    #buy_conditions2=((df["lRSI_14"]>50) & (df["lfastK"]>50) & (df["lvolume_change"]>200) & (df["lvolume_change_ratio"]>2))
    buy_conditions2=((df["lRSI_14"]>30) & (df["lRSI_14"]<65) & (df["lfastK"]>25) 
    & (df["lfastK"]>df["lfastK_5"]) & (df["lfastK_5"]<80) 
    & (df["lROC_5"]>0) & (df["lROC_10"]<0) 
    & (df["buy_score"]>0) & (df["buy_score"]<2)
    & (df["lmomentum_1"]>0) & (df["lmomentum_3"]>0)  & (df["lmomentum_5"]>0)
    )
    
    buy_conditions3 = (
      ((df["lRSI_14"]<35) & (df["lfastK"]<20) & (df["lROC_5"]<0) & (df["lROC_10"]<0) & (df["buy_score"]>scorethresh) & (df["lvolume_change_ratio"]>1.25) & (df["lvolume_change"]>10))
    )
    
    df.loc[buy_conditions1, 'signal'] = 1
    #df.loc[buy_conditions2, 'signal'] = 1
    #df.loc[buy_conditions3, 'signal'] = 1
    return df

# --- Main loop ---
def sb3_run_trading_bot(SYMBOLS,num_iter=None,buy_limit=0,signal=None,STOP_LOSS=0.1,PROFIT_TARGET=0.05,TRAIL_PERCENT=0.01,price_increase_thresh=4,timeframe="Day",scorethresh=7,POSITION_SIZE = 1800):
    print("Starting SB3 bot for multiple symbols")
    active_trades = {}
    end_date = date.today() #datetime.now(timezone.utc) #datetime.now()
    start_date = end_date - timedelta(days=60)
    
    ncount=0
    if num_iter is None:
        num_iter=360
    while ncount<num_iter:
        #now = datetime.utcnow().time()
        try:
          
            market_open=False
            now = datetime.now(ZoneInfo("America/New_York")).time()
            current_positions=get_positions()
            if datetime.strptime("07:30", "%H:%M").time() <= now <= datetime.strptime("15:59", "%H:%M").time():
                
                if now>=datetime.strptime("09:30", "%H:%M").time():
                    market_open=True
                  
                for symbol in SYMBOLS:
                    open_orders=get_open_buy_orders(symbol)
                    print(symbol)
                    if symbol not in active_trades and symbol not in current_positions['Symbol'].tolist() and len(open_orders)<1:
                        try: 
                            #signal, df= get_overall_signal(symbol) #get_overall_signal(symbol) #get_trade_signal_daily(symbol,assettype="stocks") #get_signal(latest,prev,indicators)
                            #
                            if signal is None:
                                df=fetch_and_compute_indicators(symbol,date.today(),end_date,timeframe,scorethresh)
                                signal=df.iloc[-1]['signal']
                                if signal is not None and signal==1:
                                    signal="BUY"
                            #signal=signal.upper()
                            #p1=pred_gains(symbol,date.today())
                            #cols_to_include = list(df.columns[[0, 1,2, 3,4,5]])
                            #print(df.loc[:,cols_to_include])
                            if signal=="BUY" or signal=="BUY:1" or signal=="BUY:0": #check_sb3_buy_signal(indicators):
                                try:
                                    print(f"Buy signal for {symbol}! Placing buy order...")
                                    entry_price, entry_time, order_id,qty,stop_order_id = place_buy_order(symbol, POSITION_SIZE,buy_limit,STOP_LOSS,PROFIT_TARGET)
                                    active_trades[symbol] = (entry_price, entry_time, order_id,qty,stop_order_id)
                                except Exception as e:
                                  print(e)
                        except Exception as e:
                            print(e)
                    elif symbol in active_trades and symbol in current_positions['Symbol'].tolist(): # and len(open_orders)>0
                        entry_price, entry_time, order_id,qty,stop_order_id = active_trades[symbol]
                        current_price=round(get_latest_price_alpaca(symbol),2)
                        #naive = dt.replace(tzinfo=None)
                        now = datetime.now(timezone.utc)
                        minutes_elapsed = (now - entry_time).total_seconds() / 60
                        #update_stop_to_current_profit(symbol, entry_price, stop_order_id)
                        pct_diff=100*(current_price-entry_price)/entry_price
                        profit_price=round(max(current_price * (1 + PROFIT_TARGET), current_price+0.01),2)
                        
                        #if profit more than 4%
                        if current_price>entry_price and pct_diff>(TRAIL_PERCENT*price_increase_thresh):
                            print(f"{symbol} price increase. Updating sell order...")
                            #update_stop_to_current_profit(symbol, entry_price)
                            check_open_status=get_open_sell_orders(symbol)
                            if len(check_open_status)>0 and market_open:
                                new_stop_loss=round(current_price*(1-(TRAIL_PERCENT)),2)
                                #new_stop_loss=min(round(current_price*(1-TRAIL_PERCENT*0.25),2),round(current_price-0.02,2))
                                #new_stop_loss=round(entry_price*(1-(TRAIL_PERCENT)),2)
                                #new_stop_loss=round(current_price,2)
                                try:
                                    active_trades[symbol]=(current_price, entry_time, order_id,qty,stop_order_id)
                                except Exception as e:
                                    print(e)
                                cancel_all_orders(symbol)
                                try:
                                  place_limit_sell_order(symbol,profit_price,new_stop_loss,qty)
                                except Exception as e:
                                  print(e)
                                  next
                       
                        if cancel_unfilled_orders(symbol, order_id, entry_time):
                            del active_trades[symbol]
                        elif minutes_elapsed >= MAX_HOLD_TIME_MIN:
                            print(f"{symbol} time limit reached. Closing manually...")
                            close_position(symbol, entry_price,TRAIL_PERCENT)
                            del active_trades[symbol]
                    elif symbol in active_trades and symbol not in current_positions['Symbol'].tolist():
                        del active_trades[symbol]
                    elif symbol in current_positions['Symbol'].tolist():
                        # Filter row 
                        symbol_row = current_positions[current_positions['Symbol'] == symbol]
                        entry_price=round(float(symbol_row['Avg. Entry Price']),2)
                        current_price=round(float(symbol_row['Current Price']),2)
                        qty=float(symbol_row['Qty'])
                        pct_diff=100*(current_price-entry_price)/entry_price
                        profit_price=round(max(current_price * (1 + PROFIT_TARGET), current_price+0.01),2)
                        if current_price>entry_price and pct_diff>(TRAIL_PERCENT*price_increase_thresh):
                            print(f"{symbol} price increase. Updating sell order...")
                            #update_stop_to_current_profit(symbol, entry_price)
                            if market_open:
                                new_stop_loss=round(current_price*(1-TRAIL_PERCENT),2)
                                #new_stop_loss=round(entry_price*(1-(STOP_LOSS-TRAIL_PERCENT)),2)
                                #new_stop_loss=round(current_price,2)
                                #new_stop_loss=round(current_price*(1-TRAIL_PERCENT*0.25),2)
                                #proposed_stop=round(current_price*(1-TRAIL_PERCENT),2)
                                # Only update stop if it's above the current price
                                #new_stop_loss = proposed_stop if proposed_stop > entry_price else (current_price-0.01)
                                try:
                                    now = datetime.now(timezone.utc)
                                    order_id=None
                                    stop_order_id=None
                                    active_trades[symbol]=(current_price, now, order_id,qty,stop_order_id)
                                except Exception as e:
                                    print(e)
                                cancel_all_orders(symbol)
                                try:
                                    place_limit_sell_order(symbol,profit_price,new_stop_loss,qty)
                                except Exception as e:
                                  
                                  next
                       
            else:
                print("Outside trading window.")
        except Exception as e:
            next
        ncount=ncount+1    
        time.sleep(5)

# Step 3: Define the function to calculate technical indicators
def calculate_indicators_pred(df):
    # VWAP calculation
    df['VWAP'] = (df['volume'] * df['close']).cumsum() / df['volume'].cumsum()

    # 5-MA and 20-MA
    df['5_MA'] = df['close'].rolling(window=5).mean()
    df['20_MA'] = df['close'].rolling(window=20).mean()

    # RSI calculation (14-period)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # ROC calculation (12-period)
    df['ROC'] = df['close'].pct_change(periods=12) * 100
    
    # ROC volume calculation (5-period; 5 minutes)
    df['ROC_volume'] = df['volume'].pct_change(periods=5) * 100

    # ROC volume calculation (5-period; 5 minutes)
    df['ROC_tradecount'] = df['trade_count'].pct_change(periods=5) * 100
    

    # Stochastic Oscillator FastK and FastD (14-period)
    df['stochastic_fastK'] = ((df['close'] - df['low'].rolling(window=14).min()) /
                               (df['high'].rolling(window=14).max() - df['low'].rolling(window=14).min())) * 100
    df['stochastic_fastD'] = df['stochastic_fastK'].rolling(window=3).mean()

    # MACD calculation
    df['MACD'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands calculation (20-period)
    df['20_STD'] = df['close'].rolling(window=20).std()
    df['upper_band'] = df['20_MA'] + (2 * df['20_STD'])
    df['lower_band'] = df['20_MA'] - (2 * df['20_STD'])

    # ATR calculation (14-period)
    df['TR'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))))
    df['ATR'] = df['TR'].rolling(window=14).mean()

    # Pivot Points and Support/Resistance Levels calculation
    df['pivot_point'] = (df['high'] + df['low'] + df['close']) / 3
    df['S1'] = 2 * df['pivot_point'] - df['high']
    df['S2'] = df['pivot_point'] - (df['high'] - df['low'])
    df['R1'] = 2 * df['pivot_point'] - df['low']
    df['R2'] = df['pivot_point'] + (df['high'] - df['low'])

    return df

def data_prep(symbol="VVPR",targetdate="2025-03-21"):
    
    # Step 1: Load the stock data
    df_mlgo = get_stock_data_date(symbol,targetdate) #pd.read_csv(fname)
    df_mlgo['timestamp']=df_mlgo.index
    df_mlgo['Symbol']=symbol
    # Reset index to make the row names a regular column (optional)
    df_mlgo = df_mlgo.reset_index(drop=True)
    df_mlgo['timestamp'] = pd.to_datetime(df_mlgo['timestamp'])
    df_mlgo = df_mlgo.set_index('timestamp')
    
    # Step 2: Filter data between 4 AM to 11 AM
    df_mlgo= df_mlgo.between_time('04:00:00', '14:00:00')
    
    df_filtered=df_mlgo.dropna()
    
    df_filtered['timestamp']=df_filtered.index
    # Step 4: Recalculate technical indicators for the filtered data (4 AM to 7 AM)
    df_filtered = calculate_indicators_pred(df_filtered)

    # Step 5: Define the target variable: substantial gain (5% or more) in the next 30 minutes
    #df_filtered['future_close'] = df_filtered['close'].shift(-30)  # Close price 60 minutes ahead
    #df_filtered['substantial_gain'] = (df_filtered['future_close'] >= df_filtered['close'] * 1.05).astype(int)

   

    df_filtered=df_filtered.dropna()
    return df_filtered

def data_prep_fromdf(df):
    
    df_mlgo=df
    # Reset index to make the row names a regular column (optional)
    #df_mlgo = df_mlgo.reset_index(drop=True)
    #df_mlgo['timestamp'] = pd.to_datetime(df_mlgo['timestamp'])
    df_mlgo = df_mlgo.set_index('timestamp')
    
    # Step 2: Filter data between 4 AM to 11 AM
    #df_mlgo= df_mlgo.between_time('04:00:00', '14:00:00')
    
    df_filtered=df_mlgo.dropna()
    
    df_filtered['timestamp']=df_filtered.index
    # Step 4: Recalculate technical indicators for the filtered data (4 AM to 7 AM)
    df_filtered = calculate_indicators_pred(df_filtered)

    # Step 5: Define the target variable: substantial gain (5% or more) in the next 30 minutes
    #df_filtered['future_close'] = df_filtered['close'].shift(-30)  # Close price 60 minutes ahead
    #df_filtered['substantial_gain'] = (df_filtered['future_close'] >= df_filtered['close'] * 1.05).astype(int)

   

    df_filtered=df_filtered.dropna()
    return df_filtered
  
def pred_gains_fromdf(df):
    test1=data_prep_fromdf(df)
    test1=test1.dropna()
    if test1.shape[0]>10:
          
        # Step 6: Prepare the feature matrix and target vector
        features = ['close', 'volume', 'trade_count', 'VWAP', '5_MA', '20_MA', 'RSI', 'ROC', 'ROC_volume','ROC_tradecount', 'stochastic_fastK', 
                'stochastic_fastD', 'MACD', 'MACD_signal', 'upper_band', 'lower_band', 'ATR', 'pivot_point', 'S1', 'S2', 'R1', 'R2']
        target = 'substantial_gain'
        X_test = test1[features]
        loaded_rf = joblib.load("/Users/karanuppal/Downloads/rf_2025-04-07.joblib")
        # Step 10: Make predictions on the test set
        y_pred = loaded_rf.predict(X_test)
        test1['gain_pred']=y_pred
        pred_val=test1.iloc[test1.shape[0]-1]["gain_pred"]
    else:
        pred_val=None
    
    return(pred_val)

def pred_gains_fromdfv2(test1):
    #test1=data_prep_fromdf(df)
    #test1=test1.dropna()
    test1=test1.to_frame().transpose()
    #print(test1)
    if test1.shape[0]>0:
        # Step 6: Prepare the feature matrix and target vector
        features = ['open','high','low','close','volume','trade_count', 'vwap','sma_5','sma_10', 'sma_20', 'sma_50', 'sma_120', 'rsi','rsi_5','fastk','fastd', 'fastk_5', 'roc', 'roc_5', 'roc_prev', 'momentum', 'momentum_120', '20_STD',
       'upper_band', 'lower_band', 'bb_width','volume_change','avg_vol_10','volume_change_ratio', 'bb_width_5','recovery_score', 'momentum_score', 'score', 'avoid_score','decline_score', 'buy_score','buy_signal']
        
        target = 'target'
        X_test = test1[features]
        loaded_rf = joblib.load("/Users/karanuppal/Downloads/rf_daily_scores_2025-04-15.joblib")
        # Step 10: Make predictions on the test set
        y_pred = loaded_rf.predict(X_test)
        #print(y_pred)
        test1['Pred_Gain']=y_pred
        pred_val=test1.iloc[test1.shape[0]-1]["Pred_Gain"]
    else:
        pred_val=None
    
    return(pred_val)


def pred_gains(symbol,targetdate=date.today()):
    test1=data_prep(symbol,targetdate)
    test1=test1.dropna()
    # Step 6: Prepare the feature matrix and target vector
    features = ['close', 'volume', 'trade_count', 'VWAP', '5_MA', '20_MA', 'RSI', 'ROC','ROC_volume','ROC_tradecount',  'stochastic_fastK', 
            'stochastic_fastD', 'MACD', 'MACD_signal', 'upper_band', 'lower_band', 'ATR', 'pivot_point', 'S1', 'S2', 'R1', 'R2']
    target = 'substantial_gain'
    X_test = test1[features]
    loaded_rf = joblib.load("/Users/karanuppal/Downloads/rf_2025-04-07.joblib")
    #loaded_rf = joblib.load("C:/Users/karan/OneDrive/Documents/Productivity/mystockforecast/scripts/Dev/rf_2025-04-02.joblib")
    # Step 10: Make predictions on the test set
    y_pred = loaded_rf.predict(X_test)
    test1['gain_pred']=y_pred
    return(test1.iloc[test1.shape[0]-1]["gain_pred"])
  
def get_portfolio_history():
    # Define the request parameters
    #request_params = GetPortfolioHistoryRequest()
    #request_params = PortfolioHistory()
    # Define the request parameters
    request_params = GetPortfolioHistoryRequest(
        period='1M',  # e.g., '1D', '1W', '1M', '3M', '6M', '1Y', 'all'
        timeframe='1D'  # e.g., 'minute', 'hour', 'day'
    )
    
    # Get the portfolio history data
    portfolio_history = trade_client.get_portfolio_history(request_params)
    
    # Convert the portfolio history to a DataFrame
    history_data = pd.DataFrame(portfolio_history)
    #history_data['timestamp'] = pd.to_datetime(portfolio_history['timestamp'], unit='s')
    #history_data.set_index('timestamp', inplace=True)
    
    return history_data
    # Get the portfolio history data
    #portfolio_history = trade_client.get_portfolio_history(request_params)
    
    # Convert the portfolio history to a DataFrame
    #history_data = pd.DataFrame(portfolio_history) #[ph.dict() for ph in portfolio_history])
    
    #return history_data

def get_account_info():
    res=trade_client.get_account()
    
    return res
# Function to create sell orders 5% above the bought price
def create_sell_orders(qsymbol=None,qsymbol2=None,sell_pct_thresh=1,stoplimit_pct_thresh=1,trail_pct=0.1,ordertype="normal",assettype="stocks"):
    #api = ScreenerClient(api_key=api_key, secret_key=secret_key) #, paper=paper, url_override=trade_api_url)
    positions = trade_client.get_all_positions()
    for position in positions:
        symbol = position.symbol
        print(symbol)
        if qsymbol is None:
          qty = float(position.qty)
          avg_buy_price = float(position.avg_entry_price)
          sell_price = avg_buy_price * 1.01
          
          # Check if there's already a sell order for this position
          
          existing_orders = None # trade_client.list_orders(status='open', symbols=[symbol])
          #position = trade_client.get_open_position(symbol_or_asset_id=symbol)
          #qty_open=int(pd.DataFrame(position).iloc[6][1])
          #entry_price=float(pd.DataFrame(position).iloc[5][1])
          #positions= trade_client.get_open_position(symbol_or_asset_id=symbol)
          if not existing_orders:
              print(f"Creating sell order for {symbol}: {qty} shares at ${sell_price}")
              trade_client.submit_order(
                  symbol=qsymbol2,
                  qty=qty,
                  side='sell',
                  type='limit',
                  time_in_force='gtc',
                  limit_price=sell_price
              )
        else:
          if qsymbol==symbol:
            
            qty = float(position.qty)
            avg_buy_price = float(position.avg_entry_price)
            sell_price = avg_buy_price * 1.01
            
            # Check if there's already a sell order for this position
            
            existing_orders = None #trade_client.list_orders(status='open', symbols=[symbol])
            
            #positions= trade_client.get_open_position(symbol_or_asset_id=symbol)
            if not existing_orders:
                print(f"Creating sell order for {qsymbol2}: {qty} shares at ${sell_price}")
                trade_calls_sell(symbol=qsymbol2,qty_to_sell=qty,sell_pct_thresh=sell_pct_thresh,trail_pct=trail_pct,stoplimit_pct_thresh=stoplimit_pct_thresh,ordertype="normal",
                assettype=assettype,CHECK_INTERVAL=10,entry_price=avg_buy_price)
                
        #monitor_and_adjust_sell_order(symbol,buy_price=

# Function to track newly bought positions and create sell orders
def track_new_positions():
    while True:
        try:
            create_sell_orders()
        except Exception as e:
            print(f"Error: {e}")
        
        # Check for new positions every minute
        time.sleep(60)

#Start tracking new positions
#track_new_positions()

def get_marketmovers():
    api = ScreenerClient(api_key=api_key, secret_key=secret_key) #, paper=paper, url_override=trade_api_url)
    
    req=MarketMoversRequest(market_type="stocks",top=50)
   
    market_movers=api.get_market_movers(req)
    # Fetch premarket gainers
    #premarket_gainers = api.get_screener(market_type='stocks', movers='gainers')

    # Convert the result to a DataFrame
    df2 = pd.DataFrame(market_movers)

    df7 = (df2.transpose().iloc[1,0])

    # Create an empty DataFrame
    df9 = pd.DataFrame() #columns=['col1', 'col2', 'col3'])
    #print(df8)
    # Iterate through the list and append each sublist as a row
    for i in range(0,len(df7)):
        temp1=pd.DataFrame(df7[i]).transpose()
        temp2=(pd.DataFrame(temp1.iloc[1,:]).transpose())
        #print(temp1.iloc[1,:].transpose())
        #print(temp2)
        #df8.concat(temp2,ignore_index=True) # iloc[i,:] = temp2
        df9 = pd.concat([df9, pd.DataFrame(temp2)], ignore_index=True)

    df9.columns=['symbol','percent_change','change','price']
    return(df9)

def get_premarket_benzinga():
    

    # Define the URL
    url = "https://www.benzinga.com/premarket"

    # Make a GET request to fetch the data
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the table containing the stock data
        table = soup.find('table')

        # Extract the headers
        headers = [header.text.strip() for header in table.find_all('th')]

        # Extract the rows from the table
        rows = table.find_all('tr')[1:]  # Skip the header row

        # Loop through each row to extract data
        table_data = []
        for row in rows:
            columns = row.find_all('td')
            #row_data = [column.text.strip() for column in columns]
            row_data = [column.text.strip().replace('$|%', '') for column in columns]
       
            table_data.append(row_data)

        # Print the headers
        #print(headers)
        df=pd.DataFrame(table_data).iloc[:,[0,2,3]]
        
        return(df)
    else:
        print(f"Failed to retrieve data. HTTP Status code: {response.status_code}")


def check_and_place_orders(symbol,last_price,previous_close,amount_to_invest=500,price_drop_threshold=5):
    while i<7:
        #current_price = get_current_price(symbol)
        #historical_data = api.get_barset(symbol, 'day', limit=1).df
        #previous_close = float(historical_data[symbol]['close'][-1])

        # Check if the price has dropped by 5% or more
        if current_price <= previous_close * (1 - price_drop_threshold):
            # Calculate the number of shares to buy
            quantity = amount_to_invest // last_price  # Example quantity to buy
            #place_bracket_order(symbol, qty, current_price)
            #trade_calls_buy
        time.sleep(3600)
        
def cancel_buy_order(symbol):
    req = GetOrdersRequest(
        status = QueryOrderStatus.OPEN,
        symbols = [symbol]
        
    )

    open_orders = trade_client.get_orders(req)
    for order in open_orders:
        if order.side=="buy":
            trade_client.cancel_order_by_id(order.id)
            print(f"Cancelled buy order: {order.id} for {symbol}")

def cancel_all_orders(symbol):
    req = GetOrdersRequest(
        status = QueryOrderStatus.OPEN,
        symbols = [symbol]
        
    )

    open_orders = trade_client.get_orders(req)
    for order in open_orders:
        trade_client.cancel_order_by_id(order.id)
        print(f"Cancelled all orders: {order.id} for {symbol}")


def trade_calls_buy(symbol,last_price,quantity=1,amount_to_invest=500,pct_thresh_buy=5,pct_thresh_sell=5,pct_thresh_stoploss=3,ordertype="bracket"):
    trade_client = TradingClient(api_key=api_key, secret_key=secret_key, paper=paper, url_override=trade_api_url)
    req = GetOrdersRequest(
        status = QueryOrderStatus.OPEN,
        symbols = [symbol]
        
    )
    try: 
        # Calculate the number of shares to buy
        #quantity = amount_to_invest // last_price

        open_orders = trade_client.get_orders(req)
        print(len(open_orders))
        req = GetOrdersRequest(
            status = QueryOrderStatus.CLOSED,
            symbols = [symbol]
        )
        open_orders2 = trade_client.get_orders(req)
        print(len(open_orders2))

        if len(open_orders2)>0:
            try:
                position = trade_client.get_open_position(symbol_or_asset_id=symbol)
                qty_open=int(pd.DataFrame(position).iloc[6][1])
                entry_price=float(pd.DataFrame(position).iloc[5][1])
            except:
                qty_open=0
        else:
            qty_open=0

        #last_price=get_latest_price(symbol)
        val_buy=round(last_price*(1-(1*pct_thresh_buy*0.01)),2)
        val_sell=round(last_price*(1+(1*pct_thresh_sell*0.01)),2)
        val_stoploss=round(val_buy*(1-(pct_thresh_stoploss*0.01)),2)
        open_orders=[]
        if len(open_orders)<1:

            if len(open_orders)<1:
                # simple, limit order, fractional qty

                if ordertype=="bracket":
                    req = LimitOrderRequest(
                        symbol = symbol,
                        qty = quantity,
                        limit_price = val_buy,
                        side = OrderSide.BUY,
                        type = OrderType.LIMIT,
                        time_in_force = TimeInForce.GTC,
                        extended_hours=False,
                        order_class=OrderClass.BRACKET,
                       # stop_loss=StopLossRequest(stop_price=val_stoploss),
                        #trail_percent=1,
                        stop_loss=StopLossRequest(stop_price=val_stoploss),
                        take_profit=TakeProfitRequest(limit_price=val_sell)
                    )
                elif ordertype=="limit.extended":
                     req = LimitOrderRequest(
                        symbol = symbol,
                        qty = quantity,
                        limit_price = val_buy,
                        side = OrderSide.BUY,
                        type = OrderType.LIMIT,
                        time_in_force = TimeInForce.DAY,
                        extended_hours=True,
                        stop_loss=StopLossRequest(stop_price=val_stoploss),
                        take_profit=TakeProfitRequest(limit_price=val_sell)
                    )
                elif ordertype=="limit":
                     req = LimitOrderRequest(
                        symbol = symbol,
                        qty = quantity,
                        limit_price = val_buy,
                        side = OrderSide.BUY,
                        type = OrderType.LIMIT,
                        time_in_force = TimeInForce.GTC,
                        extended_hours=False,
                        stop_loss=StopLossRequest(stop_price=val_stoploss),
                        take_profit=TakeProfitRequest(limit_price=val_sell)
                    )
                res = trade_client.submit_order(req)
                return res.id
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def get_buy_price(order_id):
    return float(trade_client.get_order_by_id(order_id).filled_avg_price) 

def get_buy_qty(order_id):
    return float(trade_client.get_order_by_id(order_id).filled_qty) 
 
def replace_with_trailing_stop(symbol, qty, buy_price, stop_limit_order_id):
    """
    Monitors price movement and replaces the stop-limit order with a trailing stop after a 3% profit.
    """
    while True:
        # Fetch the latest market price
        latest_price = float(trading_client.get_last_trade(symbol).price)

        # Check if price increased by 3%
        if latest_price >= buy_price * 1.03:
            # Cancel the stop-limit order
            trading_client.cancel_order(stop_limit_order_id)
            print("Canceled Stop-Limit Order.")

            # Place a trailing stop order
            trail_percent = 3.0  # 3% trailing stop
            order = TrailingStopOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.SELL,
                trail_percent=trail_percent,
                time_in_force=TimeInForce.GTC
            )

            response = trading_client.submit_order(order)
            print(f"Trailing Stop Order Placed: {response.id}")

            break  # Exit monitoring loop after conversion

        time.sleep(5)  # Check price every 5 seconds
        
def replace_order_stoploss(orderid,stop_price,qty):
    trade_client = TradingClient(api_key=api_key, secret_key=secret_key, paper=paper, url_override=trade_api_url)
    order=trade_client.replace_order_by_id(
            order_id=orderid,order_data=ReplaceOrderRequest(qty=qty,client_order_id=orderid,time_in_force=TimeInForce.GTC,
                            stop_price = stop_price))
    #res = trade_client.submit_order(req)

def replace_order_limit(orderid,limit_price,qty):
    trade_client = TradingClient(api_key=api_key, secret_key=secret_key, paper=paper, url_override=trade_api_url)
    
    order=trade_client.replace_order_by_id(
            order_id=orderid,order_data=ReplaceOrderRequest(qty=qty,client_order_id=orderid,time_in_force=TimeInForce.GTC,
                            limit_price = limit_price))

                            
def replace_with_trailing_stop_order(order_id, symbol,qty,val_stoploss,val_trailpct):
    """Replaces an existing order with a trailing stop sell order."""
    trailing_stop_order_data = TrailingStopOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.SELL,
        trail_percent=val_trailpct,
        time_in_force=TimeInForce.GTC,
        stop_loss=StopLossRequest(stop_price=val_stoploss)
        
    )
    trading_client.replace_order_by_id(order_id=order_id, order_data=trailing_stop_order_data)
    print(f"Replaced with trailing stop sell order with {TRAIL_PERCENT}% trail.")

def get_activity_data():
    request_params = GetActivitiesRequest()
   # trade_client = TradingClient(api_key=api_key, secret_key=secret_key, paper=paper, url_override=trade_api_url)
    activities = trading_client.get_activities(request_params)

    # Convert the activities to a DataFrame
    activity_data = pd.DataFrame([activity.dict() for activity in activities])

    return activity_data
# Function to fetch latest news and perform sentiment analysis
def get_news(symbol="ACON"):
    current_time = pd.Timestamp.now(tz="America/New_York")
    time_20_minutes_previous = current_time - pd.Timedelta(minutes=720)

    news = news_client.get_news(NewsRequest(
        symbols=symbol,
        start=time_20_minutes_previous,
        end=current_time,
        limit=50)
    )

    
    print(news)
    news_list = []
    page_token = None

    return news.df

def get_sentiment(news):
    sentiments = []
    for article in news:
        print(article)
        for a in article:
            print(a)
            analysis = TextBlob(a["summary"])
            polarity = analysis.sentiment.polarity
            #example:
            #sentiment_results=sia.polarity_scores(article.summary)

            sentiments.append(polarity)

    return np.mean(sentiments) if sentiments else 0  # Return average sentiment score

def get_last_closed_buy_order(symbol):
    """Checks for any existing open buy or sell orders."""
    request_params = GetOrdersRequest(
        status=QueryOrderStatus.CLOSED,
        side=OrderSide.BUY,
        symbols=[symbol]
    )
    open_orders = trade_client.get_orders(filter=request_params)

    df = pd.DataFrame()
    i=0
    for o in open_orders:
        # convert dot notation to dict
        d = vars(o)
        # import dict into dataframe
        dft = pd.DataFrame.from_dict(d, orient='index')
        print(dft)
        # append to dataframe
        #pd.concat([df,dft]) #, ignore_index=True)
        df[i]=dft
        i=i+1

    return df.transpose().iloc[0:1]


def get_closed_buy_orders(symbol=None):
    if symbol is None:
        """Checks for any existing open buy or sell orders."""
        request_params = GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
            side=OrderSide.BUY
            
        )
    else:
        """Checks for any existing open buy or sell orders."""
        request_params = GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
            side=OrderSide.BUY,
            symbols=[symbol]
        )
    open_orders = trade_client.get_orders(filter=request_params)

    df = pd.DataFrame()
    i=0
    for o in open_orders:
        # convert dot notation to dict
        d = vars(o)
        # import dict into dataframe
        dft = pd.DataFrame.from_dict(d, orient='index')
        #print(dft)
        # append to dataframe
        #pd.concat([df,dft]) #, ignore_index=True)
        df[i]=dft
        i=i+1

    return df.transpose()
def get_all_positions():
    
    
    # Get all open positions
    positions = trade_client.get_all_positions()
    symbol_list=[]
    price_list=[]
    qty_list=[]
    # Print positions
    for position in positions:
        print(f"Symbol: {position.symbol}")
        print(f"Quantity: {position.qty}")
        print(f"Average Entry Price: {position.avg_entry_price}")
        print(f"Current Price: {position.current_price}")
        print(f"Unrealized P/L: {position.unrealized_pl}")
        print("-" * 30)
        symbol_list.append(position.symbol)
        price_list.append(position.avg_entry_price)
        qty_list.append(position.qty)
    return symbol_list,price_list,qty_list
def get_open_sell_orders(symbol=None):
    
    if symbol is None:
        """Checks for any existing open buy or sell orders."""
        request_params = GetOrdersRequest(
            status=QueryOrderStatus.OPEN,
            side=OrderSide.SELL
            
        )
    else:
        """Checks for any existing open buy or sell orders."""
        request_params = GetOrdersRequest(
            status=QueryOrderStatus.OPEN,
            side=OrderSide.SELL,
            symbols=[symbol]
        )
    open_orders = trade_client.get_orders(filter=request_params)

    df = pd.DataFrame()
    i=0
    for o in open_orders:
        # convert dot notation to dict
        d = vars(o)
        # import dict into dataframe
        dft = pd.DataFrame.from_dict(d, orient='index')
        print(dft)
        # append to dataframe
        #pd.concat([df,dft]) #, ignore_index=True)
        df[i]=dft
        i=i+1

    return df.transpose()


def get_open_buy_orders(symbol):
    """Checks for any existing open buy or sell orders."""
    request_params = GetOrdersRequest(
        status=QueryOrderStatus.OPEN,
        symbols=[symbol]
    )
    open_orders = trade_client.get_orders(filter=request_params)
    return open_orders

def get_existing_orders(symbol):
    """Checks for any existing open buy or sell orders."""
    request_params = GetOrdersRequest(
        status=QueryOrderStatus.OPEN,
        symbols=[symbol]
    )
    open_orders = trade_client.get_orders(filter=request_params)
    #print(open_orders)
    return open_orders

def get_existing_open_orders(symbol):
    """Checks for any existing open buy or sell orders."""
    request_params = GetOrdersRequest(
        status=QueryOrderStatus.OPEN,
        symbols=[symbol]
    )
    open_orders = trade_client.get_orders(filter=request_params)
    #print(open_orders)
    return open_orders

def get_positions(symbol):
    """Checks for any existing open buy or sell orders."""
    try:
        position = trade_client.get_open_position(symbol_or_asset_id=symbol)
        qty_open= int(pd.DataFrame(position).iloc[6][1])
        entry_price=float(pd.DataFrame(position).iloc[5][1])
        df = pd.DataFrame(position) #.loc[symbol].reset_index()
        df.columns = df.iloc[0]
        df = df[1:]

        # Reset index
        df = df.reset_index(drop=True)
    except Exception as e:
        df=None
    return df

def place_bracket_limit_order(symbol, quantity,buy_price,stop_price,take_profit_price):
    req = LimitOrderRequest(
                        symbol = symbol,
                        qty = quantity,
                        limit_price = buy_price,
                        side = OrderSide.BUY,
                        type = OrderType.LIMIT,
                        time_in_force = TimeInForce.GTC,
                        extended_hours=False,
                        order_class=OrderClass.BRACKET,
      
                        stop_loss=StopLossRequest(stop_price=stop_price),
                        take_profit=TakeProfitRequest(limit_price=take_profit_price)
                    )
    response=trade_client.submit_order(order_data=req)
    print(f"Limit Bracket order placed: buy @ ${buy_price}, stop @ ${stop_price}, take-profit @ ${take_profit_price}")
    return response
                    
def place_bracket_market_order(symbol, quantity,stop_price,take_profit_price):
    bracket_order = MarketOrderRequest(
            symbol=symbol,
            qty=quantity,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
            order_class=OrderClass.BRACKET,
            stop_loss={'stop_price': stop_price},
            take_profit={'limit_price': take_profit_price}
        )
    response=trade_client.submit_order(order_data=bracket_order)
    print(f"Limit Bracket order placed: buy @ ${buy_price}, stop @ ${stop_price}, take-profit @ ${take_profit_price}")
    return response
    
def place_trailing_stop_order(symbol, quantity,trail_pct,loss_price,profit_price,ordertype="limit"):
  
    if ordertype=="limit":
        """Places a market sell order."""
        req = TrailingStopOrderRequest(
                        symbol = symbol,
                        qty = quantity,
                        side = OrderSide.SELL,
                        time_in_force = TimeInForce.GTC,
                        trail_percent=trail_pct,
                        stop_price = loss_price,
                        limit_price=profit_price
                        )
    else:
        if ordertype=="limit.extended":
            req = TrailingStopOrderRequest(
                            symbol = symbol,
                            qty = quantity,
                            side = OrderSide.SELL,
                            
                            trail_percent=trail_pct,
                            stop_price = loss_price,
                            limit_price=profit_price,
                            time_in_force = TimeInForce.DAY,
                            extended_hours=True
                            )
    res = trade_client.submit_order(req)
    #sell_order = trade_client.submit_order(order_data=sell_order_data)
    print(f"Trailing sell order placed for {quantity} shares of {symbol}.")
    return res
    
def place_limit_sell_order(symbol, profit_price, loss_price,quantity):
    """Places a limit sell order."""
    sell_order_data = LimitOrderRequest(
        order_class = OrderClass.OCO,
        symbol=symbol,
        qty=quantity,
        side=OrderSide.SELL,
        limit_price=profit_price,
        time_in_force=TimeInForce.GTC,
        stop_loss = StopLossRequest(stop_price=loss_price),
        take_profit=TakeProfitRequest(limit_price=profit_price)
    )
    sell_order = trade_client.submit_order(order_data=sell_order_data)
    print(f"Limit sell order placed at ${profit_price} and stop loss of ${loss_price} for {quantity} shares of {symbol}.")
    return sell_order

def place_limit_buy_order(symbol, limit_price,quantity,ordertype="limit"):
    """Places a limit buy order."""
    buy_order_data = LimitOrderRequest(
        #order_class = OrderClass.OCO,
        symbol=symbol,
        qty=quantity,
        side=OrderSide.BUY,
        limit_price=limit_price,
        time_in_force=TimeInForce.GTC
        #stop_loss = StopLossRequest(stop_price=loss_price),
        #take_profit=TakeProfitRequest(limit_price=profit_price)
    )
    buy_order = trade_client.submit_order(order_data=buy_order_data)
    print(f"Limit buy order placed at ${limit_price} for {quantity} shares of {symbol}.")
    return buy_order
          

def place_market_sell_order(symbol, quantity):
    """Places a market sell order."""
    sell_order_data = MarketOrderRequest(
        symbol=symbol,
        qty=quantity,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.GTC
    )
    sell_order = trade_client.submit_order(order_data=sell_order_data)
    print(f"Market sell order placed for {quantity} shares of {symbol}.")
    return sell_order
  
def trade_calls_sell(symbol,qty_to_sell=1,sell_pct_thresh=5,trail_pct=1,stoplimit_pct_thresh=1,
ordertype="limit",assettype="stocks",CHECK_INTERVAL=10,entry_price=0):
    pct_thresh=sell_pct_thresh
    
    req = GetOrdersRequest(
        status = QueryOrderStatus.OPEN,
        symbols = [symbol],
        side=OrderSide.BUY
    )
    open_orders = trade_client.get_orders(req)
      
    #mean_price=get_latest_price(symbol)
   
    
    # Monitor buy order for execution
    while True:
        existing_orders = get_existing_orders(symbol)
        print(existing_orders)
        if not existing_orders:
            print("Buy order filled.")
            position = 1
            break
        print("Waiting for open order to fill...")
        time.sleep(CHECK_INTERVAL)
    
    len(open_orders)
    req = GetOrdersRequest(
        status = QueryOrderStatus.CLOSED,
        
        symbols = [symbol]
    )
    open_orders2 = trade_client.get_orders(req)
    len(open_orders2)

    qty_open=qty_to_sell
    #if qty_to_sell is not None:
        
    res=None     
        
    val_buy=entry_price #round(entry_price*(1-(1*pct_thresh*0.01)),2)
    val_sell=round(entry_price*(1+(1*pct_thresh*0.01)),2)
    
    if assettype=="stocks":
      current_price=get_latest_price_alpaca(symbol)
    else:
      current_price=get_latest_cryptoprice_alpaca(symbol)
    stop_sell1=round(current_price*(1+(1*stoplimit_pct_thresh*0.01)),2)
    stop_sell2=round(current_price*(1-(1*stoplimit_pct_thresh*0.01)),2)
    
    price_increase_percent = ((current_price - val_buy) / val_buy) * 100
    # Place corresponding sell order
    #sell_price = get_last30min_high(SYMBOL) #round(buy_price * (1 + SELL_PREMIUM_PERCENT / 100), 2)
    #place_limit_sell_order(SYMBOL, sell_price, quantity)

    if len(open_orders2)>0 and qty_open>0:
       
        # get positions by symbol
        # ref. https://docs.alpaca.markets/reference/getopenposition-1
        #position = trade_client.get_open_position(symbol_or_asset_id=symbol)
        #num_qty=int(pd.DataFrame(position).iloc[6][1])
         # simple, limit order, fractional qty
        if ordertype=="limit":
                     req = LimitOrderRequest(
                        symbol = symbol,
                        qty = qty_open,
                        limit_price = val_buy,
                        side = OrderSide.BUY,
                        type = OrderType.LIMIT,
                        time_in_force = TimeInForce.GTC,
                        extended_hours=False,
                        stop_loss=StopLossRequest(stop_price=stop_sell2),
                        take_profit=TakeProfitRequest(limit_price=val_sell)
                    )
        elif price_increase_percent>pct_thresh:
            if qty_open>10:
            

                req = StopLimitOrderRequest(
                    symbol = symbol,
                    qty = qty_open,
                    side = OrderSide.SELL,
                    time_in_force = TimeInForce.GTC,
                    limit_price =  stop_sell1,
                    stop_price = stop_sell2
                    )
                res = trade_client.submit_order(req)
            else:
                req = TrailingStopOrderRequest(
                    symbol = symbol,
                    qty = qty_open,
                    side = OrderSide.SELL,
                    time_in_force = TimeInForce.GTC,
                    trail_percent=trail_pct
                    )
                res = trade_client.submit_order(req)
                
        elif current_price>=entry_price and current_price<val_sell:
            req = StopLimitOrderRequest(
                    symbol = symbol,
                    qty = qty_open,
                    side = OrderSide.SELL,
                    time_in_force = TimeInForce.GTC,
                    limit_price = val_sell,
                    stop_price = stop_sell1
                    )
            res = trade_client.submit_order(req)
            
        elif current_price<entry_price:
            req = StopLimitOrderRequest(
                    symbol = symbol,
                    qty = qty_open,
                    side = OrderSide.SELL,
                    time_in_force = TimeInForce.GTC,
                    limit_price = val_sell,
                    stop_price = stop_sell2
                    )
            res = trade_client.submit_order(req)
        elif ordertype=="short":
            req = StopLimitOrderRequest(
                    symbol = symbol,
                    qty = qty_open,
                    side = OrderSide.SELL,
                    time_in_force = TimeInForce.GTC,
                    limit_price = val_sell,
                    stop_price = stop_sell2
                    )
            res = trade_client.submit_order(req)
        elif ordertype=="limit.extended":
            if current_price<entry_price:
                
                req = StopLimitOrderRequest(
                        symbol = symbol,
                        qty = qty_open,
                        side = OrderSide.SELL,
                        time_in_force = TimeInForce.DAY,
                        extended_hours=True,
                        limit_price = val_sell,
                        stop_price = stop_sell2
                        )
                res = trade_client.submit_order(req)
            else:

                req = StopLimitOrderRequest(
                        symbol = symbol,
                        qty = qty_open,
                        side = OrderSide.SELL,
                        time_in_force = TimeInForce.DAY,
                        extended_hours=True,
                        limit_price = current_price,
                        stop_price = stop_sell1
                        )
                res = trade_client.submit_order(req)
        print("Selling {symbol}")
        return res
        
def get_latest_quote_alpaca(symbol):
    trade_api_url = None
    trade_api_wss = None
    data_api_url = None
    stream_data_wss = None
    # setup stock historical data client
    stock_historical_data_client = StockHistoricalDataClient(api_key, secret_key, url_override = data_api_url)
    # Request latest trade data for the specified symbol
    request_params = StockLatestTradeRequest(symbol_or_symbols=symbol)
    latest_trade = stock_historical_data_client.get_stock_latest_quote(request_params)

    # Print the premarket trade price
    return(latest_trade)

def get_latest_bar_alpaca(symbol):
    trade_api_url = None
    trade_api_wss = None
    data_api_url = None
    stream_data_wss = None
    # setup stock historical data client
    
    # Request latest trade data for the specified symbol
    request_params = StockLatestTradeRequest(symbol_or_symbols=symbol)
    latest_trade = stock_historical_data_client.get_stock_latest_bar(request_params)
    latest_trade=pd.DataFrame.from_dict(latest_trade)
    #latest_trade=latest_trade.tz_convert('America/New_York')

    # Print the premarket trade price
    return(latest_trade)

        
def get_latest_trade_alpaca(symbol):
    trade_api_url = None
    trade_api_wss = None
    data_api_url = None
    stream_data_wss = None
    # setup stock historical data client
    stock_historical_data_client = StockHistoricalDataClient(api_key, secret_key, url_override = data_api_url)
    # Request latest trade data for the specified symbol
    request_params = StockLatestTradeRequest(symbol_or_symbols=symbol)
    latest_trade = stock_historical_data_client.get_stock_latest_trade(request_params)

    # Print the premarket trade price
    return(latest_trade[symbol].price)


def get_latest_price_alpaca(symbol):
    trade_api_url = None
    trade_api_wss = None
    data_api_url = None
    stream_data_wss = None
    # setup stock historical data client
    stock_historical_data_client = StockHistoricalDataClient(api_key, secret_key, url_override = data_api_url)
    # Request latest trade data for the specified symbol
    request_params = StockLatestTradeRequest(symbol_or_symbols=symbol)
    latest_trade = stock_historical_data_client.get_stock_latest_trade(request_params)

    # Print the premarket trade price
    return(latest_trade[symbol].price)

def get_stockdata_alpaca(symbol,timeframe="hour"):
    trade_api_url = None
    trade_api_wss = None
    data_api_url = None
    stream_data_wss = None
    # setup stock historical data client
    stock_historical_data_client = StockHistoricalDataClient(api_key, secret_key, url_override = data_api_url)
    # Request latest trade data for the specified symbol
    request_params = StockLatestTradeRequest(symbol_or_symbols=symbol)
    now = datetime.now(ZoneInfo("America/New_York"))
    
    if timeframe=="30minute":
        req = StockBarsRequest(
            symbol_or_symbols = [symbol],
            timeframe=TimeFrame(amount = 30, unit = TimeFrameUnit.Minute),
            #, # specify timeframe- timedelta(days = 4)
            #start = now                          # specify start datetime, default=the beginning of the current day.
           # end_date=now - timedelta(days = 3)                                        # specify end datetime, default=now
            #limit = 1,                                               # specify limit
        )
    elif timeframe=="hour":
        req = StockBarsRequest(
            symbol_or_symbols = [symbol],
            timeframe=TimeFrame(amount = 1, unit = TimeFrameUnit.Hour),
            #, # specify timeframe- timedelta(days = 4)
            #start = now                          # specify start datetime, default=the beginning of the current day.
           # end_date=now - timedelta(days = 3)                                        # specify end datetime, default=now
            #limit = 1,                                               # specify limit
        )
    elif timeframe=="15minute":
        req = StockBarsRequest(
            symbol_or_symbols = [symbol],
            timeframe=TimeFrame(amount = 15, unit = TimeFrameUnit.Minute),
            #, # specify timeframe- timedelta(days = 4)
            #start = now                          # specify start datetime, default=the beginning of the current day.
           # end_date=now - timedelta(days = 3)                                        # specify end datetime, default=now
            #limit = 1,                                               # specify limit
        )
    elif timeframe=="1minute":
        req = StockBarsRequest(
            symbol_or_symbols = [symbol],
            timeframe=TimeFrame(amount = 1, unit = TimeFrameUnit.Minute),
            #, # specify timeframe- timedelta(days = 4)
            #start = now                          # specify start datetime, default=the beginning of the current day.
           # end_date=now - timedelta(days = 3)                                        # specify end datetime, default=now
            #limit = 1,                                               # specify limit
        )
    #stock_historical_data_client.get_stock_bars(req).df
    #latest_data=stock_historical_data_client.get_stock_bars(req) #.df.iloc[:,:] #.quantile(0.25).round(2)
    #print(mean_price)

    #last_price=stock_historical_data_client.get_stock_bars(req).df.iloc[-1:].quantile(0.25).round(2)
    latest_data = stock_historical_data_client.get_stock_bars(req).df.tz_convert('America/New_York', level=1)
    #print(ars_df)
    #last_price=ars_df['close']
    #print(latest_data.iloc[0:3,])
    latest_data.index.name = 'Timestamp'
    latest_data.reset_index(inplace=True)
  
    return(latest_data)


# Function to fetch historical OHLCV bars
def get_crypto_bars(symbol, timeframe="1Hour"):
    timeframes = {
        "1Min": TimeFrame.Minute,
       
       
        "1Hour": TimeFrame.Hour,
        "1Day": TimeFrame.Day
    }
    request_params = CryptoBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=timeframes[timeframe],
        start=datetime.utcnow() - timedelta(days=1),  # Last 24h
        #limit=limit,
        feed=DataFeed.SIP
    )
    bars = crypto_client.get_crypto_bars(request_params)
    df = bars.df.loc[symbol].reset_index()
    df["timestamp"] = df["timestamp"].dt.tz_convert('America/New_York')
    return df #bars[symbol]


# Function to get top crypto movers (gainers/losers)
def get_crypto_movers():
    # Get top gainers/losers based on percent change
    assets = trade_client.get_all_assets()
    
    crypto_assets = [a for a in assets if a.asset_class == "crypto"]
    
    movers = []
    for asset in crypto_assets:
        symbol = asset.symbol
        print(symbol)
        try:
            quote = get_crypto_bars(symbol) #get_latest_crypto_quote #get_latest_crypto_quote(symbol)
            print(quote[-2:])
            price_change=0
            price_change = 100*(quote.iloc[-1]["close"] - quote.iloc[0]["close"]) / quote.iloc[0]["close"]
            #print(price_change)
            #price_change = (quote.ask_price - quote.bid_price) / quote.bid_price * 100
            movers.append((symbol, price_change))
        except Exception as e:
            print(e)
            continue
    
    movers.sort(key=lambda x: x[1], reverse=True)
    #return {"gainers": movers[:5].df, "losers": movers[-5:].df}
    df=pd.DataFrame(movers)
    df.columns = ['symbol', 'percent_change']
    #filtered_df = df[df['Symbol'].str.contains("USD$", case=False)]
    filtered_df = df[df['symbol'].str.endswith('USD', na=False)]
    return filtered_df


# Function to fetch historical OHLCV bars
def get_crypto_data_days(symbol, days=1,daily=False):
    
    today = date.today()
    start_date = today - timedelta(days=days)
    
    if daily==False:
        request_params = CryptoBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start_date,
            feed=DataFeed.SIP
        )
    else:
        request_params = CryptoBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame(amount = 1, unit = TimeFrameUnit.Minute),
            feed=DataFeed.SIP
        )
        
    bars = crypto_client.get_crypto_bars(request_params)
    df = bars.df.loc[symbol].reset_index()
    df["timestamp"] = df["timestamp"].dt.tz_convert('America/New_York')
    return df #bars[symbol]

def get_latest_cryptoprice_alpaca(symbol,days=1,daily=True):
    df=get_crypto_data_days(symbol, days,daily)
    return df["close"].iloc[-1]

def get_stock_data_window(symbol,start_date,end_date,timeframe="Minute",days=200):
    #today = date.today()
    #start_date = today - timedelta(days=days)
    
    #begin_date=date.today()-timedelta(days=days) #start_date #"2025-01-01" #start_date - timedelta(days=days)
    #trading_days = pd.bdate_range(start=start_date, end=end_date)
    #print(trading_days)trading_days[0]trading_days[-1
    
    if timeframe=="Minute":
        request_params = StockBarsRequest(
            symbol_or_symbols = [symbol],
            timeframe=TimeFrame(amount = 1, unit = TimeFrameUnit.Minute),
            start=f"{start_date}T00:00:00Z",
            end=f"{end_date}T23:59:59Z"
            
            
            
        )
    else:
       request_params = StockBarsRequest(
          symbol_or_symbols = [symbol],
          timeframe=TimeFrame(amount = 1, unit = TimeFrameUnit.Day),
          start=f"{start_date}T00:00:00Z",
          end=f"{end_date}T23:59:59Z"
          
          
      )
    bars = historical_client.get_stock_bars(request_params).df.tz_convert('America/New_York', level=1)
    df = bars.loc[symbol]
    return df
  
  
def get_stock_data_date(symbol,start_date,timeframe="Minute",days=200):
    #today = date.today()
    #start_date = today - timedelta(days=days)
    
    if timeframe=="Minute":
        request_params = StockBarsRequest(
            symbol_or_symbols = [symbol],
            timeframe=TimeFrame(amount = 1, unit = TimeFrameUnit.Minute),
            start=f"{start_date}T00:00:00Z",
            end=f"{start_date}T23:59:59Z"
            
            
        )
    else:
        begin_date=start_date-timedelta(days=days) #datetime.strptime(start_date, "%Y-%m-%d")-timedelta(days=days) #date.today()-timedelta(days=days) #start_date #"2025-01-01" #start_date - timedelta(days=days)
        trading_days = pd.bdate_range(start=begin_date, end=start_date)
        trading_days=trading_days.strftime('%Y-%m-%d').tolist()
        #print(trading_days)
        #print(trading_days[-1])
        request_params = StockBarsRequest(
            symbol_or_symbols = [symbol],
            timeframe=TimeFrame(amount = 1, unit = TimeFrameUnit.Day),
            start=f"{trading_days[0]}T00:00:00Z",
            end=f"{trading_days[-1]}T23:59:59Z"
            
            
        )
        
    bars = historical_client.get_stock_bars(request_params).df.tz_convert('America/New_York', level=1)

    df = bars.loc[symbol]
    if timeframe=="DAY":
        #df.index=df.index.strptime(start_date, "%Y-%m-%d")
        df = df.reset_index().rename(columns={'index': 'timestamp'})
        
        #df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df
  
def get_stock_data_daily(symbol, days):
    today = date.today()
    start_date = today - timedelta(days=days)
    
    request_params = StockBarsRequest(
        symbol_or_symbols = [symbol],
        timeframe=TimeFrame(amount = 1, unit = TimeFrameUnit.Minute)
        
    )
    bars = historical_client.get_stock_bars(request_params).df.tz_convert('America/New_York', level=1)
    df = bars.loc[symbol]
    return df
  
def get_stock_data_days(symbol, days,daily=False,start_date=None):
    now = datetime.now(ZoneInfo("America/New_York")).time()
    market_time=datetime.strptime("09:30", "%H:%M").time()
    if market_time<=now:
        today = date.today()
    else:
        today=date.today()-timedelta(days=1)
    #print(start_date)
    if start_date is None:
        start_date=today
    #else:
        #start_date = today - timedelta(days=days)
    
    if daily==False:
        start_date = today - timedelta(days=days)
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=f"{start_date}T00:00:00Z",
            end=f"{today}T23:59:59Z"
        )
    else:
        #print(start_date)
       # print(today)
        start_date = today #- timedelta(days=days)
        request_params = StockBarsRequest(
            symbol_or_symbols = [symbol],
            timeframe=TimeFrame(amount = 1, unit = TimeFrameUnit.Minute),
            start=f"{start_date}T00:00:00Z",
            end=f"{today}T23:59:59Z"
            
        )
    bars = historical_client.get_stock_bars(request_params).df.tz_convert('America/New_York', level=1)
    df = bars.loc[symbol]
    return df

def evaluate_stock_health_daily_fromdf(df):
    
    last_50_days = df.tail(50)
    last_20_days = df.tail(20)
    last_120_days = df.tail(120)
    last_10_days = df.tail(10)
    # Condition 1: Stock must be above 200-day MA (long-term uptrend)
    is_above_200_MA = df["close"].iloc[-1] > df["200_MA"].iloc[-1]

    # Condition 2: Stock must not have a steady downtrend in last 6 months
    price_trend_short = np.polyfit(range(len(last_10_days)), last_10_days["close"], 1)[0]  # Linear trend slope
    price_trend_long = np.polyfit(range(len(last_120_days)), last_120_days["close"], 1)[0]  # Linear trend slope
    is_uptrend = (price_trend_short > (-1) or price_trend_long > (-1))
    #print(price_trend_short)
    #print(price_trend_long)
    #price_trend = np.polyfit(range(len(last_20_days)), last_20_days["close"], 1)[0]  # Linear trend slope
    #is_uptrend = price_trend > 0

    # Condition 3: Avoid oversold or weak momentum stocks
    avg_rsi = last_50_days["RSI_14"].mean()
    avg_roc = last_50_days["ROC_10"].mean()
    has_good_momentum = avg_rsi > 40 or avg_roc > (-1)
    
    return is_uptrend or has_good_momentum
  
def evaluate_stock_health_daily(symbol,assettype="stocks"):
    df=calculate_indicators_histall(symbol,1500,True,assettype)
    last_50_days = df.tail(50)
    last_20_days = df.tail(20)
    last_120_days = df.tail(120)
    last_10_days = df.tail(10)
    # Condition 1: Stock must be above 200-day MA (long-term uptrend)
    is_above_200_MA = df["close"].iloc[-1] > df["200_MA"].iloc[-1]

    # Condition 2: Stock must not have a steady downtrend in last 6 months
    price_trend_short = np.polyfit(range(len(last_10_days)), last_10_days["close"], 1)[0]  # Linear trend slope
    price_trend_long = np.polyfit(range(len(last_120_days)), last_120_days["close"], 1)[0]  # Linear trend slope
    is_uptrend = (price_trend_short > (-1) or price_trend_long > (-1))
    #print(price_trend_short)
    #print(price_trend_long)
    #price_trend = np.polyfit(range(len(last_20_days)), last_20_days["close"], 1)[0]  # Linear trend slope
    #is_uptrend = price_trend > 0

    # Condition 3: Avoid oversold or weak momentum stocks
    avg_rsi = last_10_days["RSI_14"].mean()
    avg_roc = last_10_days["ROC_10"].mean()
    has_good_momentum = avg_rsi > 40 or avg_roc > (-0.5)
    #print(avg_rsi)
    #print(avg_roc)
    #print(has_good_momentum)
    
    return is_uptrend and has_good_momentum

def evaluate_stock_health(symbol,daily=True,assettype="stocks"):
    if daily==True:
      
        df=calculate_indicators_histall(symbol,1500,daily,assettype)
    else:
        df=calculate_indicators_histall(symbol,200,daily,assettype)
    #print(df.iloc[0:10])
    last_50_days = df.tail(50)
    last_20_days = df.tail(20)
    last_120_days = df.tail(120)
    last_10_days = df.tail(10)
    # Condition 1: Stock must be above 200-day MA (long-term uptrend)
    is_above_200_MA = df["close"].iloc[-1] > df["200_MA"].iloc[-1]

    # Condition 2: Stock must not have a steady downtrend in last 6 months
    price_trend_short = np.polyfit(range(len(last_10_days)), last_10_days["close"], 1)[0]  # Linear trend slope
    price_trend_long = np.polyfit(range(len(last_120_days)), last_120_days["close"], 1)[0]  # Linear trend slope
    is_uptrend = (price_trend_short > (-0.1) or price_trend_long > (-0.2)) and (price_trend_short > (-0.1))
    #print(price_trend_short)
    #print(price_trend_long)
    #print(is_uptrend)
    #price_trend = np.polyfit(range(len(last_20_days)), last_20_days["close"], 1)[0]  # Linear trend slope
    #is_uptrend = price_trend > 0

    # Condition 3: Avoid oversold or weak momentum stocks
    avg_rsi = last_10_days["RSI_14"].mean()
    avg_roc = last_10_days["ROC_10"].mean()
    has_good_momentum = avg_rsi >40 and avg_rsi < 65 and avg_roc > (-0.5)
    #print(avg_rsi)
    #print(avg_roc)
    #print(has_good_momentum)
    
    return is_uptrend and has_good_momentum  
  

def evaluate_stock_trend_daily_etfs(symbol,assettype="stocks"):
    df=calculate_indicators_histall(symbol,1500,True,assettype)
    last_50_days = df.tail(50)
    last_20_days = df.tail(20)
    last_120_days = df.tail(120)
    last_10_days = df.tail(10)
    # Condition 1: Stock must be above 200-day MA (long-term uptrend)
    is_above_200_MA = df["close"].iloc[-1] > df["200_MA"].iloc[-1]

    # Condition 2: Stock must not have a steady downtrend in last 6 months
    price_trend_short = np.polyfit(range(len(last_10_days)), last_10_days["close"], 1)[0]  # Linear trend slope
    price_trend_long = np.polyfit(range(len(last_120_days)), last_120_days["close"], 1)[0]  # Linear trend slope
    is_uptrend = (price_trend_short > (-0.25) or price_trend_long > (-0.25)) and (price_trend_short < (0.05))
    #print(price_trend_short)
    #print(price_trend_long)
    #print(is_uptrend)
    #price_trend = np.polyfit(range(len(last_20_days)), last_20_days["close"], 1)[0]  # Linear trend slope
    #is_uptrend = price_trend > 0

    # Condition 3: Avoid oversold or weak momentum stocks
    avg_rsi = last_10_days["RSI_14"].mean()
    avg_roc = last_10_days["ROC_10"].mean()
    has_good_momentum = avg_rsi >40 and avg_rsi < 65 and avg_roc > (-0.5)
    #print(avg_rsi)
    #print(avg_roc)
    #print(has_good_momentum)
    
    return is_uptrend and has_good_momentum  
  
# Stock Health Evaluation: Filter out poorly performing stocks
def evaluate_stock_health_old(symbol,assettype="stocks",start_date=None):
    
    df=calculate_indicators_histall(symbol,1500,False,assettype,start_date)
    last_6_months = df.tail(1260)  # Approx. 6 months of 5-min data
    last_50_days = df.tail(50)
    last_20_days = df.tail(20)
    last_120_days = df.tail(120)
    last_10_days = df.tail(10)
    # Condition 1: Stock must be above 200-day MA (long-term uptrend)
    is_above_200_MA = df["close"].iloc[-1] > df["200_MA"].iloc[-1]

    # Condition 2: Stock must not have a steady downtrend in last 6 months
    price_trend_short = np.polyfit(range(len(last_10_days)), last_10_days["close"], 1)[0]  # Linear trend slope
    price_trend_long = np.polyfit(range(len(last_120_days)), last_120_days["close"], 1)[0]  # Linear trend slope
    is_uptrend = (price_trend_short > (-1) and price_trend_long > (-1))
    #print(price_trend_short)
    #print(price_trend_long)
    #price_trend = np.polyfit(range(len(last_20_days)), last_20_days["close"], 1)[0]  # Linear trend slope
    #is_uptrend = price_trend > 0

    # Condition 3: Avoid oversold or weak momentum stocks
    avg_rsi = last_50_days["RSI_14"].mean()
    avg_roc = last_50_days["ROC_10"].mean()
    has_good_momentum = avg_rsi > 40 and avg_roc > (-1)
    #print(is_above_200_MA)
    #print(has_good_momentum)
    #print(price_trend_short)
    #print(price_trend_long)
    #print(is_uptrend)
    #print(avg_rsi)
    #print(avg_roc)
    #is_above_200_MA and 
    return is_uptrend or has_good_momentum

# Stock Health Evaluation: Filter out poorly performing stocks
def evaluate_df_health(df):
    
    last_6_months = df.tail(1260)  # Approx. 6 months of 5-min data
    last_50_days = df.tail(50)
    last_20_days = df.tail(20)
    last_120_days = df.tail(120)
    last_10_days = df.tail(10)
    # Condition 1: Stock must be above 200-day MA (long-term uptrend)
    is_above_200_MA = df["close"].iloc[-1] > df["200_MA"].iloc[-1]

    # Condition 2: Stock must not have a steady downtrend in last 6 months
    price_trend_short = np.polyfit(range(len(last_10_days)), last_10_days["close"], 1)[0]  # Linear trend slope
    price_trend_long = np.polyfit(range(len(last_120_days)), last_120_days["close"], 1)[0]  # Linear trend slope
    is_uptrend = (price_trend_short > (-1) or price_trend_long > (-1))

    #price_trend = np.polyfit(range(len(last_20_days)), last_20_days["close"], 1)[0]  # Linear trend slope
    #is_uptrend = price_trend > 0

    # Condition 3: Avoid oversold or weak momentum stocks
    avg_rsi = last_50_days["RSI_14"].mean()
    avg_roc = last_50_days["ROC_10"].mean()
    has_good_momentum = avg_rsi > 40 or avg_roc > (-1)
    #print(is_above_200_MA)
    #print(has_good_momentum)
    #print(price_trend_short)
    #print(price_trend_long)
    #print(is_uptrend)
    #print(avg_rsi)
    #print(avg_roc)
    #is_above_200_MA and 
    return is_uptrend or has_good_momentum

# List of tickers to backtest
TICKERS = ["SOXL", "SOXS", "TSLA", "AMZN", "META", "NVDA", "AVGO", "AAPL"]

def calculate_indicators_hist_fromdf_v2(df):
    #df = get_stock_data_days(symbol, 200)
    df["l5_MA"] = df["close"].rolling(window=5).mean()
    df["l20_MA"] = df["close"].rolling(window=20).mean()
    df["l50_MA"] = df["close"].rolling(window=50).mean()
    df["l120_MA"] = df["close"].rolling(window=120).mean()
    df["lPivot_Point"] = (df["high"].rolling(5).mean() + df["low"].rolling(5).mean() + df["close"].rolling(5).mean()) / 3
    df["lPrev_Close"] = df["close"].shift(1)
    df["lSMA_20"] = ta.trend.sma_indicator(df["close"], window=20)
    df["lEMA_20"] = ta.trend.ema_indicator(df["close"], window=20)
    df["lRSI_14"] = ta.momentum.rsi(df["close"], window=14)
    df["lfastK"] = ta.momentum.stoch(df["high"], df["low"], df["close"], window=14, smooth_window=3)
    df['lfastK_5'] = df['lfastK'].rolling(5).mean()
    df["lfastD"] = ta.momentum.stoch_signal(df["high"], df["low"], df["close"], window=14, smooth_window=3)
    df["lROC_5"] = ta.momentum.roc(df["close"], window=5)
    df["lROC_10"] = ta.momentum.roc(df["close"], window=10)
    df['l20_STD'] = df['close'].rolling(window=20).std()
    df['lupper_band'] = df['l20_MA'] + (2 * df['l20_STD'])
    df['llower_band'] = df['l20_MA'] - (2 * df['l20_STD'])

    #df['roc_5']=df['roc'].rolling(window=5).mean()
    #df['roc_prev'] = df['roc'].shift(1)
    df['lmomentum_1'] = df['close'].pct_change(periods=1) * 100 #df['close'].diff(1)
    df['lmomentum_3'] = df['close'].pct_change(periods=3) * 100 #df['close'].diff(3)
    df['lmomentum_5'] = df['close'].pct_change(periods=5) * 100 #df['close'].diff(5)
    
    df['lbb_width'] = 100*(df['lupper_band']-df['llower_band'])/df['lSMA_20']  #BollingerBands(df['close']).bollinger_wband()
    df['lvolume_change'] = df['volume'].pct_change(periods=5) * 100
    df['lavg_vol_10'] = df['volume'].rolling(10).mean()
    df['lvolume_change_ratio'] = df['volume'] / df['lavg_vol_10']
    
    df['lbb_width_5'] = df['lbb_width'].rolling(5).mean()
    
    # Calculate MACD (12-day and 26-day EMA, Signal Line with 9-day EMA)
    df['lMACD'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
    df['lSignal_Line'] = df['lMACD'].ewm(span=9, adjust=False).mean()
    df['lMACD_histogram'] = df['lMACD'] - df['lSignal_Line']
    
    #df['lATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)


    
    # Calculate Pivot Points
    df['lpivot_point'] = (df['high'].rolling(5).mean() + df['low'].rolling(5).mean() + df['close'].rolling(5).mean()) / 3
    df['lS1'] = 2 * df['lpivot_point'] - df['high'].rolling(5).mean()
    df['lS2'] = df['lpivot_point'] - (df['high'].rolling(5).mean() - df['low'].rolling(5).mean())
    df['lR1'] = 2 * df['lpivot_point'] - df['low'].rolling(5).mean()
    df['lR2'] = df['lpivot_point'] + (df['high'].rolling(5).mean() - df['low']).rolling(5).mean()

    try:
        df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
    except Exception as e:
        df['adx']=0
    df['volume_surge'] = df['volume'] > 1.5 * df['lavg_vol_10']
    #macd = ta.trend.MACD(df['close'])
    #df['macd_hist'] = macd.macd_diff()
    # Support/Resistance zones
    df['support'] = df['low'].rolling(window=5).min()
    df['resistance'] = df['high'].rolling(window=5).max()
    
    return df #.iloc[-1]  # Return the latest row
  
def calculate_indicators_hist_fromdf(df):
    #df = get_stock_data_days(symbol, 200)
    df["l5_MA"] = df["close"].rolling(window=5).mean()
    df["l20_MA"] = df["close"].rolling(window=20).mean()
    df["l50_MA"] = df["close"].rolling(window=50).mean()
    df["l120_MA"] = df["close"].rolling(window=120).mean()
    df["lPivot_Point"] = (df["high"].rolling(5).mean() + df["low"].rolling(5).mean() + df["close"].rolling(5).mean()) / 3
    df["lPrev_Close"] = df["close"].shift(1)
    df["lSMA_20"] = ta.trend.sma_indicator(df["close"], window=20)
    df["lEMA_20"] = ta.trend.ema_indicator(df["close"], window=20)
    df["lRSI_14"] = ta.momentum.rsi(df["close"], window=14)
    df["lfastK"] = ta.momentum.stoch(df["high"], df["low"], df["close"], window=14, smooth_window=3)
    df["lfastD"] = ta.momentum.stoch_signal(df["high"], df["low"], df["close"], window=14, smooth_window=3)
    df["lROC_10"] = ta.momentum.roc(df["close"], window=10)
    df['l20_STD'] = df['close'].rolling(window=20).std()
    df['lupper_band'] = df['l20_MA'] + (2 * df['l20_STD'])
    df['llower_band'] = df['l20_MA'] - (2 * df['l20_STD'])

    # Calculate MACD (12-day and 26-day EMA, Signal Line with 9-day EMA)
    df['lMACD'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
    df['lSignal_Line'] = df['lMACD'].ewm(span=9, adjust=False).mean()
    df['lMACD_histogram'] = df['lMACD'] - df['lSignal_Line']
    
    #df['lATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)


    # Calculate Pivot Points
    df['lpivot_point'] = (df['high'] + df['low'] + df['close']) / 3
    df['lS1'] = 2 * df['lpivot_point'] - df['high']
    df['lS2'] = df['lpivot_point'] - (df['high'] - df['low'])
    df['lR1'] = 2 * df['lpivot_point'] - df['low']
    df['lR2'] = df['lpivot_point'] + (df['high'] - df['low'])

    
    return df.iloc[-1]  # Return the latest row

def calculate_indicators_hist_date(symbol,start_date=None,timeframe="Minute"):
    df = get_stock_data_date(symbol, start_date,timeframe)
    df["l5_MA"] = df["close"].rolling(window=5).mean()
    df["l20_MA"] = df["close"].rolling(window=20).mean()
    df["l50_MA"] = df["close"].rolling(window=50).mean()
    df["l120_MA"] = df["close"].rolling(window=120).mean()
    df["lPivot_Point"] = (df["high"].rolling(5).mean() + df["low"].rolling(5).mean() + df["close"].rolling(5).mean()) / 3
    df["lPrev_Close"] = df["close"].shift(1)
    df["lSMA_20"] = ta.trend.sma_indicator(df["close"], window=20)
    df["lEMA_20"] = ta.trend.ema_indicator(df["close"], window=20)
    df["lRSI_14"] = ta.momentum.rsi(df["close"], window=14)
    df["lfastK"] = ta.momentum.stoch(df["high"], df["low"], df["close"], window=14, smooth_window=3)
    df["lfastD"] = ta.momentum.stoch_signal(df["high"], df["low"], df["close"], window=14, smooth_window=3)
    df["lROC_10"] = ta.momentum.roc(df["close"], window=10)
    df['l20_STD'] = df['close'].rolling(window=20).std()
    df['lupper_band'] = df['l20_MA'] + (2 * df['l20_STD'])
    df['llower_band'] = df['l20_MA'] - (2 * df['l20_STD'])
    df['bollinger_width'] = 100*(df['lupper_band']-df['llower_band'])/df['l20_MA']  #BollingerBands(df['close']).bollinger_wband()
    df['bollinger_width_20MA'] =df['bollinger_width'].rolling(20).mean()
    
    # Calculate MACD (12-day and 26-day EMA, Signal Line with 9-day EMA)
    df['lMACD'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
    df['lSignal_Line'] = df['lMACD'].ewm(span=9, adjust=False).mean()
    df['lMACD_histogram'] = df['lMACD'] - df['lSignal_Line']
    
    #df['lATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    #df["lATR_MA"] = df["lATR"].rolling(window=20).mean()
    df["lPercentChange"]=100*(df["close"]-df["close"].shift(1))/df["close"].shift(1)
    df["lfastDfastK"]=df["lfastD"]/(df["lfastK"]+0.01)

    # Calculate Pivot Points
    df['lpivot_point'] = (df['high'] + df['low'] + df['close']) / 3
    df['lS1'] = 2 * df['lpivot_point'] - df['high']
    df['lS2'] = df['lpivot_point'] - (df['high'] - df['low'])
    df['lR1'] = 2 * df['lpivot_point'] - df['low']
    df['lR2'] = df['lpivot_point'] + (df['high'] - df['low'])
    
    # Extract last 15-minute stats
    #last_15min = df.iloc[-15:]
    
    #df["price_now"]=df['close'] #.iloc[-1],
    
    #df["rate_of_change"]= df['roc'].iloc[-1],
    
    
    
    return df.iloc[-1],df.iloc[-2]  # Return the latest and prev row

def calculate_indicators_hist(symbol,start_date=None):
    df = get_stock_data_days(symbol, 200, False,start_date)
    df["l5_MA"] = df["close"].rolling(window=5).mean()
    df["l20_MA"] = df["close"].rolling(window=20).mean()
    df["l50_MA"] = df["close"].rolling(window=50).mean()
    df["l120_MA"] = df["close"].rolling(window=120).mean()
    df["lPivot_Point"] = (df["high"].rolling(5).mean() + df["low"].rolling(5).mean() + df["close"].rolling(5).mean()) / 3
    df["lPrev_Close"] = df["close"].shift(1)
    df["lSMA_20"] = ta.trend.sma_indicator(df["close"], window=20)
    df["lEMA_20"] = ta.trend.ema_indicator(df["close"], window=20)
    df["lRSI_14"] = ta.momentum.rsi(df["close"], window=14)
    df["lfastK"] = ta.momentum.stoch(df["high"], df["low"], df["close"], window=14, smooth_window=3)
    df["lfastD"] = ta.momentum.stoch_signal(df["high"], df["low"], df["close"], window=14, smooth_window=3)
    df["lROC_10"] = ta.momentum.roc(df["close"], window=10)
    df['l20_STD'] = df['close'].rolling(window=20).std()
    df['lupper_band'] = df['l20_MA'] + (2 * df['l20_STD'])
    df['llower_band'] = df['l20_MA'] - (2 * df['l20_STD'])

    # Calculate MACD (12-day and 26-day EMA, Signal Line with 9-day EMA)
    df['lMACD'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
    df['lSignal_Line'] = df['lMACD'].ewm(span=9, adjust=False).mean()
    df['lMACD_histogram'] = df['lMACD'] - df['lSignal_Line']
    
    df['lATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    df["lATR_MA"] = df["lATR"].rolling(window=20).mean()
    df["lPercentChange"]=100*(df["close"]-df["close"].shift(1))/df["close"].shift(1)
    df["lfastDfastK"]=df["lfastD"]/(df["lfastK"]+0.01)

    # Calculate Pivot Points
    df['lpivot_point'] = (df['high'] + df['low'] + df['close']) / 3
    df['lS1'] = 2 * df['lpivot_point'] - df['high']
    df['lS2'] = df['lpivot_point'] - (df['high'] - df['low'])
    df['lR1'] = 2 * df['lpivot_point'] - df['low']
    df['lR2'] = df['lpivot_point'] + (df['high'] - df['low'])

    
    return df.iloc[-1]  # Return the latest row

# Function to compute indicators
def calculate_indicators(df):
    df["SMA_20"] = ta.trend.sma_indicator(df["close"], window=20)
    df["EMA_20"] = ta.trend.ema_indicator(df["close"], window=20)
    df["RSI_14"] = ta.momentum.rsi(df["close"], window=14)
    df["fastK"] = ta.momentum.stoch(df["high"], df["low"], df["close"], window=14, smooth_window=3)
    df["fastD"] = ta.momentum.stoch_signal(df["high"], df["low"], df["close"], window=14, smooth_window=3)
    df["ROC_10"] = ta.momentum.roc(df["close"], window=10)

    return df

def calculate_rsi(df):

    df["RSI_14"] = ta.momentum.rsi(df["close"], window=14)
    last_row_index = df.index[-1]
    return df.loc[last_row_index,"RSI_14"] 

def calculate_indicators_histall_fromdf(df):
        
    df["trade_count_MA"] = df["trade_count"].rolling(window=20).mean()
    df["volume_MA"] = df["volume"].rolling(window=20).mean()
    df["5_MA"] = df["close"].rolling(window=5).mean()
    df["7_MA"] = df["close"].rolling(window=7).mean()
    df["10_MA"] = df["close"].rolling(window=10).mean()
    df["20_MA"] = df["close"].rolling(window=20).mean()
    df["50_MA"] = df["close"].rolling(window=50).mean()
    df["120_MA"] = df["close"].rolling(window=120).mean()
    df["200_MA"] = df["close"].rolling(window=200).mean()
    df['above_ma_5'] = (df['close'] > df['5_MA']).astype(int)
    df['above_ma_10'] = (df['close'] > df['10_MA']).astype(int)
    df['above_ma_50'] = (df['close'] > df['50_MA']).astype(int)
    df['above_ma_120'] = (df['close'] > df['120_MA']).astype(int)
    df['above_ma_5_3MA']=df['above_ma_5'].rolling(window=3).mean()
    df['above_ma_50_3MA']=df['above_ma_50'].rolling(window=3).mean()
    df["Pivot_Point"] = (df["high"].rolling(5).mean() + df["low"].rolling(5).mean() + df["close"].rolling(5).mean()) / 3
    df["Pivot_Point_long"] = (df["high"].rolling(50).mean() + df["low"].rolling(50).mean() + df["close"].rolling(50).mean()) / 3
    
    df['ATR'] = 0 #ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    df["ATR_MA"] = 0 #df["ATR"].rolling(window=20).mean()

    # Calculate Pivot Points
    df['pivot_point'] = (df['high'] + df['low'] + df['close']) / 3
    df['S1'] = 2 * df['pivot_point'] - df['high']
    df['S2'] = df['pivot_point'] - (df['high'] - df['low'])
    df['R1'] = 2 * df['pivot_point'] - df['low']
    df['R2'] = df['pivot_point'] + (df['high'] - df['low'])

    
    df["Prev_Close"] = df["close"].shift(1)
    df["SMA_20"] = ta.trend.sma_indicator(df["close"], window=20)
    df["EMA_20"] = ta.trend.ema_indicator(df["close"], window=20)
    df["RSI_14"] = ta.momentum.rsi(df["close"], window=14)
    df["fastK"] = ta.momentum.stoch(df["high"], df["low"], df["close"], window=14, smooth_window=3)
    df["fastD"] = ta.momentum.stoch_signal(df["high"], df["low"], df["close"], window=14, smooth_window=3)
    df["ROC_10"] = ta.momentum.roc(df["close"], window=10)
    df['momentum'] = df['close'].diff(10)
    df['momentum_3MA']=df['momentum'].rolling(window=3).mean()
    df["PercentChange"]=100*(df["close"]-df["close"].shift(1))/df["close"].shift(1)
    df["fastDfastK"]=df["fastD"]/(df["fastK"]+0.01)
    df["ROC_10_last3"]=df["ROC_10"].rolling(window=3).mean()
    df["ROC_10_last10"]=df["ROC_10"].rolling(window=10).mean()
    #last_10 = df['ROC_10'].tail(10)
    #correlation = last_10.corr(last_10)  # Self-correlation (always 1.0 unless NaN values exist)
    #mean_value = last_10.mean()
    df['20_STD'] = df['close'].rolling(window=20).std()
    df['upper_band'] = df['20_MA'] + (2 * df['20_STD'])
    df['lower_band'] = df['20_MA'] - (2 * df['20_STD'])
    df['bollinger_width'] = 100*(df['upper_band']-df['lower_band'])/df['20_MA']  #BollingerBands(df['close']).bollinger_wband()
    
    # Calculate MACD (12-day and 26-day EMA, Signal Line with 9-day EMA)
    df['MACD'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_histogram'] = df['MACD'] - df['Signal_Line']
    
    df["RSI_14_3ma"]=df["RSI_14"].rolling(3).mean()
    df["RSI_14_20ma"]=df["RSI_14"].rolling(20).mean()
    df["RSI_14_50ma"]=df["RSI_14"].rolling(50).mean()
    
    df["fastK_3ma"]=df["fastK"].rolling(3).mean()
    df["fastK_20ma"]=df["fastK"].rolling(20).mean()
    df["fastK_50ma"]=df["fastK"].rolling(50).mean()
    
    df['return'] = df['close'].pct_change()
    df['volatility'] = df['return'].rolling(window=5).std()
    df['rsi'] = RSIIndicator(df['close']).rsi()
    
    #df_long=calculate_indicators_hist(symbol) #.iloc[-1]
    # Merge DataFrames using cross join
    #merged_df = df.merge(df_long, how='cross')
    
    #print(merged_df.iloc[-1])
    time.sleep(0.1)
    return df #.dropna()

def calculate_indicators_histall(symbol,days=360,daily=False,assettype="stocks",start_date=None):
    if assettype=="stocks":
        
        df = get_stock_data_days(symbol, days,daily,start_date)
        #df2=get_stock_data_days(symbol, days,daily=False)
    else:
        df = get_crypto_data_days(symbol, days,daily=daily)
    df["trade_count_MA"] = df["trade_count"].rolling(window=20).mean()
    df["volume_MA"] = df["volume"].rolling(window=20).mean()
    df["5_MA"] = df["close"].rolling(window=5).mean()
    df["7_MA"] = df["close"].rolling(window=7).mean()
    df["10_MA"] = df["close"].rolling(window=10).mean()
    df["20_MA"] = df["close"].rolling(window=20).mean()
    df["50_MA"] = df["close"].rolling(window=50).mean()
    df["120_MA"] = df["close"].rolling(window=120).mean()
    df["200_MA"] = df["close"].rolling(window=200).mean()
    df['above_ma_5'] = (df['close'] > df['5_MA']).astype(int)
    df['above_ma_10'] = (df['close'] > df['10_MA']).astype(int)
    df['above_ma_50'] = (df['close'] > df['50_MA']).astype(int)
    df['above_ma_120'] = (df['close'] > df['120_MA']).astype(int)
    df['above_ma_5_3MA']=df['above_ma_5'].rolling(window=3).mean()
    df['above_ma_50_3MA']=df['above_ma_50'].rolling(window=3).mean()
    df["Pivot_Point"] = (df["high"].rolling(5).mean() + df["low"].rolling(5).mean() + df["close"].rolling(5).mean()) / 3
    df["Pivot_Point_long"] = (df["high"].rolling(50).mean() + df["low"].rolling(50).mean() + df["close"].rolling(50).mean()) / 3
    
    df['ATR'] = 0 #ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    df["ATR_MA"] = 0 #df["ATR"].rolling(window=20).mean()

    # Calculate Pivot Points
    df['pivot_point'] = (df['high'] + df['low'] + df['close']) / 3
    df['S1'] = 2 * df['pivot_point'] - df['high']
    df['S2'] = df['pivot_point'] - (df['high'] - df['low'])
    df['R1'] = 2 * df['pivot_point'] - df['low']
    df['R2'] = df['pivot_point'] + (df['high'] - df['low'])

    
    df["Prev_Close"] = df["close"].shift(1)
    df["SMA_20"] = ta.trend.sma_indicator(df["close"], window=20)
    df["EMA_20"] = ta.trend.ema_indicator(df["close"], window=20)
    df["RSI_14"] = ta.momentum.rsi(df["close"], window=14)
    df["fastK"] = ta.momentum.stoch(df["high"], df["low"], df["close"], window=14, smooth_window=3)
    df["fastD"] = ta.momentum.stoch_signal(df["high"], df["low"], df["close"], window=14, smooth_window=3)
    df["ROC_10"] = ta.momentum.roc(df["close"], window=10)
    df['momentum'] = df['close'].diff(10)
    df['momentum_3MA']=df['momentum'].rolling(window=3).mean()
    df["PercentChange"]=100*(df["close"]-df["close"].shift(1))/df["close"].shift(1)
    df["fastDfastK"]=df["fastD"]/(df["fastK"]+0.01)
    df["ROC_10_last3"]=df["ROC_10"].rolling(window=3).mean()
    df["ROC_10_last10"]=df["ROC_10"].rolling(window=10).mean()
    #last_10 = df['ROC_10'].tail(10)
    #correlation = last_10.corr(last_10)  # Self-correlation (always 1.0 unless NaN values exist)
    #mean_value = last_10.mean()
    df['20_STD'] = df['close'].rolling(window=20).std()
    df['upper_band'] = df['20_MA'] + (2 * df['20_STD'])
    df['lower_band'] = df['20_MA'] - (2 * df['20_STD'])
    df['bollinger_width'] = 100*(df['upper_band']-df['lower_band'])/df['20_MA']  #BollingerBands(df['close']).bollinger_wband()
    
    # Calculate MACD (12-day and 26-day EMA, Signal Line with 9-day EMA)
    df['MACD'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_histogram'] = df['MACD'] - df['Signal_Line']
    
    df["RSI_14_3ma"]=df["RSI_14"].rolling(3).mean()
    df["RSI_14_20ma"]=df["RSI_14"].rolling(20).mean()
    df["RSI_14_50ma"]=df["RSI_14"].rolling(50).mean()
    
    df["fastK_3ma"]=df["fastK"].rolling(3).mean()
    df["fastK_20ma"]=df["fastK"].rolling(20).mean()
    df["fastK_50ma"]=df["fastK"].rolling(50).mean()
    
    df['return'] = df['close'].pct_change()
    df['volatility'] = df['return'].rolling(window=5).std()
    df['rsi'] = RSIIndicator(df['close']).rsi()
    
    
    #df_long=calculate_indicators_hist(symbol) #.iloc[-1]
    # Merge DataFrames using cross join
    #merged_df = df.merge(df_long, how='cross')
    
    #print(merged_df.iloc[-1])
    time.sleep(0.1)
    return df #.dropna()

# Calculate indicators
def calculate_indicators_v1(df):
    df["50_MA"] = df["close"].rolling(window=50).mean()
    df["200_MA"] = df["close"].rolling(window=200).mean()
    df["Pivot_Point"] = (df["high"].rolling(5).mean() + df["low"].rolling(5).mean() + df["close"].rolling(5).mean()) / 3
    df["Prev_Close"] = df["close"].shift(1)
    return df

def is_stock_owned(symbol):
    """
    Checks if the given stock symbol is already owned in the Alpaca portfolio.
    Returns True if the stock is owned, False otherwise.
    """
    positions = trade_client.get_all_positions()
    
    for position in positions:
        if position.symbol == symbol and float(position.qty) > 0:
            return True  # Stock is already owned
    
    return False  # Stock is not owned

def get_today_trades():
    """
    Fetches only buy and sell trades for today.
    """
    today = date.today().isoformat()
    
    request_params = GetActivitiesRequest(date=today, activity_types=[ActivityType.FILL])  
    trades = trading_client.get_activities(request_params)

    return trades

def get_today_orders_data():
    """
    Fetches all completed buy and sell orders for today.
    """
    # Define EST timezone
    EST = pytz.timezone("US/Eastern")
    #today = date.today().isoformat()
    today = datetime.now(EST).date()
    
    req = GetOrdersRequest(
        status = "all",
        after=f"{today}T00:00:00Z",  # Start of today (UTC)
        #until=f"{today}T23:59:59Z",  # End of today (UTC)
        direction="desc",
        limit=2000 # Oldest first
        
    )
    orders = trade_client.get_orders(req)
    
    # Filter out only filled orders
    filled_orders = [order for order in orders if order.status == "filled"]
    # Filter only filled orders and convert timestamps to EST
    # Convert orders to DataFrame
    order_data = pd.DataFrame([
        {
            "symbol": order.symbol,
            "side": order.side.value,  # "buy" or "sell"
            "qty": float(order.filled_qty),
            "price": float(order.filled_avg_price),
            "timestamp": order.filled_at.astimezone(EST)
        }
        for order in filled_orders
    ])
    return order_data  # Returns a list of filled orders

def get_all_orders_data(dateval=None,side="BUY",direction="asc",symbol=None):
    """
    Fetches all completed buy and sell orders for today.
    """
    # Define EST timezone
    EST = pytz.timezone("US/Eastern")
    #today = date.today().isoformat()
    today = datetime.now(EST).date()
    print(today)
    dateval1="2025-02-01"
    if side=="BUY":
      req = GetOrdersRequest(
          symbols=[symbol],
          status = "all",
          after=f"{dateval}T00:00:00Z",  # Start of today (UTC)
          #until=f"{today}T23:59:59Z",  # End of today (UTC)
          side=OrderSide.BUY,
          direction=direction,
          limit=1500 # Oldest first
          
      )
    else:
      req = GetOrdersRequest(
          symbols=[symbol],
          status = "all",
          after=f"{dateval}T00:00:00Z",  # Start of today (UTC)
          #until=f"{dateval}T23:59:59Z",  # End of today (UTC)
          side=OrderSide.SELL,
          direction=direction,
          limit=1500 # Oldest first
          
      )
    orders = trade_client.get_orders(req)
    
    # Filter out only filled orders
    filled_orders = [order for order in orders if order.status == "filled"]
    # Filter only filled orders and convert timestamps to EST
    # Convert orders to DataFrame
    order_data = pd.DataFrame([
        {
            "symbol": order.symbol,
            "side": order.side.value,  # "buy" or "sell"
            "qty": float(order.filled_qty),
            "price": float(order.filled_avg_price),
            "timestamp": order.filled_at.astimezone(EST)
        }
        for order in filled_orders
    ])
    return order_data  # Returns a list of filled orders


def get_today_orders():
    """
    Fetches all completed buy and sell orders for today.
    """
    # Define EST timezone
    EST = pytz.timezone("US/Eastern")
    #today = date.today().isoformat()
    today = datetime.now(EST).date()
    
    req = GetOrdersRequest(
        status = "all",
        after=f"{today}T00:00:00Z",  # Start of today (UTC)
        #until=f"{today}T23:59:59Z",  # End of today (UTC)
        direction="desc",
        limit=1500 # Oldest first
        
    )
    orders = trade_client.get_orders(req)
    
    # Filter out only filled orders
    filled_orders = [order for order in orders if order.status == "filled"]
    # Filter only filled orders and convert timestamps to EST
    
    return filled_orders  # Returns a list of filled orders
    #return pd.DataFrame(filled_orders)  # Return as a DataFrame

def calculate_trade_performance3():
    """
    Calculates profit/loss for all buy and sell orders per stock.
    Uses the average buy price and average sell price for each stock.
    """
    orders = get_today_orders()
    
    # Convert orders to DataFrame
    order_data = pd.DataFrame([
        {
            "symbol": order.symbol,
            "side": order.side.value,  # "buy" or "sell"
            "qty": float(order.filled_qty),
            "price": float(order.filled_avg_price),
            "timestamp": order.filled_at.astimezone(EST)
        }
        for order in orders
    ])

    if order_data.empty:
        print("⚠️ No trades found for today.")
        return None

    # Sort orders by timestamp
    order_data = order_data.sort_values(by="timestamp")
    print(order_data)
    #return order_data
    # Separate buys and sells
    buys = order_data[order_data["side"] == "buy"]
    sells = order_data[order_data["side"] == "sell"]

    # Find symbols that have at least one buy order
    #valid_symbols = set(buys["symbol","qty"].unique())

    # Keep only sell orders that have a corresponding buy order
    #valid_sells = sells[sells["symbol"].isin(valid_symbols)]
    # Create a set of (symbol, quantity) pairs from buys
    valid_symbol_qty_pairs = set(zip(buys["symbol"], buys["qty"]))

    # Keep only sell orders that match a buy order in symbol AND quantity
    valid_sells = sells[sells.apply(lambda row: (row["symbol"], row["qty"]) in valid_symbol_qty_pairs, axis=1)]

    
    # Combine valid buys and sells
    #order_data = pd.concat([buys, valid_sells])
    
    # Separate buys and sells
    buys = order_data[order_data["side"] == "buy"]
    sells = order_data[order_data["side"] == "sell"]

    # Count buy quantities per (symbol, qty)
    buy_counts = buys.groupby(["symbol", "qty"]).size().reset_index(name="buy_count")

    # Merge sell orders with buy counts
    sells = sells.merge(buy_counts, on=["symbol", "qty"], how="left")

    # Remove sells that do not have a matching buy count
    valid_sells = sells[sells["buy_count"].notna()].drop(columns=["buy_count"])

    # Combine valid buys and valid sells
    valid_orders = pd.concat([buys, valid_sells])
    order_data=valid_orders
    print(order_data)
    # Group by stock symbol and side, then calculate the average price
    avg_prices = order_data.groupby(["symbol", "side"])["price"].mean().unstack()

    # Ensure both buy and sell prices exist for calculation
    avg_prices = avg_prices.dropna()
    print(avg_prices)
    # Calculate Profit/Loss Percentage
    avg_prices["profit_loss_%"] = ((avg_prices["sell"] - avg_prices["buy"]) / avg_prices["buy"]) * 100

    return avg_prices[["buy", "sell", "profit_loss_%"]]
  
def calculate_trade_performance2():
    """
    Matches sell trades to any buy orders with the same quantity.
    Calculates profit/loss percentage and ignores unsold buys.
    """
    orders = get_today_orders()
    
    # Convert orders to DataFrame
    order_data = pd.DataFrame([
        {
            "symbol": order.symbol,
            "side": order.side.value,  # "buy" or "sell"
            "qty": float(order.filled_qty),
            "price": float(order.filled_avg_price),
            "timestamp": order.filled_at
        }
        for order in orders
    ])

    if order_data.empty:
        print("⚠️ No trades found for today.")
        return None

    # Sort orders by timestamp
    order_data = order_data.sort_values(by="timestamp")

    # Separate buys and sells
    buys = order_data[order_data["side"] == "buy"]
    sells = order_data[order_data["side"] == "sell"]

    # Match sells to any available buy with the same quantity
    results = []
    used_buys = set()  # Keep track of used buy orders

    for _, sell in sells.iterrows():
        match = buys[(buys["symbol"] == sell["symbol"]) & 
                     (~buys.index.isin(used_buys))]  # Ignore used buys

        if not match.empty:
            matched_buy = match.iloc[0]
            buy_price = matched_buy["price"]
            sell_price = sell["price"]

            # Mark the buy order as used
            used_buys.add(matched_buy.name)

            # Calculate Profit/Loss Percentage
            pnl_pct = ((sell_price - buy_price) / buy_price) * 100
            pnl_price = ((sell_price*sell["qty"] - buy_price*sell["qty"]))
            results.append({
                "symbol": sell["symbol"],
                "qty":sell["qty"],
                "buy_price": buy_price,
                "sell_price": sell_price,
                "proft_loss_$":round(pnl_price,2),
                "profit_loss_%": round(pnl_pct, 2)
            })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    return results_df
  
def calculate_trade_performance():
    """
    Matches buy and sell trades for the same stock and calculates profit/loss percentage.
    Ignores buy trades that have not been sold.
    """
    orders = get_today_orders()
    #print(orders)
     # Convert orders to DataFrame
    trade_data = pd.DataFrame([
        {
            "symbol": order.symbol,
            "side": order.side.value,  # "buy" or "sell"
            "qty": float(order.filled_qty),
            "price": float(order.filled_avg_price),
            "timestamp": order.filled_at
        }
        for order in orders
    ])

    
    if trade_data.empty:
        print("⚠️ No trades found for today.")
        return None

    # Sort trades by timestamp
    trade_data = trade_data.sort_values(by="timestamp")

    # Separate buys and sells
    buys = trade_data[trade_data["side"] == "buy"]
    sells = trade_data[trade_data["side"] == "sell"]

    # Match sells to the earliest available buys (FIFO method)
    results = []
    for _, sell in sells.iterrows():
        match = buys[(buys["symbol"] == sell["symbol"]) & (buys["qty"] == sell["qty"])]

        if not match.empty:
            buy_price = match.iloc[0]["price"]
            sell_price = sell["price"]
            
            # Calculate Profit/Loss Percentage
            pnl_pct = ((sell_price - buy_price) / buy_price) * 100
            pnl_price = ((sell_price*sell["qty"] - buy_price*sell["qty"]))
            results.append({
                "symbol": sell["symbol"],
                "qty":sell["qty"],
                "buy_price": buy_price,
                "sell_price": sell_price,
                "proft_loss_$":round(pnl_price,2),
                "profit_loss_%": round(pnl_pct, 2)
            })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    return results_df


def get_signalvorig(latest,prev):
    #print(latest["close"])
    
    buy_signal = (
       (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] 
       and latest["close"]<prev["20_MA"] and latest["close"]<prev["50_MA"] and latest["RSI_14"]>35 and latest["fastK"]<40 and prev["RSI_14"]<latest["RSI_14"] and latest["fastK"]>35 and latest["fastK"]<40 and latest["fastK"]>prev["fastK"] and latest["ROC_10"]<(0.5) and latest["PercentChange"]<0.05) or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["close"]<prev["20_MA"] and latest["close"]<prev["50_MA"] and latest["RSI_14"]<30 and prev["fastK"]<30 and latest["fastK"]<15 and latest["ROC_10"]<(0.5) and latest["PercentChange"]<0) or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["close"]<prev["20_MA"] and latest["close"]<prev["50_MA"] and latest["RSI_14"]<30 and prev["fastK"]<30 and prev["RSI_14"]<latest["RSI_14"] and latest["fastK"]<15 and latest["fastK"]>prev["fastK"] and prev["fastK"]<5 and latest["ROC_10"]<(0.5) and latest["PercentChange"]<0.05) or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["close"]<prev["20_MA"] and latest["close"]<prev["50_MA"] and latest["RSI_14"]<35 and latest["fastK"]<3 and latest["ROC_10"]<(0.5) and latest["PercentChange"]<0.5) or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["close"]<prev["20_MA"] and latest["close"]<prev["50_MA"] and latest["close"]>prev["120_MA"] and latest["RSI_14"]<40 and latest["fastK"]<20 and latest["ROC_10"]<(0) and latest["PercentChange"]<0.5 and latest["fastDfastK"]>2) or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["ROC_10"]<(-5) and latest["PercentChange"]<(-5) and latest["close"]<latest["5_MA"] and latest["close"]>latest["20_MA"] and latest["close"]<latest["50_MA"]) or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["ROC_10"]<(-5) and latest["PercentChange"]<(-5) and latest["close"]<latest["5_MA"] and latest["fastK"]<5 and latest["fastDfastK"]>3 and latest["close"]<latest["50_MA"]) or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["ROC_10"]>1 and latest["PercentChange"]<(-10) and latest["close"]<latest["5_MA"] and latest["fastK"]<10 and latest["fastDfastK"]>3 and latest["close"]<latest["50_MA"]) or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["ROC_10"]<(-5) and prev["ROC_10"]>5 and latest["PercentChange"]<(-5) and latest["close"]<latest["5_MA"] and latest["fastK"]<15 and latest["fastDfastK"]>3 and latest["close"]<latest["50_MA"]) or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["ROC_10"]>prev["ROC_10"] and prev["ROC_10"]<(-1) and latest["PercentChange"]<(0.5) and prev["PercentChange"]<(-1) and latest["PercentChange"]>prev["PercentChange"] and latest["close"]<latest["5_MA"] and prev["fastK"]<15 and latest["fastDfastK"]>1 and prev["fastK"]<latest["fastK"] and latest["fastK"]<20 and prev["RSI_14"]<30 and latest["RSI_14"]>prev["RSI_14"] and latest["RSI_14"]<35) or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["PercentChange"]<(0.5) and prev["PercentChange"]<(0.5) and latest["fastDfastK"]>1 and prev["fastDfastK"]>5 and latest["fastK"]>prev["fastK"] and latest["fastK"]>10 and prev["fastK"]<10 and latest["fastK"]<20 and latest["RSI_14"]<40) or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["PercentChange"]<(-2) and latest["fastDfastK"]>1.25 and latest["fastK"]<30 and latest["PercentChange"]>(-25))
            
    )
    
    sell_signal = (
        (latest["close"]>prev["Pivot_Point"] and latest["close"]>latest["Pivot_Point"] and latest["RSI_14"]>50 and latest["fastK"]>50 and latest["fastK"]<prev["fastK"] and latest["fastK"]>latest["fastD"] and latest["ROC_10"]<prev["ROC_10"] and latest["ROC_10"]>0 and latest["close"] > latest["EMA_20"] and latest["close"] > prev["EMA_20"] and prev["EMA_20"] >= prev["SMA_20"]) or (latest["close"]>prev["Pivot_Point"] and latest["close"]>latest["Pivot_Point"] and latest["RSI_14"]>50 and latest["fastK"]>50 and latest["fastK"]>prev["fastK"] and latest["fastK"]>latest["fastD"] and latest["fastD"]>prev["fastD"] and latest["ROC_10"]>prev["ROC_10"] and latest["ROC_10"]>1 and latest["close"] > latest["EMA_20"] and latest["close"] > prev["EMA_20"] and prev["EMA_20"] >= prev["SMA_20"]) or (latest["close"]>prev["close"] and latest["ROC_10"]>prev["ROC_10"] and latest["PercentChange"]>(5))
    )

    return "BUY" if buy_signal else "SELL" if sell_signal else "HOLD"


def get_signalvorig1(latest,prev):
    #print(latest["close"])
    
    buy_signal = (
       (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] 
       and latest["close"]<prev["20_MA"] and latest["close"]<prev["50_MA"] and latest["RSI_14"]>35 
       and latest["fastK"]<40 and prev["RSI_14"]<latest["RSI_14"] and latest["fastK"]>35 
       and latest["fastK"]<40 and latest["fastK"]>prev["fastK"] and latest["ROC_10"]<(0.5) 
       and latest["PercentChange"]<0.05) or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["close"]<prev["20_MA"] and latest["close"]<prev["50_MA"] and latest["RSI_14"]<30 and prev["fastK"]<30 and latest["fastK"]<15 and latest["ROC_10"]<(0.5) and latest["PercentChange"]<0) or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["close"]<prev["20_MA"] and latest["close"]<prev["50_MA"] and latest["RSI_14"]<30 and prev["fastK"]<30 and prev["RSI_14"]<latest["RSI_14"] and latest["fastK"]<15 and latest["fastK"]>prev["fastK"] and prev["fastK"]<5 and latest["ROC_10"]<(0.5) and latest["PercentChange"]<0.05) or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["close"]<prev["20_MA"] and latest["close"]<prev["50_MA"] and latest["RSI_14"]<35 and latest["fastK"]<3 and latest["ROC_10"]<(0.5) and latest["PercentChange"]<0.5) or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["close"]<prev["20_MA"] and latest["close"]<prev["50_MA"] and latest["close"]>prev["120_MA"] and latest["RSI_14"]<40 and latest["fastK"]<20 and latest["ROC_10"]<(0) and latest["PercentChange"]<0.5 and latest["fastDfastK"]>2) or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["ROC_10"]<(-5) and latest["PercentChange"]<(-5) and latest["close"]<latest["5_MA"] and latest["close"]>latest["20_MA"] and latest["close"]<latest["50_MA"]) or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["ROC_10"]<(-5) and latest["PercentChange"]<(-5) and latest["close"]<latest["5_MA"] and latest["fastK"]<5 and latest["fastDfastK"]>3 and latest["close"]<latest["50_MA"]) or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["ROC_10"]>1 and latest["PercentChange"]<(-10) and latest["close"]<latest["5_MA"] and latest["fastK"]<10 and latest["fastDfastK"]>3 and latest["close"]<latest["50_MA"]) or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["ROC_10"]<(-5) and prev["ROC_10"]>5 and latest["PercentChange"]<(-5) and latest["close"]<latest["5_MA"] and latest["fastK"]<15 and latest["fastDfastK"]>3 and latest["close"]<latest["50_MA"]) or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["ROC_10"]>prev["ROC_10"] and prev["ROC_10"]<(-1) and latest["PercentChange"]<(0.5) and prev["PercentChange"]<(-1) and latest["PercentChange"]>prev["PercentChange"] and latest["close"]<latest["5_MA"] and prev["fastK"]<15 and latest["fastDfastK"]>1 and prev["fastK"]<latest["fastK"] and latest["fastK"]<20 and prev["RSI_14"]<30 and latest["RSI_14"]>prev["RSI_14"] and latest["RSI_14"]<35) or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["PercentChange"]<(0.5) and prev["PercentChange"]<(0.5) and latest["fastDfastK"]>1 and prev["fastDfastK"]>5 and latest["fastK"]>prev["fastK"] and latest["fastK"]>10 and prev["fastK"]<10 and latest["fastK"]<20 and latest["RSI_14"]<40) or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["PercentChange"]<(-2) and latest["fastDfastK"]>1.25 and latest["fastK"]<30 and latest["PercentChange"]>(-25))
            
    )
    
    sell_signal = (
        (latest["close"]>prev["Pivot_Point"] and latest["close"]>latest["Pivot_Point"] and latest["RSI_14"]>50 and latest["fastK"]>50 and latest["fastK"]<prev["fastK"] and latest["fastK"]>latest["fastD"] and latest["ROC_10"]<prev["ROC_10"] and latest["ROC_10"]>0 and latest["close"] > latest["EMA_20"] and latest["close"] > prev["EMA_20"] and prev["EMA_20"] >= prev["SMA_20"]) or (latest["close"]>prev["Pivot_Point"] and latest["close"]>latest["Pivot_Point"] and latest["RSI_14"]>50 and latest["fastK"]>50 and latest["fastK"]>prev["fastK"] and latest["fastK"]>latest["fastD"] and latest["fastD"]>prev["fastD"] and latest["ROC_10"]>prev["ROC_10"] and latest["ROC_10"]>1 and latest["close"] > latest["EMA_20"] and latest["close"] > prev["EMA_20"] and prev["EMA_20"] >= prev["SMA_20"]) or (latest["close"]>prev["close"] and latest["ROC_10"]>prev["ROC_10"] and latest["PercentChange"]>(5))
    )

    return "BUY" if buy_signal else "SELL" if sell_signal else "HOLD"


def get_signalvorig2(latest,prev):
    rsi_buy_thresh=15
    rsi_sell_thresh1=70
    rsi_sell_thresh2=90
    #and latest["MACD_histogram"]>0 and latest['close']<=latest['lower_band']
    # and latest['close']>=latest['upper_band']
    #latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["close"]<prev["20_MA"] 
    #and latest["close"]<prev["50_MA"] and and latest["PercentChange"]<(0) and 
    
    buy_signal = (
    (latest["RSI_14"]>rsi_buy_thresh and latest["RSI_14"]<70 and prev["fastK"]<20 and latest["fastK"]>latest["fastD"]
    and latest["ROC_10"]<(0) and latest["close"]<prev["20_MA"] and latest["close"]<prev["5_MA"])) 
    
    sell_signal = (
      (latest["RSI_14"]>rsi_sell_thresh1 and prev["RSI_14"]<rsi_sell_thresh2 and latest["fastD"]>latest["fastK"] and latest["close"]>prev["close"] 
      and latest["RSI_14"]>prev["RSI_14"] and latest["ROC_10"]>prev["ROC_10"]
      and latest["PercentChange"]>0 and latest["close"] > latest["EMA_20"] and latest["close"] > prev["EMA_20"] 
      and prev["EMA_20"] >= prev["SMA_20"] and latest["close"]>prev["Pivot_Point"]) 
          )

    return "BUY" if buy_signal else "SELL" if sell_signal else "HOLD"
  


def get_signalv3(latest,prev):
    print(latest["close"])
    print("THIS")
    rsi_buy_thresh=35
    rsi_sell_thresh=70
    #_buy_thresh=20
    buy_signal = (
      # and df["ROC_10_last3"]>0 
   (latest["ROC_10_last3"]>-1) and ((latest["RSI_14"]<rsi_sell_thresh and latest["RSI_14"]>50 and prev["ROC_10"]>(0) and latest["ROC_10"]>(0) and latest["ROC_10"]<(1) and latest["fastDfastK"]>0.75 and latest["PercentChange"]<(0.3)) or (latest["ROC_10_last3"]>-1) and (latest["RSI_14"]<rsi_sell_thresh and latest["RSI_14"]>rsi_buy_thresh and latest["ROC_10"]>(1) and latest["fastDfastK"]>1.2 and latest["PercentChange"]<(-3)) or (latest["RSI_14"]<rsi_buy_thresh and latest["RSI_14"]>prev["RSI_14"] and latest["close"]<prev["Pivot_Point"] and latest["fastDfastK"]>3) or (latest["RSI_14"]<rsi_buy_thresh 
    and latest["RSI_14"]>prev["RSI_14"] and latest["fastDfastK"]>1) or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["close"]<prev["20_MA"] 
    and latest["close"]<prev["50_MA"] and latest["RSI_14"]<rsi_buy_thresh and prev["fastK"]<30 and latest["fastK"]<15 
    and latest["ROC_10"]<(0.5) and latest["PercentChange"]<0) 
    or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["close"]<prev["20_MA"] 
    and latest["close"]<prev["50_MA"] and latest["RSI_14"]<rsi_buy_thresh and prev["fastK"]<30 and prev["RSI_14"]<latest["RSI_14"] 
    and latest["fastK"]<15 and latest["fastK"]>prev["fastK"] and prev["fastK"]<5 and latest["ROC_10"]<(0.5) and latest["PercentChange"]<0.05) 
    or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["close"]<prev["20_MA"] 
    and latest["close"]<prev["50_MA"] and latest["RSI_14"]<rsi_buy_thresh and latest["fastK"]<3 and latest["ROC_10"]<(0.5) and latest["PercentChange"]<0.5) 
    or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["close"]<prev["20_MA"]
    and latest["close"]<prev["50_MA"] and latest["close"]>prev["120_MA"] and latest["RSI_14"]<rsi_buy_thresh and latest["fastK"]<20 and latest["ROC_10"]<(0) 
    and latest["PercentChange"]<0.5 and latest["fastDfastK"]>2) or 
    (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["RSI_14"]<rsi_buy_thresh and latest["close"]<prev["5_MA"] and latest["ROC_10"]<(-5) 
    and latest["PercentChange"]<(-5) and latest["close"]<latest["5_MA"] and latest["close"]>latest["20_MA"] and latest["close"]<latest["50_MA"])
    or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["ROC_10"]<(-5) 
    and latest["PercentChange"]<(-5) and latest["close"]<latest["5_MA"] and latest["fastK"]<5 and latest["RSI_14"]<rsi_buy_thresh and latest["fastDfastK"]>3 and latest["close"]<latest["50_MA"])
    or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["ROC_10"]>1 and latest["PercentChange"]<(-10)
    and latest["close"]<latest["5_MA"] and latest["fastK"]<10 and latest["fastDfastK"]>3 and latest["close"]<latest["50_MA"] and latest["RSI_14"]<rsi_buy_thresh) or 
    (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["ROC_10"]<(-5) and prev["ROC_10"]>5 
    and latest["PercentChange"]<(-5) and latest["close"]<latest["5_MA"] and latest["fastK"]<15 and latest["RSI_14"]<rsi_buy_thresh and latest["fastDfastK"]>3 and latest["close"]<latest["50_MA"])
    or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["ROC_10"]>prev["ROC_10"] 
    and prev["ROC_10"]<(-1) and latest["PercentChange"]<(0.5) and prev["PercentChange"]<(-1) and latest["PercentChange"]>prev["PercentChange"] 
    and latest["close"]<latest["5_MA"] and prev["fastK"]<15 and latest["fastDfastK"]>1 and prev["fastK"]<latest["fastK"] and latest["fastK"]<20 and prev["RSI_14"]<rsi_buy_thresh
    and latest["RSI_14"]>prev["RSI_14"] and latest["RSI_14"]<rsi_buy_thresh) or 
    (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["PercentChange"]<(0.5)
    and prev["PercentChange"]<(0.5) and latest["fastDfastK"]>1 and prev["fastDfastK"]>5 and latest["fastK"]>prev["fastK"] and latest["fastK"]>10 and prev["fastK"]<10 
    and latest["fastK"]<20 and latest["RSI_14"]<rsi_buy_thresh) or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] 
    and latest["PercentChange"]<(-2) and latest["fastDfastK"]>1.25 and latest["fastK"]<30 and latest["PercentChange"]>(-25)))
            
    )
    
    if latest["ROC_10_last3"]<(-2) and prev["ROC_10_last3"]<(-2) and latest["fastDfastK"]<1:
      buy_signal=False
      
    #(latest["RSI_14"]>80 and latest["fastK"]>80) or  
    sell_signal = (
      (latest["RSI_14"]>rsi_sell_thresh and latest["RSI_14"]<prev["RSI_14"]) or (latest["close"]>prev["close"] and latest["RSI_14"]>prev["RSI_14"] and latest["ROC_10"]>prev["ROC_10"] and prev["ROC_10"]<(-10) and latest["PercentChange"]>1 and latest["fastK"]>(20) and prev["fastK"]<20) or (latest["close"]>prev["close"] and latest["RSI_14"]>prev["RSI_14"] and latest["ROC_10"]>prev["ROC_10"] and latest["PercentChange"]>3 and prev["PercentChange"]<(-2)) or (latest["close"]>prev["Pivot_Point"] and latest["close"]>latest["Pivot_Point"] and latest["RSI_14"]>50 and latest["fastK"]>50 and latest["fastK"]<prev["fastK"] and latest["fastK"]>latest["fastD"] and latest["ROC_10"]<prev["ROC_10"] and latest["ROC_10"]>0 and latest["close"] > latest["EMA_20"] and latest["close"] > prev["EMA_20"] and prev["EMA_20"] >= prev["SMA_20"]) or (latest["close"]>prev["Pivot_Point"] and latest["close"]>latest["Pivot_Point"] and latest["RSI_14"]>50 and latest["fastK"]>50 and latest["fastK"]>prev["fastK"] and latest["fastK"]>latest["fastD"] and latest["fastD"]>prev["fastD"] and latest["ROC_10"]>prev["ROC_10"] and latest["ROC_10"]>1 and latest["close"] > latest["EMA_20"] and latest["close"] > prev["EMA_20"] and prev["EMA_20"] >= prev["SMA_20"]) or (latest["close"]>prev["close"] and latest["ROC_10"]>prev["ROC_10"] and latest["PercentChange"]>(5))
    )

    return "BUY" if buy_signal else "SELL" if sell_signal else "HOLD"
    

def get_signalv2(latest,prev):
    print(latest["close"])
    print("THIS")
    rsi_buy_thresh=35
    rsi_sell_thresh=70
    #_buy_thresh=20
    buy_signal = (
      # and df["ROC_10_last3"]>0 
   (latest["ROC_10_last3"]>-1) and ((latest["RSI_14"]<rsi_sell_thresh and latest["RSI_14"]>50 and prev["ROC_10"]<(0) 
   and latest["ROC_10"]>(0) and latest["ROC_10"]<(1) and latest["fastDfastK"]>0.75 and latest["fastK"]<80 and latest["PercentChange"]<(0.3)) 
   or (latest["ROC_10_last3"]>-1) and (latest["RSI_14"]<rsi_sell_thresh and latest["RSI_14"]>rsi_buy_thresh and latest["ROC_10"]>(1) and latest["fastDfastK"]>1.2 and latest["PercentChange"]<(-3)) or (latest["RSI_14"]<rsi_buy_thresh and latest["RSI_14"]>prev["RSI_14"] and latest["close"]<prev["Pivot_Point"] and latest["fastDfastK"]>3) or (latest["RSI_14"]<rsi_buy_thresh 
    and latest["RSI_14"]>prev["RSI_14"] and latest["fastDfastK"]>1) or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["close"]<prev["20_MA"] 
    and latest["close"]<prev["50_MA"] and latest["RSI_14"]<rsi_buy_thresh and prev["fastK"]<30 and latest["fastK"]<15 and latest["ROC_10"]<(0.5) and latest["PercentChange"]<0) 
    or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["close"]<prev["20_MA"] 
    and latest["close"]<prev["50_MA"] and latest["RSI_14"]<rsi_buy_thresh and prev["fastK"]<30 and prev["RSI_14"]<latest["RSI_14"] 
    and latest["fastK"]<15 and latest["fastK"]>prev["fastK"] and prev["fastK"]<5 and latest["ROC_10"]<(0.5) and latest["PercentChange"]<0.05) 
    or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["close"]<prev["20_MA"] 
    and latest["close"]<prev["50_MA"] and latest["RSI_14"]<rsi_buy_thresh and latest["fastK"]<3 and latest["ROC_10"]<(0.5) and latest["PercentChange"]<0.5) 
    or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["close"]<prev["20_MA"]
    and latest["close"]<prev["50_MA"] and latest["close"]>prev["120_MA"] and latest["RSI_14"]<rsi_buy_thresh and latest["fastK"]<20 and latest["ROC_10"]<(0) 
    and latest["PercentChange"]<0.5 and latest["fastDfastK"]>2) or 
    (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["RSI_14"]<rsi_buy_thresh and latest["close"]<prev["5_MA"] and latest["ROC_10"]<(-5) 
    and latest["PercentChange"]<(-5) and latest["close"]<latest["5_MA"] and latest["close"]>latest["20_MA"] and latest["close"]<latest["50_MA"])
    or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["ROC_10"]<(-5) 
    and latest["PercentChange"]<(-5) and latest["close"]<latest["5_MA"] and latest["fastK"]<5 and latest["RSI_14"]<rsi_buy_thresh and latest["fastDfastK"]>3 and latest["close"]<latest["50_MA"])
    or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["ROC_10"]>1 and latest["PercentChange"]<(-10)
    and latest["close"]<latest["5_MA"] and latest["fastK"]<10 and latest["fastDfastK"]>3 and latest["close"]<latest["50_MA"] and latest["RSI_14"]<rsi_buy_thresh) or 
    (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["ROC_10"]<(-5) and prev["ROC_10"]>5 
    and latest["PercentChange"]<(-5) and latest["close"]<latest["5_MA"] and latest["fastK"]<15 and latest["RSI_14"]<rsi_buy_thresh and latest["fastDfastK"]>3 and latest["close"]<latest["50_MA"])
    or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["ROC_10"]>prev["ROC_10"] 
    and prev["ROC_10"]<(-1) and latest["PercentChange"]<(0.5) and prev["PercentChange"]<(-1) and latest["PercentChange"]>prev["PercentChange"] 
    and latest["close"]<latest["5_MA"] and prev["fastK"]<15 and latest["fastDfastK"]>1 and prev["fastK"]<latest["fastK"] and latest["fastK"]<20 and prev["RSI_14"]<rsi_buy_thresh
    and latest["RSI_14"]>prev["RSI_14"] and latest["RSI_14"]<rsi_buy_thresh) or 
    (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["PercentChange"]<(0.5)
    and prev["PercentChange"]<(0.5) and latest["fastDfastK"]>1 and prev["fastDfastK"]>5 and latest["fastK"]>prev["fastK"] and latest["fastK"]>10 and prev["fastK"]<10 
    and latest["fastK"]<20 and latest["RSI_14"]<rsi_buy_thresh) or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] 
    and latest["PercentChange"]<(-2) and latest["fastDfastK"]>1.25 and latest["fastK"]<30 and latest["PercentChange"]>(-25)))
            
    )
    
    if latest["ROC_10_last3"]<(-2) and prev["ROC_10_last3"]<(-2) and latest["fastDfastK"]<1:
      buy_signal=False
      
    #(latest["RSI_14"]>80 and latest["fastK"]>80) or  
    sell_signal = (
      (latest["RSI_14"]>rsi_sell_thresh and latest["RSI_14"]<prev["RSI_14"]) or (latest["close"]>prev["close"] and latest["RSI_14"]>prev["RSI_14"] and latest["ROC_10"]>prev["ROC_10"] and prev["ROC_10"]<(-10) and latest["PercentChange"]>1 and latest["fastK"]>(20) and prev["fastK"]<20) or (latest["close"]>prev["close"] and latest["RSI_14"]>prev["RSI_14"] and latest["ROC_10"]>prev["ROC_10"] and latest["PercentChange"]>3 and prev["PercentChange"]<(-2)) or (latest["close"]>prev["Pivot_Point"] and latest["close"]>latest["Pivot_Point"] and latest["RSI_14"]>50 and latest["fastK"]>50 and latest["fastK"]<prev["fastK"] and latest["fastK"]>latest["fastD"] and latest["ROC_10"]<prev["ROC_10"] and latest["ROC_10"]>0 and latest["close"] > latest["EMA_20"] and latest["close"] > prev["EMA_20"] and prev["EMA_20"] >= prev["SMA_20"]) or (latest["close"]>prev["Pivot_Point"] and latest["close"]>latest["Pivot_Point"] and latest["RSI_14"]>50 and latest["fastK"]>50 and latest["fastK"]>prev["fastK"] and latest["fastK"]>latest["fastD"] and latest["fastD"]>prev["fastD"] and latest["ROC_10"]>prev["ROC_10"] and latest["ROC_10"]>1 and latest["close"] > latest["EMA_20"] and latest["close"] > prev["EMA_20"] and prev["EMA_20"] >= prev["SMA_20"]) or (latest["close"]>prev["close"] and latest["ROC_10"]>prev["ROC_10"] and latest["PercentChange"]>(5))
    )

    return "BUY" if buy_signal else "SELL" if sell_signal else "HOLD"

#03252025
def get_signal_debug(latest,prev,indicators=None):
  
    last_row=latest
    # ATR thresholds to adjust risk
    high_volatility = last_row['ATR'] > last_row['ATR_MA'] #.mean()  # High volatility if ATR is above the average ATR
    low_volatility = last_row['ATR'] < last_row['ATR_MA'] #.mean()   # Low volatility if ATR is below the average ATR
    
    # Pivot Point Levels
    price_above_r1 = last_row['close'] > last_row['R1']  # Price above R1 (bullish signal)
    price_below_s1 = last_row['close'] < last_row['S1']  # Price below S1 (bearish signal)
    price_below_r1 = last_row['close'] < last_row['R1'] 
    price_above_s1 = last_row['close'] > last_row['S1']
    
    print(latest["close"])
    print(latest["vwap"])
    print("Resistance:")
    print(latest["R2"])
    print(latest["R1"])
    print(latest["S1"])
    print(latest["S2"])
    print(latest["ATR"])
    print(latest["ATR_MA"])
    print(latest["volume"])
    print(latest["volume_MA"])
    print(latest["RSI_14"])
    print(latest["ROC_10"])
    print(latest["fastK"])
    print(prev["fastK"])
    print(latest["fastD"])
    print(prev["5_MA"])
    print(latest["fastK_20ma"])
    print(prev["fastK_20ma"])
    print(latest["fastDfastK"])
    print(latest["ROC_10_last10"])
    
    #print(indicators)
    print(price_above_r1)
    print(high_volatility)
    print(low_volatility)
    print(indicators["l5_MA"])
    print(indicators[:])
    #and latest["ROC_10_last10"]>0 
    buy_signal = (
      (price_below_r1 and price_above_s1 and latest["close"]<latest["5_MA"] and latest["RSI_14"]<40 and latest["ROC_10"]<(-3) and latest["fastDfastK"]>1.2
        and latest["fastK_20ma"]>prev["fastK_20ma"]and prev["fastK_20ma"]<40 and latest["ROC_10_last10"]>0 ) or
        (price_below_r1 and price_above_s1 and latest["close"]<latest["5_MA"] and latest["RSI_14"]>20 and latest["RSI_14"]<40 and latest["ROC_10"]<(0.5)
        and latest["fastDfastK"]>1.2
        and latest["fastK_20ma"]<prev["fastK_20ma"] and latest["ROC_10_last10"]>0 and latest["close"]<indicators["l5_MA"] and indicators["lMACD_histogram"]<0
        and indicators["lfastK"]<15 and indicators["lfastK"]>15 and indicators["lRSI_14"]<40 and indicators["close"]<indicators["open"]) or 
        (price_below_r1 and price_above_s1 and latest["close"]<latest["5_MA"] and latest["RSI_14"]<40 and latest["ROC_10"]<(0.5)
        and latest["fastDfastK"]>1.1
        and latest["ROC_10_last10"]>0 and latest["close"]<indicators["l5_MA"] and indicators["lMACD_histogram"]<0
        and indicators["lfastK"]<15 and indicators["lfastD"]>10 and indicators["lRSI_14"]<40 and latest["close"]<indicators["open"]) or
        (price_below_r1 and price_above_s1 and latest["close"]<latest["vwap"] and latest["close"]<latest["5_MA"] 
        and latest["close"]<latest["20_MA"] and latest["RSI_14"]>30 and latest["RSI_14"]<40 and latest["ROC_10"]<(0.5)
        and latest["fastDfastK"]>2 and latest["fastK"]<15 and latest["fastD"]>25 and prev["fastK"]>30 and latest["close"]<latest["pivot_point"]
        and latest["ROC_10_last10"]>1 and latest["RSI_14_3ma"]>40 and latest["RSI_14_3ma"]<60 and latest["RSI_14_20ma"]>40 and latest["RSI_14_20ma"]<60
        and latest["5_MA"]>latest["10_MA"] and latest["5_MA"]>latest["20_MA"] and latest["5_MA"]<latest["50_MA"] and latest["20_MA"]<latest["50_MA"]) or
        (price_below_r1 and price_above_s1 and latest["close"]<latest["vwap"] and latest["close"]<latest["lower_band"] and latest["close"]<latest["5_MA"] 
        and latest["close"]<latest["20_MA"] 
        and latest["RSI_14"]<40 and latest["RSI_14_3ma"]>30 and latest["ROC_10"]<(-0.1)
        and latest["fastDfastK"]>10 and latest["fastK"]<5 and latest["fastD"]>10 and latest["fastK_3ma"]<20 and latest["fastK_20ma"]>45 
        and latest["ROC_10_last10"]>(-0.3)  
        and latest["Pivot_Point"]<latest["Pivot_Point_long"] and latest["vwap"]<latest["lower_band"] and latest["close"]<indicators["l50_MA"]) or
        (price_below_r1 and price_above_s1 and latest["close"]<latest["vwap"] and latest["close"]<latest["5_MA"] 
        and latest["close"]<latest["20_MA"] and latest["close"]<latest["50_MA"]
        and latest["RSI_14"]<35 and latest["ROC_10"]>(-0.3) and latest["fastDfastK"]>2
        and latest["fastK"]<15 and latest["fastD"]>20 and latest["fastK_3ma"]>20
        and latest["ROC_10_last10"]>(-0.1) and latest["close"]<latest["5_MA"] and latest["close"]<latest["pivot_point"]) or
        (price_below_r1 and price_above_s1 and latest["close"]<latest["vwap"] and latest["close"]<latest["lower_band"] 
        and latest["close"]<latest["5_MA"] and latest["RSI_14"]<30 and latest["ROC_10"]>(-2) and latest["ROC_10"]<0.5 
        and latest["fastK"]<15 and latest["fastD"]<15 and latest["fastK"]>latest["fastD"] and prev["RSI_14_20ma"]>40 
        and latest["ROC_10_last10"]>(-1) and latest["close"]<latest["pivot_point"] and latest["close"]<indicators["l5_MA"]) or
        (price_below_r1 and price_above_s1 and latest["close"]<latest["vwap"] and latest["close"]<latest["5_MA"] and latest["close"]<latest["10_MA"] 
        and latest["RSI_14"]<40 and latest["ROC_10"]>(-0.1) and latest["fastK"]>20 and latest["fastDfastK"]>1 
        and latest["fastK"]<50 and latest["close"]<latest["pivot_point"] and latest["ROC_10_last10"]>(-0.5) and latest["close"]<indicators["l5_MA"]) or
        
        #daily movers
        (price_below_r1 and price_above_s1 and latest["close"]<latest["vwap"] and latest["close"]<latest["pivot_point"] and latest["close"]<latest["5_MA"] 
        and latest["close"]<latest["20_MA"] and latest["close"]<latest["50_MA"] and latest["close"]<prev["S1"] and latest["close"]<prev["lower_band"]
        and latest["fastK"]<15 and latest["fastD"]<20 and latest["fastDfastK"]>1 
        and latest["RSI_14"]<30 and latest["ROC_10"]<(-5)
        and latest["ROC_10_last10"]>(-1) 
        and latest["close"]<latest["Pivot_Point"]
        and latest["close"]<latest["Pivot_Point_long"]) or
        
        #belowMA
        (price_below_r1 and price_above_s1 and latest["close"]<latest["vwap"] and latest["close"]<latest["pivot_point"] and latest["close"]<latest["5_MA"] 
        #and latest["close"]>prev["5_MA"]
        #and latest["close"]<prev["lower_band"]
        and latest["ATR"]<latest["ATR_MA"]
        and latest["close"]<latest["20_MA"] and latest["close"]<latest["50_MA"] and latest["RSI_14"]<45 and latest["ROC_10"]>(-0.5)
        and latest["ROC_10_last10"]>(-0.5) and latest["fastDfastK"]>1.1
        and latest["close"]<indicators["l5_MA"] and latest["close"]<indicators["l20_MA"] and latest["close"]<indicators["llower_band"]) #and low_volatility 
        
        or 
         #gainers
        (price_below_r1 and price_above_s1 and latest["RSI_14"]<40 and latest["ROC_10"]<(-0.1) and latest["ROC_10"]>(-2) 
        and latest["fastK"]<20 and latest["fastK"]<prev["fastK"]
        and latest["fastDfastK"]>1.1 and latest["close"]<latest["20_MA"] and latest["close"]<latest["vwap"] 
        and latest["vwap"]>indicators["l20_MA"] and latest["ROC_10_last10"]>(-1.5) and latest["close"]<latest["pivot_point"]
        and latest["ATR"]<latest["ATR_MA"] and latest["fastK_20ma"]>20 and latest["RSI_14_20ma"]>20)
        
        )
        
       
    
    if indicators is not None:
        #print(indicators)
        buy_signal=(buy_signal) and latest["volume"]>1000 and latest["trade_count"]>10 and latest["RSI_14"]<45 #or (latest["close"]<indicators["l5_MA"] and latest["close"]<indicators["l50_MA"] 
        #and latest["close"]<indicators["l20_MA"] and latest["close"]<indicators["lPivot_Point"]
        #and indicators["lRSI_14"]>35 and indicators["lRSI_14"]<70 and indicators["lfastK"]>indicators["lfastD"]) 
    # and latest["MACD_histogram"]>0) and price_above_r1 and high_volatility
    sell_signal = (
         
        (
          (latest["close"]>prev["5_MA"] and latest["close"]>prev["20_MA"] and latest["RSI_14"]>80 and latest["RSI_14"]>prev["RSI_14"]
        and latest["close"]>prev["close"] and latest["ROC_10"]>3
        and latest["fastK"]>90) or (latest["close"]>indicators["l5_MA"] and latest["RSI_14"]>80 and latest["fastK"]>90 
        and latest["ROC_10"]>1 and latest["close"]>prev["close"])
        or (latest["close"]>indicators["lR2"] and latest["RSI_14"]>80 and
        latest["fastK"]>90 and latest["ROC_10"]>1 and latest["close"]>prev["close"])
        )
        
            )
    
   #s sell_signal = (sell_signal and latest["fastK"]>80 and latest["close"]>latest["Pivot_Point"])

    return "BUY" if buy_signal else "SELL" if sell_signal else "HOLD"

#04072025_v1
def get_signal04072025_v1(latest,prev,indicators=None):
  
    last_row=latest
    # ATR thresholds to adjust risk
    high_volatility = last_row['ATR'] > last_row['ATR_MA'] #.mean()  # High volatility if ATR is above the average ATR
    low_volatility = last_row['ATR'] < last_row['ATR_MA'] #.mean()   # Low volatility if ATR is below the average ATR
    
    # Pivot Point Levels
    price_above_r1 = last_row['close'] > last_row['R1']  # Price above R1 (bullish signal)
    price_below_s1 = last_row['close'] < last_row['S1']  # Price below S1 (bearish signal)
    price_below_r1 = last_row['close'] < last_row['R1'] 
    price_above_s1 = last_row['close'] > last_row['S1']
    

    #and latest["ROC_10_last10"]>0 
    buy_signal = (
      (price_below_r1 and price_above_s1 and latest["close"]<latest["5_MA"] and latest["RSI_14"]<40 and latest["ROC_10"]<(-3) and latest["fastDfastK"]>1.2
        and latest["fastK_20ma"]>prev["fastK_20ma"]and prev["fastK_20ma"]<40 and latest["ROC_10_last10"]>0 ) or
        (price_below_r1 and price_above_s1 and latest["close"]<latest["5_MA"] and latest["RSI_14"]>20 and latest["RSI_14"]<40 and latest["ROC_10"]<(0.5)
        and latest["fastDfastK"]>1.2
        and latest["fastK_20ma"]<prev["fastK_20ma"] and latest["ROC_10_last10"]>0 and latest["close"]<indicators["l5_MA"] and indicators["lMACD_histogram"]<0
        and indicators["lfastK"]<15 and indicators["lfastK"]>15 and indicators["lRSI_14"]<40 and indicators["close"]<indicators["open"]) or 
        (price_below_r1 and price_above_s1 and latest["close"]<latest["5_MA"] and latest["RSI_14"]<40 and latest["ROC_10"]<(0.5)
        and latest["fastDfastK"]>1.1
        and latest["ROC_10_last10"]>0 and latest["close"]<indicators["l5_MA"] and indicators["lMACD_histogram"]<0
        and indicators["lfastK"]<15 and indicators["lfastD"]>10 and indicators["lRSI_14"]<40 and latest["close"]<indicators["open"]) or
        (price_below_r1 and price_above_s1 and latest["close"]<latest["vwap"] and latest["close"]<latest["5_MA"] 
        and latest["close"]<latest["20_MA"] and latest["RSI_14"]>30 and latest["RSI_14"]<40 and latest["ROC_10"]<(0.5)
        and latest["fastDfastK"]>2 and latest["fastK"]<15 and latest["fastD"]>25 and prev["fastK"]>30 and latest["close"]<latest["pivot_point"]
        and latest["ROC_10_last10"]>1 and latest["RSI_14_3ma"]>40 and latest["RSI_14_3ma"]<60 and latest["RSI_14_20ma"]>40 and latest["RSI_14_20ma"]<60
        and latest["5_MA"]>latest["10_MA"] and latest["5_MA"]>latest["20_MA"] and latest["5_MA"]<latest["50_MA"] and latest["20_MA"]<latest["50_MA"]) or
        (price_below_r1 and price_above_s1 and latest["close"]<latest["vwap"] and latest["close"]<latest["lower_band"] and latest["close"]<latest["5_MA"] 
        and latest["close"]<latest["20_MA"] 
        and latest["RSI_14"]<40 and latest["RSI_14_3ma"]>30 and latest["ROC_10"]<(-0.1)
        and latest["fastDfastK"]>10 and latest["fastK"]<5 and latest["fastD"]>10 and latest["fastK_3ma"]<20 and latest["fastK_20ma"]>45 
        and latest["ROC_10_last10"]>(-0.3)  
        and latest["Pivot_Point"]<latest["Pivot_Point_long"] and latest["vwap"]<latest["lower_band"] and latest["close"]<indicators["l50_MA"]) or
        (price_below_r1 and price_above_s1 and latest["close"]<latest["vwap"] and latest["close"]<latest["5_MA"] 
        and latest["close"]<latest["20_MA"] and latest["close"]<latest["50_MA"]
        and latest["RSI_14"]<35 and latest["ROC_10"]>(-0.3) and latest["fastDfastK"]>2
        and latest["fastK"]<15 and latest["fastD"]>20 and latest["fastK_3ma"]>20
        and latest["ROC_10_last10"]>(-0.1) and latest["close"]<latest["5_MA"] and latest["close"]<latest["pivot_point"]) or
        (price_below_r1 and price_above_s1 and latest["close"]<latest["vwap"] and latest["close"]<latest["lower_band"] 
        and latest["close"]<latest["5_MA"] and latest["RSI_14"]<30 and latest["ROC_10"]>(-2) and latest["ROC_10"]<0.5 
        and latest["fastK"]<15 and latest["fastD"]<15 and latest["fastK"]>latest["fastD"] and prev["RSI_14_20ma"]>40 
        and latest["ROC_10_last10"]>(-1) and latest["close"]<latest["pivot_point"] and latest["close"]<indicators["l5_MA"]) or
        (price_below_r1 and price_above_s1 and latest["close"]<latest["vwap"] and latest["close"]<latest["5_MA"] and latest["close"]<latest["10_MA"] 
        and latest["RSI_14"]<40 and latest["ROC_10"]>(-0.1) and latest["fastK"]>20 and latest["fastDfastK"]>1 
        and latest["fastK"]<50 and latest["close"]<latest["pivot_point"] and latest["ROC_10_last10"]>(-0.5) and latest["close"]<indicators["l5_MA"]) or
        
        #daily movers
        (price_below_r1 and price_above_s1 and latest["close"]<latest["vwap"] and latest["close"]<latest["pivot_point"] and latest["close"]<latest["5_MA"] 
        and latest["close"]<latest["20_MA"] and latest["close"]<latest["50_MA"] and latest["close"]<prev["S1"] and latest["close"]<prev["lower_band"]
        and latest["fastK"]<15 and latest["fastD"]<20 and latest["fastDfastK"]>1 
        and latest["RSI_14"]<30 and latest["ROC_10"]<(-5)
        and latest["ROC_10_last10"]>(-1) 
        and latest["close"]<latest["Pivot_Point"]
        and latest["close"]<latest["Pivot_Point_long"]) or
        
        #belowMA
        (price_below_r1 and price_above_s1 and latest["close"]<latest["vwap"] and latest["close"]<latest["pivot_point"] and latest["close"]<latest["5_MA"] 
        #and latest["close"]>prev["5_MA"]
        #and latest["close"]<prev["lower_band"]
        and latest["ATR"]<latest["ATR_MA"]
        and latest["close"]<latest["20_MA"] and latest["close"]<latest["50_MA"] and latest["RSI_14"]<45 and latest["ROC_10"]>(-0.5)
        and latest["ROC_10_last10"]>(-0.5) and latest["fastDfastK"]>1.1
        and latest["close"]<indicators["l5_MA"] and latest["close"]<indicators["l20_MA"] and latest["close"]<indicators["llower_band"]) #and low_volatility 
        
        or 
         #gainers
        (price_below_r1 and price_above_s1 and latest["RSI_14"]<40 and latest["ROC_10"]<(-0.1) and latest["ROC_10"]>(-2) 
        and latest["fastK"]<20 and latest["fastK"]<prev["fastK"]
        and latest["fastDfastK"]>1.1 and latest["close"]<latest["20_MA"] and latest["close"]<latest["vwap"] 
        and latest["vwap"]>indicators["l20_MA"] and latest["ROC_10_last10"]>(-1.5) and latest["close"]<latest["pivot_point"]
        and latest["ATR"]<latest["ATR_MA"] and latest["fastK_20ma"]>20 and latest["RSI_14_20ma"]>20) or 
        
        #moving up but sudden drop
        (price_below_r1 and price_above_s1 and latest["RSI_14"]<30 and prev["RSI_14"]>40 and latest["ROC_10"]<prev["ROC_10"] and prev["ROC_10"]>0 
        and latest["ROC_10"]<(-1) and latest["PercentChange"]<(-1)
        and latest["fastK"]<20 and latest["fastD"]>30 and latest["momentum"]<0 and latest["momentum"]<prev["momentum"] and latest["bollinger_width"]>prev["bollinger_width"]
        and prev["fastK"]>50 and latest["close"]<indicators["l5_MA"] and latest["close"]<indicators["l20_MA"]
        and latest["fastK"]<latest["fastD"] 
        and latest["above_ma_5"]<1
        and latest["fastK_3ma"]<50 and prev["fastK_3ma"]>50 and latest["fastK_20ma"]>20 and latest["RSI_14_20ma"]>20 and prev["ROC_10_last10"]>(0.3)) or
        
         #moving up but below 20MA and 5MA
        (price_below_r1 and price_above_s1 and latest["RSI_14"]<40 and latest["ROC_10"]>prev["ROC_10"] and latest["fastK"]>10
        and latest["fastK"]<30 and latest["momentum"]>prev["momentum"] and latest["bollinger_width"]<3
        and prev["fastK"]<30 and latest["close"]<indicators["l5_MA"] and latest["close"]<indicators["l20_MA"]
        and prev["above_ma_5"]>0 and latest["fastK"]>latest["fastD"] 
        and latest["above_ma_5"]>0 and latest["above_ma_5_3MA"]>0.33 and latest["above_ma_5_3MA"]<1 
        and latest["fastK_3ma"]<30 and latest["fastK_20ma"]>20 and latest["RSI_14_20ma"]>20)
        
        
        )
        
       
    
    if indicators is not None:
        #print(indicators)
        buy_signal=(buy_signal) and latest["volume"]>1000 and latest["trade_count"]>10 and latest["RSI_14"]<60 #or (latest["close"]<indicators["l5_MA"] and latest["close"]<indicators["l50_MA"] 
        #and latest["close"]<indicators["l20_MA"] and latest["close"]<indicators["lPivot_Point"]
        #and indicators["lRSI_14"]>35 and indicators["lRSI_14"]<70 and indicators["lfastK"]>indicators["lfastD"]) 
    # and latest["MACD_histogram"]>0) and price_above_r1 and high_volatility
    sell_signal = (
         
        (
          (latest["close"]>prev["5_MA"] and latest["close"]>prev["20_MA"] and latest["RSI_14"]>80 and latest["RSI_14"]>prev["RSI_14"]
        and latest["close"]>prev["close"] and latest["ROC_10"]>3
        and latest["fastK"]>90) or (latest["close"]>indicators["l5_MA"] and latest["RSI_14"]>80 and latest["fastK"]>90 
        and latest["ROC_10"]>1 and latest["close"]>prev["close"])
        or (latest["close"]>indicators["lR2"] and latest["RSI_14"]>80 and
        latest["fastK"]>90 and latest["ROC_10"]>1 and latest["close"]>prev["close"])
        )
        
            )
    
   #s sell_signal = (sell_signal and latest["fastK"]>80 and latest["close"]>latest["Pivot_Point"])

    return "BUY" if buy_signal else "SELL" if sell_signal else "HOLD"

#belowMA, gainers, drops
def get_signal(latest,prev,indicators=None):
  
    last_row=latest
    # ATR thresholds to adjust risk
    high_volatility = last_row['ATR'] > last_row['ATR_MA'] #.mean()  # High volatility if ATR is above the average ATR
    low_volatility = last_row['ATR'] < last_row['ATR_MA'] #.mean()   # Low volatility if ATR is below the average ATR
    
    # Pivot Point Levels
    price_above_r1 = last_row['close'] > last_row['R1']  # Price above R1 (bullish signal)
    price_below_s1 = last_row['close'] < last_row['S1']  # Price below S1 (bearish signal)
    price_below_r1 = last_row['close'] < last_row['R1'] 
    price_above_s1 = last_row['close'] > last_row['S1']
    

    #and latest["ROC_10_last10"]>0 
    buy_signal = (

        #price below long 5MA and long 20MA; moving up but sudden drop
        (price_below_r1 and price_above_s1 and latest["RSI_14"]<30 and prev["RSI_14"]>40 and latest["ROC_10"]<prev["ROC_10"] and prev["ROC_10"]>0 
        and latest["ROC_10"]<(-1) and latest["PercentChange"]<(-1)
        and latest["fastK"]<20 and latest["fastD"]>30 and latest["momentum"]<0 and latest["momentum"]<prev["momentum"] and latest["bollinger_width"]>prev["bollinger_width"]
        and prev["fastK"]>50 and latest["close"]<indicators["l5_MA"] and latest["close"]<indicators["l20_MA"]
        and latest["fastK"]<latest["fastD"] 
        and latest["above_ma_5"]<1
        and latest["fastK_3ma"]<50 and prev["fastK_3ma"]>50 and latest["fastK_20ma"]>20 and latest["RSI_14_20ma"]>20 and prev["ROC_10_last10"]>(0.3)) or
        
         #moving up but below 20MA and 5MA; low RSI and fastK
        (price_below_r1 and price_above_s1 and latest["RSI_14"]<40 and latest["ROC_10"]>prev["ROC_10"] and latest["fastK"]>10
        and latest["fastK"]<30 and latest["momentum"]>prev["momentum"] #and (100*(latest["upper_band"]-latest["lower_band"])/latest["lower_band"])<3
        and prev["fastK"]<30 and latest["close"]<indicators["l5_MA"] and latest["close"]<indicators["l20_MA"]
        and prev["above_ma_5"]>0 and latest["fastK"]>latest["fastD"] 
        and latest["above_ma_5"]>0 and latest["above_ma_5_3MA"]>0.33 and latest["above_ma_5_3MA"]<1 
        and latest["fastK_3ma"]<30 and latest["fastK_20ma"]>20 and latest["RSI_14_20ma"]>20) or 
        
         #moving up but below 20MA and 5MA; mid RSI and fastK with consistent above 5MA
        (price_below_r1 and price_above_s1 and latest["RSI_14"]>50 and latest["RSI_14"]<60 and latest["fastK"]>10
        and latest["fastK"]<50 and latest["momentum"]>prev["momentum"] and latest["momentum"]>0 #and (100*(latest["upper_band"]-latest["lower_band"])/latest["lower_band"])<3
        and prev["fastK"]<50 and latest["close"]<indicators["l5_MA"] and latest["close"]<indicators["l20_MA"]
        and prev["above_ma_5"]>0 and latest["fastK"]>latest["fastD"] and latest["fastK"]>prev["fastK"] 
        and latest["above_ma_5"]>0 and latest["above_ma_5_3MA"]>0.33 and latest["above_ma_5_3MA"]<1 and latest["RSI_14_3ma"]<60
        and latest["fastK_3ma"]<40 and latest["fastK_20ma"]>20 and latest["RSI_14_20ma"]>20) or
        
         #moving up but below 20MA and 5MA; high RSI and fastK with consistent above 5MA
        (price_below_r1 and price_above_s1 and latest["RSI_14"]>50 and latest["RSI_14"]<60 and latest["fastK"]>10
        and latest["fastK"]>50 and latest["fastK"]<80 and latest["momentum"]>prev["momentum"] and latest["momentum"]>0 #and (100*(latest["upper_band"]-latest["lower_band"])/latest["lower_band"])<3
        and prev["fastK"]>50 and latest["close"]<indicators["l5_MA"] and latest["close"]<indicators["l20_MA"]
        and prev["above_ma_5"]>0 and latest["fastK"]>latest["fastD"] and latest["fastK"]>prev["fastK"] 
        and latest["above_ma_5"]>0 and latest["above_ma_5_3MA"]>0.33 and latest["above_ma_5_3MA"]<=1 and latest["RSI_14_3ma"]<60
        and latest["fastK_3ma"]<40 and latest["fastK_20ma"]>20 and latest["RSI_14_20ma"]>20) or
        
        #moving up but below 20MA and 5MA; high RSI and fastK with consistent above 5MA
        (price_below_r1 and price_above_s1 and latest["RSI_14"]>30 and latest["RSI_14"]<60 and latest["fastK"]>10
        and latest["fastK"]<40 and latest["momentum"]>prev["momentum"] and latest["momentum"]>latest["momentum_3MA"] #and (100*(latest["upper_band"]-latest["lower_band"])/latest["lower_band"])<3
        and prev["fastK"]>latest["fastK"] and latest["close"]<indicators["l5_MA"] and latest["close"]<indicators["l20_MA"]
        and prev["above_ma_5"]>0 and latest["fastK"]>latest["fastD"]
        and latest["above_ma_5"]>0 and latest["above_ma_5_3MA"]>0.33 and latest["above_ma_5_3MA"]<=1 and latest["RSI_14_3ma"]<60
        and latest["fastK_3ma"]<30 and prev["fastK_3ma"]<25 and latest["fastK_20ma"]>20 and latest["RSI_14_20ma"]>20) or
        
        #moving up but below 20MA and 5MA; high RSI and fastK with consistent above 5MA; temporary drop
        (price_below_r1 and price_above_s1 and latest["RSI_14"]>30 and latest["RSI_14"]<65 and latest["fastK"]>10
        and latest["fastK"]<70 and latest["momentum"]<prev["momentum"] and latest["momentum"]>0 and prev["momentum"]>0 and latest["momentum"]<latest["momentum_3MA"] 
        and prev["fastK"]>latest["fastK"] and latest["close"]<indicators["l5_MA"] and latest["close"]<indicators["l20_MA"]
        and prev["above_ma_5"]>0 and latest["fastK"]<latest["fastD"]
        and latest["above_ma_5"]>0 and latest["above_ma_5_3MA"]>0.33 and latest["above_ma_5_3MA"]<=1 and latest["above_ma_50_3MA"]==1 and latest["RSI_14_3ma"]>50
        and latest["fastK_3ma"]>60 and prev["fastK_3ma"]>50 and latest["fastK_20ma"]>80) or
        
        check_sb3_buy_signal(latest)
        
        )
        
       
    
    if indicators is not None:
        #print(indicators)
        buy_signal=(buy_signal) and latest["volume"]>1000 and latest["trade_count"]>10 and latest["RSI_14"]<65 #and (100*(latest["upper_band"]-latest["lower_band"])/(latest["lower_band"]))<5 #or (latest["close"]<indicators["l5_MA"] and latest["close"]<indicators["l50_MA"] 
        #and latest["close"]<indicators["l20_MA"] and latest["close"]<indicators["lPivot_Point"]
        #and indicators["lRSI_14"]>35 and indicators["lRSI_14"]<70 and indicators["lfastK"]>indicators["lfastD"]) 
    # and latest["MACD_histogram"]>0) and price_above_r1 and high_volatility
    sell_signal = (
         
        (
          (latest["close"]>prev["5_MA"] and latest["close"]>prev["20_MA"] and latest["RSI_14"]>80 and latest["RSI_14"]>prev["RSI_14"]
        and latest["close"]>prev["close"] and latest["ROC_10"]>1
        and latest["fastK"]>90) or (latest["close"]>indicators["l5_MA"] and latest["RSI_14"]>80 and latest["fastK"]>90 
        and latest["ROC_10"]>1 and latest["close"]>prev["close"])
        or (latest["close"]>indicators["lR2"] and latest["RSI_14"]>80 and
        latest["fastK"]>90 and latest["ROC_10"]>1 and latest["close"]>prev["close"])
        )
        
            )
    
   #s sell_signal = (sell_signal and latest["fastK"]>80 and latest["close"]>latest["Pivot_Point"])

    return "BUY" if buy_signal else "SELL" if sell_signal else "HOLD"

#03252025
def get_signalv0325(latest,prev,indicators=None):
  
    last_row=latest
    # ATR thresholds to adjust risk
    high_volatility = last_row['ATR'] > last_row['ATR_MA'] #.mean()  # High volatility if ATR is above the average ATR
    low_volatility = last_row['ATR'] < last_row['ATR_MA'] #.mean()   # Low volatility if ATR is below the average ATR
    
    # Pivot Point Levels
    price_above_r1 = last_row['close'] > last_row['R1']  # Price above R1 (bullish signal)
    price_below_s1 = last_row['close'] < last_row['S1']  # Price below S1 (bearish signal)
    
    #print(latest["close"]) (latest["close"]<latest["lower_band"]) and 
    #and latest["RSI_14"]>prev["RSI_14"] and latest["close"]>prev["close"] # and latest["MACD_histogram"]<0 and latest["MACD_histogram"]>prev["MACD_histogram"]
    #and latest["MACD_histogram"]>prev["MACD_histogram"]) and price_below_s1 and low_volatility
    buy_signal = (
        ((latest["close"]<prev["5_MA"] and latest["close"]<prev["20_MA"] and prev["RSI_14"]>20 and latest["RSI_14"]<70 and latest["RSI_14"]>prev["RSI_14"]
        and latest["close"]<prev["close"] and latest["PercentChange"]<(-2) 
        and latest["fastK"]>latest["fastK_20MA"] and latest["fastK"]<25) ) 
    )
    
    
    if indicators is not None:
        #print(indicators)
        buy_signal=(buy_signal) # and latest["close"]<indicators["l5_MA"] and latest["close"]<indicators["l50_MA"] 
        #and latest["close"]<indicators["l20_MA"] and latest["close"]<indicators["lPivot_Point"]) 
        #and indicators["lRSI_14"]>35 and indicators["lRSI_14"]<70 and indicators["lfastK"]>indicators["lfastD"]
    # and latest["MACD_histogram"]>0) and price_above_r1 and high_volatility
    sell_signal = (
         
        ((latest["close"]>prev["5_MA"] and latest["close"]>prev["20_MA"] and latest["RSI_14"]>80 and latest["RSI_14"]>prev["RSI_14"]
        and latest["close"]>prev["close"] 
        and latest["fastK"]<prev["fastK"])  )
        
            )
    
   #s sell_signal = (sell_signal and latest["fastK"]>80 and latest["close"]>latest["Pivot_Point"])

    return "BUY" if buy_signal else "SELL" if sell_signal else "HOLD"


def get_signalv03242025(latest,prev,indicators=None):
    #print(latest["close"]) (latest["close"]<latest["lower_band"]) and 
    #and latest["RSI_14"]>prev["RSI_14"] and latest["close"]>prev["close"] # and latest["MACD_histogram"]<0 and latest["MACD_histogram"]>prev["MACD_histogram"]
    buy_signal = (
        ((latest["close"]<prev["5_MA"] and latest["close"]<prev["20_MA"] and prev["RSI_14"]>20 and latest["RSI_14"]<45 and latest["RSI_14"]>prev["RSI_14"]
        and latest["close"]>prev["close"] 
        and latest["fastK"]>prev["fastK"] and latest["MACD_histogram"]>prev["MACD_histogram"]) or (latest["close"]<prev["5_MA"] and latest["close"]<prev["20_MA"] and prev["RSI_14"]>30 and latest["RSI_14"]<45 and latest["RSI_14"]>prev["RSI_14"] and latest["close"]>prev["close"] 
        and latest["fastDfastK"]>0.9 and latest["fastK"]>prev["fastK"]) or (latest["close"]<prev["Pivot_Point"] 
        and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] 
        and latest["close"]<prev["20_MA"] and latest["close"]<prev["50_MA"] 
        and latest["RSI_14"]>35 and latest["fastK"]<40 and 
        prev["RSI_14"]<latest["RSI_14"] and latest["fastK"]>35
        and latest["fastK"]<40 and latest["fastK"]>prev["fastK"] and latest["ROC_10"]<(0.5) and latest["PercentChange"]<0.05) or (latest["close"]<prev["Pivot_Point"]
        and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] 
        and latest["close"]<prev["20_MA"] and latest["close"]<prev["50_MA"] 
        and latest["RSI_14"]<30 and prev["fastK"]<30 and latest["fastK"]<15 and 
        latest["ROC_10"]<(0.5) and latest["PercentChange"]<0) or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["close"]<prev["20_MA"] and latest["close"]<prev["50_MA"] and latest["RSI_14"]<30 and prev["fastK"]<30 and prev["RSI_14"]<latest["RSI_14"] and latest["fastK"]<15 and latest["fastK"]>prev["fastK"] and prev["fastK"]<5 and latest["ROC_10"]<(0.5) and latest["PercentChange"]<0.05) or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["close"]<prev["20_MA"] and latest["close"]<prev["50_MA"] and latest["RSI_14"]<35 and latest["fastK"]<3 and latest["ROC_10"]<(0.5) and latest["PercentChange"]<0.5) or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["close"]<prev["20_MA"] and latest["close"]<prev["50_MA"] and latest["close"]>prev["120_MA"] and latest["RSI_14"]<40 and latest["fastK"]<20 and latest["ROC_10"]<(0) and latest["PercentChange"]<0.5 and latest["fastDfastK"]>2) or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["ROC_10"]<(-5) and latest["PercentChange"]<(-5) and latest["close"]<latest["5_MA"] and latest["close"]>latest["20_MA"] and latest["close"]<latest["50_MA"]) or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["ROC_10"]<(-5) and latest["PercentChange"]<(-5) and latest["close"]<latest["5_MA"] and latest["fastK"]<5 and latest["fastDfastK"]>3 and latest["close"]<latest["50_MA"]) or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["ROC_10"]>1 and latest["PercentChange"]<(-10) and latest["close"]<latest["5_MA"] and latest["fastK"]<10 and latest["fastDfastK"]>3 and latest["close"]<latest["50_MA"]) or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["ROC_10"]<(-5) and prev["ROC_10"]>5 and latest["PercentChange"]<(-5) and latest["close"]<latest["5_MA"] and latest["fastK"]<15 and latest["fastDfastK"]>3 and latest["close"]<latest["50_MA"]) or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["ROC_10"]>prev["ROC_10"] and prev["ROC_10"]<(-1) and latest["PercentChange"]<(0.5) and prev["PercentChange"]<(-1) and latest["PercentChange"]>prev["PercentChange"] and latest["close"]<latest["5_MA"] and prev["fastK"]<15 and latest["fastDfastK"]>1 and prev["fastK"]<latest["fastK"] and latest["fastK"]<20 and prev["RSI_14"]<30 and latest["RSI_14"]>prev["RSI_14"] and latest["RSI_14"]<35) or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["PercentChange"]<(0.5) and prev["PercentChange"]<(0.5) and latest["fastDfastK"]>1 and prev["fastDfastK"]>5 and latest["fastK"]>prev["fastK"] and latest["fastK"]>10 and prev["fastK"]<10 and latest["fastK"]<20 and latest["RSI_14"]<40) or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["PercentChange"]<(-2) and latest["fastDfastK"]>1.25 and latest["fastK"]<30 and latest["PercentChange"]>(-25))
          )  
    )
    #latest["close"]<latest["lower_band"] and latest["MACD_histogram"]<0 and latest["MACD_histogram"]>prev["MACD_histogram"] and latest["MACD_histogram"]<0
    #100*(latest["Pivot_Point_long"]-latest["Pivot_Point"])/latest["Pivot_Point"]>0
    buy_signal=(buy_signal and latest["RSI_14"]>20 and latest["ROC_10"]<(-1) and latest["Pivot_Point_long"]>latest["Pivot_Point"]
    and latest["ROC_10_last3"]<(-1) and latest["fastK"]<30 and latest["fastD"]>10 and latest["MACD_histogram"]<(0) 
    and latest["vwap"]>latest["close"] and latest["close"]<latest["lower_band"])
    
    if indicators is not None:
        #print(indicators)
        buy_signal=(buy_signal) # and latest["close"]<indicators["l5_MA"] and latest["close"]<indicators["l50_MA"] 
        #and latest["close"]<indicators["l20_MA"] and latest["close"]<indicators["lPivot_Point"]) 
        #and indicators["lRSI_14"]>35 and indicators["lRSI_14"]<70 and indicators["lfastK"]>indicators["lfastD"]
    
    sell_signal = (
        (latest["close"]>prev["Pivot_Point"] and latest["close"]>latest["Pivot_Point"] 
        and latest["RSI_14"]>50 and latest["fastK"]>50 and latest["fastK"]<prev["fastK"] 
        and latest["fastK"]>latest["fastD"] and latest["ROC_10"]<prev["ROC_10"] 
        and latest["ROC_10"]>0 and latest["close"] > latest["EMA_20"] 
        and latest["close"] > prev["EMA_20"] and prev["EMA_20"] >= prev["SMA_20"]) or (latest["close"]>prev["Pivot_Point"] and latest["close"]>latest["Pivot_Point"] and latest["RSI_14"]>50 and latest["fastK"]>50 and latest["fastK"]>prev["fastK"] and latest["fastK"]>latest["fastD"] and latest["fastD"]>prev["fastD"] and latest["ROC_10"]>prev["ROC_10"] and latest["ROC_10"]>1 and latest["close"] > latest["EMA_20"] and latest["close"] > prev["EMA_20"] and prev["EMA_20"] >= prev["SMA_20"]) or (latest["close"]>prev["close"] and latest["ROC_10"]>prev["ROC_10"] and latest["PercentChange"]>(5))
    )
    
    sell_signal = (sell_signal and latest["fastK"]>80 and latest["close"]>latest["Pivot_Point"])

    return "BUY" if buy_signal else "SELL" if sell_signal else "HOLD"

def get_signalv032425(latest,prev):
    #print(latest["close"])
    #and latest["RSI_14"]>prev["RSI_14"] and latest["close"]>prev["close"] 
    buy_signal = (
        (latest["close"]<prev["5_MA"] and latest["close"]<prev["20_MA"] and prev["RSI_14"]>20 and latest["RSI_14"]<45 and latest["RSI_14"]>prev["RSI_14"]
        and latest["close"]>prev["close"] 
        and latest["fastK"]>prev["fastK"] and latest["MACD_histogram"]>prev["MACD_histogram"]) or (latest["close"]<prev["5_MA"] and latest["close"]<prev["20_MA"] and prev["RSI_14"]>30 and latest["RSI_14"]<45 and latest["RSI_14"]>prev["RSI_14"] and latest["close"]>prev["close"] 
        and latest["fastDfastK"]>0.9 and latest["fastK"]>prev["fastK"]) or (latest["close"]<prev["Pivot_Point"] 
        and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] 
        and latest["close"]<prev["20_MA"] and latest["close"]<prev["50_MA"] 
        and latest["RSI_14"]>35 and latest["fastK"]<40 and 
        prev["RSI_14"]<latest["RSI_14"] and latest["fastK"]>35
        and latest["fastK"]<40 and latest["fastK"]>prev["fastK"] and latest["ROC_10"]<(0.5) and latest["PercentChange"]<0.05) or (latest["close"]<prev["Pivot_Point"]
        and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] 
        and latest["close"]<prev["20_MA"] and latest["close"]<prev["50_MA"] 
        and latest["RSI_14"]<30 and prev["fastK"]<30 and latest["fastK"]<15 and 
        latest["ROC_10"]<(0.5) and latest["PercentChange"]<0) or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["close"]<prev["20_MA"] and latest["close"]<prev["50_MA"] and latest["RSI_14"]<30 and prev["fastK"]<30 and prev["RSI_14"]<latest["RSI_14"] and latest["fastK"]<15 and latest["fastK"]>prev["fastK"] and prev["fastK"]<5 and latest["ROC_10"]<(0.5) and latest["PercentChange"]<0.05) or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["close"]<prev["20_MA"] and latest["close"]<prev["50_MA"] and latest["RSI_14"]<35 and latest["fastK"]<3 and latest["ROC_10"]<(0.5) and latest["PercentChange"]<0.5) or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["close"]<prev["20_MA"] and latest["close"]<prev["50_MA"] and latest["close"]>prev["120_MA"] and latest["RSI_14"]<40 and latest["fastK"]<20 and latest["ROC_10"]<(0) and latest["PercentChange"]<0.5 and latest["fastDfastK"]>2) or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["ROC_10"]<(-5) and latest["PercentChange"]<(-5) and latest["close"]<latest["5_MA"] and latest["close"]>latest["20_MA"] and latest["close"]<latest["50_MA"]) or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["ROC_10"]<(-5) and latest["PercentChange"]<(-5) and latest["close"]<latest["5_MA"] and latest["fastK"]<5 and latest["fastDfastK"]>3 and latest["close"]<latest["50_MA"]) or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["ROC_10"]>1 and latest["PercentChange"]<(-10) and latest["close"]<latest["5_MA"] and latest["fastK"]<10 and latest["fastDfastK"]>3 and latest["close"]<latest["50_MA"]) or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["ROC_10"]<(-5) and prev["ROC_10"]>5 and latest["PercentChange"]<(-5) and latest["close"]<latest["5_MA"] and latest["fastK"]<15 and latest["fastDfastK"]>3 and latest["close"]<latest["50_MA"]) or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["ROC_10"]>prev["ROC_10"] and prev["ROC_10"]<(-1) and latest["PercentChange"]<(0.5) and prev["PercentChange"]<(-1) and latest["PercentChange"]>prev["PercentChange"] and latest["close"]<latest["5_MA"] and prev["fastK"]<15 and latest["fastDfastK"]>1 and prev["fastK"]<latest["fastK"] and latest["fastK"]<20 and prev["RSI_14"]<30 and latest["RSI_14"]>prev["RSI_14"] and latest["RSI_14"]<35) or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["PercentChange"]<(0.5) and prev["PercentChange"]<(0.5) and latest["fastDfastK"]>1 and prev["fastDfastK"]>5 and latest["fastK"]>prev["fastK"] and latest["fastK"]>10 and prev["fastK"]<10 and latest["fastK"]<20 and latest["RSI_14"]<40) or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["PercentChange"]<(-2) and latest["fastDfastK"]>1.25 and latest["fastK"]<30 and latest["PercentChange"]>(-25))
            
    )
    
    sell_signal = (
        (latest["close"]>prev["Pivot_Point"] and latest["close"]>latest["Pivot_Point"] and latest["RSI_14"]>50 and latest["fastK"]>50 and latest["fastK"]<prev["fastK"] and latest["fastK"]>latest["fastD"] and latest["ROC_10"]<prev["ROC_10"] and latest["ROC_10"]>0 and latest["close"] > latest["EMA_20"] and latest["close"] > prev["EMA_20"] and prev["EMA_20"] >= prev["SMA_20"]) or (latest["close"]>prev["Pivot_Point"] and latest["close"]>latest["Pivot_Point"] and latest["RSI_14"]>50 and latest["fastK"]>50 and latest["fastK"]>prev["fastK"] and latest["fastK"]>latest["fastD"] and latest["fastD"]>prev["fastD"] and latest["ROC_10"]>prev["ROC_10"] and latest["ROC_10"]>1 and latest["close"] > latest["EMA_20"] and latest["close"] > prev["EMA_20"] and prev["EMA_20"] >= prev["SMA_20"]) or (latest["close"]>prev["close"] and latest["ROC_10"]>prev["ROC_10"] and latest["PercentChange"]>(5))
    )

    return "BUY" if buy_signal else "SELL" if sell_signal else "HOLD"


def get_signal_crypto(latest,prev):
    #print(latest["close"])
    
    buy_signal = (
        (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["close"]<prev["20_MA"] and latest["close"]<prev["50_MA"] and latest["close"]>prev["120_MA"] and latest["RSI_14"]<40 and latest["fastK"]<20 and latest["ROC_10"]<(0) and latest["PercentChange"]<0.5 and latest["fastDfastK"]>2) or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["ROC_10"]<(-5) and latest["PercentChange"]<(-5) and latest["close"]<latest["5_MA"] and latest["close"]>latest["20_MA"] and latest["close"]<latest["50_MA"]) or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["ROC_10"]<(-5) and latest["PercentChange"]<(-5) and latest["close"]<latest["5_MA"] and latest["fastK"]<5 and latest["fastDfastK"]>3 and latest["close"]<latest["50_MA"]) or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["ROC_10"]>1 and latest["PercentChange"]<(-10) and latest["close"]<latest["5_MA"] and latest["fastK"]<10 and latest["fastDfastK"]>3 and latest["close"]<latest["50_MA"]) or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["ROC_10"]<(-5) and prev["ROC_10"]>5 and latest["PercentChange"]<(-5) and latest["close"]<latest["5_MA"] and latest["fastK"]<15 and latest["fastDfastK"]>3 and latest["close"]<latest["50_MA"]) or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["ROC_10"]>prev["ROC_10"] and prev["ROC_10"]<(-1) and latest["PercentChange"]<(0.5) and prev["PercentChange"]<(-1) and latest["PercentChange"]>prev["PercentChange"] and latest["close"]<latest["5_MA"] and prev["fastK"]<15 and latest["fastDfastK"]>1 and prev["fastK"]<latest["fastK"] and latest["fastK"]<20 and prev["RSI_14"]<30 and latest["RSI_14"]>prev["RSI_14"] and latest["RSI_14"]<35) or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["PercentChange"]<(0.5) and prev["PercentChange"]<(0.5) and latest["fastDfastK"]>1 and prev["fastDfastK"]>5 and latest["fastK"]>prev["fastK"] and latest["fastK"]>10 and prev["fastK"]<10 and latest["fastK"]<20 and latest["RSI_14"]<40) or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["PercentChange"]<(-2) and latest["fastDfastK"]>1.25 and latest["fastK"]<30 and latest["PercentChange"]>(-25))
            
    )
    
    sell_signal = (
        (latest["close"]>prev["Pivot_Point"] and latest["close"]>latest["Pivot_Point"] and latest["RSI_14"]>50 and latest["fastK"]>50 and latest["fastK"]<prev["fastK"] and latest["fastK"]>latest["fastD"] and latest["ROC_10"]<prev["ROC_10"] and latest["ROC_10"]>0 and latest["close"] > latest["EMA_20"] and latest["close"] > prev["EMA_20"] and prev["EMA_20"] >= prev["SMA_20"]) or (latest["close"]>prev["Pivot_Point"] and latest["close"]>latest["Pivot_Point"] and latest["RSI_14"]>50 and latest["fastK"]>50 and latest["fastK"]>prev["fastK"] and latest["fastK"]>latest["fastD"] and latest["fastD"]>prev["fastD"] and latest["ROC_10"]>prev["ROC_10"] and latest["ROC_10"]>1 and latest["close"] > latest["EMA_20"] and latest["close"] > prev["EMA_20"] and prev["EMA_20"] >= prev["SMA_20"]) or (latest["close"]>prev["close"] and latest["ROC_10"]>prev["ROC_10"] and latest["PercentChange"]>(5))
    )

    return "BUY" if buy_signal else "SELL" if sell_signal else "HOLD"


def get_signal_adv(latest,prev):
    buy_signal = (
        (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["close"]<prev["20_MA"] and latest["close"]<prev["50_MA"] and latest["close"]>prev["120_MA"] and latest["RSI_14"]<50 and latest["fastK"]<50 and (latest["fastK"]>prev["fastK"] or latest["fastK"]<5) and latest["fastK"]<latest["fastD"] and latest["ROC_10"]>prev["ROC_10"] and latest["ROC_10"]<(0) and latest["close"]<latest["EMA_20"] and latest["close"]<latest["SMA_20"])
            or (latest["close"]<prev["Pivot_Point"] and latest["close"]<latest["Pivot_Point"] and latest["close"]<prev["5_MA"] and latest["close"]<prev["20_MA"] and latest["close"]<prev["50_MA"] and latest["close"]>prev["120_MA"] and latest["RSI_14"]<50 and latest["fastK"]<20 and latest["fastK"]<latest["fastD"]  and latest["ROC_10"]<(-1) and latest["close"]<latest["EMA_20"] and latest["close"]<latest["SMA_20"])
    )
    
    sell_signal = (
        (latest["close"]>prev["Pivot_Point"] and latest["close"]>latest["Pivot_Point"] and latest["RSI_14"]>50 and latest["fastK"]>50 and latest["fastK"]<prev["fastK"] and latest["fastK"]>latest["fastD"] and latest["ROC_10"]<prev["ROC_10"] and latest["ROC_10"]>0 and latest["close"] > latest["EMA_20"] and latest["close"] > prev["EMA_20"] and prev["EMA_20"] >= prev["SMA_20"]) or (latest["close"]>prev["Pivot_Point"] and latest["close"]>latest["Pivot_Point"] and latest["RSI_14"]>50 and latest["fastK"]>50 and latest["fastK"]>prev["fastK"] and latest["fastK"]>latest["fastD"] and latest["fastD"]>prev["fastD"] and latest["ROC_10"]>prev["ROC_10"] and latest["ROC_10"]>1 and latest["close"] > latest["EMA_20"] and latest["close"] > prev["EMA_20"] and prev["EMA_20"] >= prev["SMA_20"])
    )

    return "BUY" if buy_signal else "SELL" if sell_signal else "HOLD"

def generate_trade_signals_backtest_fromscores(df,symbol=None,score_threshold=7,avoid_score_thresh=1,decline_score_thresh=3):

    #indicators = calculate_indicators_hist(symbol)
    signals = []
    stop_losses=[]
    take_profits=[]
    pred_gains=[]
    print(df["buy_score"].max())
    print(df[df["buy_score"]>6])
    for i in range(1, len(df)):
        
        price=df['close'].iloc[i]
        latest=df.iloc[i,:]
        prev=df.iloc[i-1,:]
        # Calculate stop-loss and take-profit levels
        stop_loss = price * (1 - RISK_TOLERANCE)
        take_profit = price * (1 + REWARD_THRESHOLD)
        #indicators = calculate_indicators_hist_fromdf(df.iloc[1:i])
         #,score_threshold=7,avoid_score_thresh=5,decline_score_thresh=3)
        #latest=df.iloc[-1]
        #and latest["buy_score"]>4
        #print(latest)
        #latest["Technical.Indicators.Signal"]=get_trade_signal_daily(symbol)
        #df.loc[df.index[-1], "Technical.Indicators.Signal"]=latest["Technical.Indicators.Signal"]
        #pred_gain=pred_gains_fromdfv2(latest)
        #or (buy_score>8)
        buy_signal= ((
        (1 < latest["recovery_score"] < 3 and 1 < latest["momentum_score"] <3 and latest["avoid_score"]<1 and latest["decline_score"]<6 and latest["decline_score"]>6
        and latest["buy_score"]>3 and latest["buy_score"]<6 and latest["rsi"]<30 and latest["fastk"]<5 and latest["volume_change"]>100) or
        (3<latest["momentum_score"]<5 and 5 < latest["recovery_score"] < 7 and latest["avoid_score"]>0 and 2<latest["avoid_score"]<4 and latest["decline_score"]<1 and latest["buy_score"]>4 and latest["volume_change"]>100) or
        (0<latest["momentum_score"]<2 and 0 < latest["recovery_score"] < 2 and latest["avoid_score"]==0 and 3<latest["decline_score"]<5 and 3<latest["buy_score"]<5 and latest["volume_change"]>300) or
        (latest["fastk"]>75 and latest["fastk_5"]<50 and latest["fastd"]<50 and latest["close"]>latest["sma_5"] and latest["close"]>latest["sma_20"] and latest["sma_20"]>latest["sma_5"]
        and latest["sma_5"]<latest["sma_10"] and latest["rsi"]>50 and latest["roc"]>0 and latest["roc_5"]<0 and latest["rsi_5"]<50 
        and 4<latest["momentum_score"]<6 and 5 < latest["recovery_score"] < 7 and 1<latest["avoid_score"]<3 and latest["decline_score"]<1 and 2<latest["buy_score"]<5 and latest["volume_change"]>300) or
        (latest["rsi"]<30 and latest["rsi_5"]<30 and latest["fastk"]<5 and latest["fastd"]>5 and latest["bb_width"]<latest["bb_width_5"] and latest["volume_change"]>40
        and 4<latest["decline_score"]<6 and 2<latest["buy_score"]<5 and 1 < latest["recovery_score"] < 3 and 1<latest["avoid_score"]<3 and 0<latest["momentum_score"]<2) or
        ( (50<latest["rsi"]<60 and latest["rsi_5"]<50 and 80<latest["fastk"]<90 and latest["fastk_5"]<55 and 45<latest["fastd"]<60 and latest["bb_width"]>latest["bb_width_5"] 
        and latest["volume_change"]>1000
        and latest["decline_score"]<1 and latest["buy_score"]>5 and latest["recovery_score"] >6 and 1<latest["avoid_score"]<3 and latest["momentum_score"]>4)) or 
        (( (60<latest["rsi"]<70 and 60<latest["rsi_5"]<70 and 70<latest["fastk"]<80 and latest["fastk_5"]>85 and latest["fastk"]<latest["fastd"]<95 and latest["bb_width"]>latest["bb_width_5"] 
        and latest["volume_change"]>(-1) and latest["momentum"]>0 and latest["roc_5"]>0 and latest["roc"]>latest["roc_5"] and latest["roc_prev"]>latest["roc"] and latest["roc_prev"]>latest["roc_5"]
        and latest["close"]<latest["sma_5"] and latest["close"]>latest["sma_10"] and 2<latest["decline_score"]<4 and 1<latest["buy_score"]<3
        and 0<latest["recovery_score"]<2 and 0<latest["avoid_score"]<2 and 2<latest["momentum_score"]<4))) or 
        (50<latest["rsi"]<65 and latest["rsi_5"]<50 and 40<latest["fastk"]<50 and latest["fastk_5"]<50 and 40<latest["fastd"]<50 and latest["bb_width"]>latest["bb_width_5"] 
        and latest["volume_change"]>200  and latest["roc"]>latest["roc_5"]
        and latest["decline_score"]<1 and latest["buy_score"]>5 and latest["recovery_score"] >6 and 0<latest["avoid_score"]<2 and latest["momentum_score"]>4) or
        ((40<latest["rsi"]<50 and latest["rsi_5"]>50 and latest["fastk"]<25 and latest["fastk_5"]>50 and latest["fastd"]>50 and latest["bb_width"]>latest["bb_width_5"] 
        and latest["volume_change"]>1000  and latest["roc"]<latest["roc_5"] and latest["roc"]>(-0.1) and latest["momentum"]>0
        and 2<latest["decline_score"]<5 and latest["buy_score"]>5 and 1<latest["recovery_score"]<3 and latest["avoid_score"]<1 and 2<latest["momentum_score"]<4)) or 
        (latest["buy_signal"] and 4<latest["buy_score"]<6 and latest["recovery_score"]>5 and latest["momentum_score"]>3) #or (pred_gain>0)
        ) and (latest["momentum_score"]<4 and latest['volume_change_ratio']>2 and latest['fastk']<50))
        #print(latest["buy_score"])  and 2<latest["momentum_score"]<5 and 2<latest["buy_score"]<6
        #print(latest["fastk"])
        #print(latest["fastk_5"])
        #print(latest["buy_signal"])
        #print(latest)
        buy_signal= latest["pricedrop_signal"] #or latest["rally_signal"] or latest["recovery_signal"] #(pred_gain>0 and latest["buy_signal"] and 2<latest["buy_score"]<6 and 1<latest["momentum_score"]<5 and latest['volume_change_ratio']>1 and latest['fastk']<80)
        #if (score >= score_threshold and avoid_score<avoid_score_thresh and decline_score<decline_score_thresh and buy_score>4):
        
        sell_signal=False #((latest["avoid_score"]>4))
        
        signal= "BUY" if buy_signal else "SELL" if sell_signal else "HOLD"
        #signal="BUY" if buy_signal else "SELL" if sell_signal else "HOLD"
        #signal=get_signal(latest,prev,indicators)
        #pred_gain=pred_gains_fromdf(df.iloc[1:i])
        #pred_gains.append(pred_gain)
        #signal=get_signal(latest,prev)
        if(i>10):
          health_status=True #evaluate_df_health(df.iloc[1:(i-1)])
          pred_gain=None #pred_gains_fromdf(df.iloc[1:i])
          pred_gains.append(pred_gain)
        else:
          health_status=True
          pred_gains.append(None)
        if health_status:
          signals.append(signal)
          if signal=="BUY" or signal=="SELL":
              stop_losses.append(stop_loss)
              take_profits.append(take_profit)
          else:
              stop_losses.append(None)
              take_profits.append(None)
        else:
          print("Poor health")
          signals.append("HOLD-Poor")
          stop_losses.append(None)
          take_profits.append(None)
        #if price<df["close"].iloc[i-1] and price < pivot and price < ma_5 and price<ma_20 and price<ma_50 and :
        #    signals.append("BUY")
        #elif price > pivot and price > ma_50 and price>ma_5:
        #    signals.append("SELL")
        #else:
        #    signals.append("HOLD")
    df["Signal"] = ["HOLD"] + signals  # Align with data
    df["Stop_Loss"] = [None] + stop_losses
    df["Take_Profit"] = [None] + take_profits
    df["Pred_Gain"] = [None] + pred_gains
    return df

def generate_trade_signals_backtest_fromdfA(df,symbol):
    #print("Here2")
    #print(start_date)
    #df = calculate_indicators_histall(symbol,200,daily,assettype,start_date)
    #df = calculate_indicators(df)
    
    #df2=get_stock_data_days(symbol,200)
    #print(df.iloc[0:10])

    indicators = calculate_indicators_hist(symbol)
    signals = []
    stop_losses=[]
    take_profits=[]
    pred_gains=[]
    
    for i in range(1, len(df)):
        price = df["close"].iloc[i]
        pivot = df["Pivot_Point"].iloc[i]
        ma_5 = df["5_MA"].iloc[i]
        ma_20 = df["20_MA"].iloc[i]
        ma_50 = df["50_MA"].iloc[i]
        ma_120 = df["120_MA"].iloc[i]
        latest=df.iloc[i,:]
        prev=df.iloc[i-1,:]
        # Calculate stop-loss and take-profit levels
        stop_loss = price * (1 - RISK_TOLERANCE)
        take_profit = price * (1 + REWARD_THRESHOLD)
        signal=get_signal(latest,prev,indicators)
        pred_gain=None #pred_gains_fromdf(df.iloc[1:i])
        if(i>10):
          health_status=evaluate_df_health(df.iloc[1:(i-1)])
        else:
          health_status=True
        if health_status:
          signals.append(signal)
          pred_gains.append(pred_gain)
          if signal=="BUY" or signal=="SELL":
              stop_losses.append(stop_loss)
              take_profits.append(take_profit)
          else:
              stop_losses.append(None)
              take_profits.append(None)
        else:
          print("Poor health")
          signals.append("HOLD-Poor")
          stop_losses.append(None)
          take_profits.append(None)
          pred_gains.append(pred_gain)
        #if price<df["close"].iloc[i-1] and price < pivot and price < ma_5 and price<ma_20 and price<ma_50 and :
        #    signals.append("BUY")
        #elif price > pivot and price > ma_50 and price>ma_5:
        #    signals.append("SELL")
        #else:
        #    signals.append("HOLD")
    df["Signal"] = ["HOLD"] + signals  # Align with data
    df["Stop_Loss"] = [None] + stop_losses
    df["Take_Profit"] = [None] + take_profits
    df["Pred_Gain"] = [None] + pred_gains
    return df

def generate_trade_signals_backtest_fromdfB(df,symbol):
    #print("Here2")
    #print(start_date)
    #df = calculate_indicators_histall(symbol,200,daily,assettype,start_date)
    #df = calculate_indicators(df)
    
    #df2=get_stock_data_days(symbol,200)
    #print(df.iloc[0:10])

    indicators = calculate_indicators_hist(symbol)
    signals = []
    stop_losses=[]
    take_profits=[]
    pred_gains=[]
    
    for i in range(1, len(df)):
        price = df["close"].iloc[i]
        pivot = df["Pivot_Point"].iloc[i]
        ma_5 = df["5_MA"].iloc[i]
        ma_20 = df["20_MA"].iloc[i]
        ma_50 = df["50_MA"].iloc[i]
        ma_120 = df["120_MA"].iloc[i]
        latest=df.iloc[i,:]
        prev=df.iloc[i-1,:]
        # Calculate stop-loss and take-profit levels
        stop_loss = price * (1 - RISK_TOLERANCE)
        take_profit = price * (1 + REWARD_THRESHOLD)
        signal=get_overall_signal()(latest,prev,indicators)
        pred_gain=None #pred_gains_fromdf(df.iloc[1:i])
        if(i>10):
          health_status=evaluate_df_health(df.iloc[1:(i-1)])
        else:
          health_status=True
        if health_status:
          signals.append(signal)
          pred_gains.append(pred_gain)
          if signal=="BUY" or signal=="SELL":
              stop_losses.append(stop_loss)
              take_profits.append(take_profit)
          else:
              stop_losses.append(None)
              take_profits.append(None)
        else:
          print("Poor health")
          signals.append("HOLD-Poor")
          stop_losses.append(None)
          take_profits.append(None)
          pred_gains.append(pred_gain)
        #if price<df["close"].iloc[i-1] and price < pivot and price < ma_5 and price<ma_20 and price<ma_50 and :
        #    signals.append("BUY")
        #elif price > pivot and price > ma_50 and price>ma_5:
        #    signals.append("SELL")
        #else:
        #    signals.append("HOLD")
    df["Signal"] = ["HOLD"] + signals  # Align with data
    df["Stop_Loss"] = [None] + stop_losses
    df["Take_Profit"] = [None] + take_profits
    df["Pred_Gain"] = [None] + pred_gains
    return df

# Generate trade signals
def generate_trade_signals_backtest(symbol,assettype="stocks",daily=False,start_date=None):
    #print("Here2")
    print(start_date)
    df = calculate_indicators_histall(symbol,200,daily,assettype,start_date)
    #df = calculate_indicators(df)
    
    #df2=get_stock_data_days(symbol,200)
    #print(df.iloc[0:10])

    indicators = calculate_indicators_hist(symbol)
    signals = []
    stop_losses=[]
    take_profits=[]
    pred_gains=[]
    
    for i in range(1, len(df)):
        price = df["close"].iloc[i]
        pivot = df["Pivot_Point"].iloc[i]
        ma_5 = df["5_MA"].iloc[i]
        ma_20 = df["20_MA"].iloc[i]
        ma_50 = df["50_MA"].iloc[i]
        ma_120 = df["120_MA"].iloc[i]
        latest=df.iloc[i,:]
        prev=df.iloc[i-1,:]
        # Calculate stop-loss and take-profit levels
        stop_loss = price * (1 - RISK_TOLERANCE)
        take_profit = price * (1 + REWARD_THRESHOLD)
        signal=get_signal(latest,prev,indicators)
        pred_gain=None #pred_gains_fromdf(df.iloc[1:i])
        if(i>10):
          health_status=evaluate_df_health(df.iloc[1:(i-1)])
        else:
          health_status=True
        if health_status:
          signals.append(signal)
          pred_gains.append(pred_gain)
          if signal=="BUY" or signal=="SELL":
              stop_losses.append(stop_loss)
              take_profits.append(take_profit)
          else:
              stop_losses.append(None)
              take_profits.append(None)
        else:
          print("Poor health")
          signals.append("HOLD-Poor")
          stop_losses.append(None)
          take_profits.append(None)
          pred_gains.append(pred_gain)
        #if price<df["close"].iloc[i-1] and price < pivot and price < ma_5 and price<ma_20 and price<ma_50 and :
        #    signals.append("BUY")
        #elif price > pivot and price > ma_50 and price>ma_5:
        #    signals.append("SELL")
        #else:
        #    signals.append("HOLD")
    df["Signal"] = ["HOLD"] + signals  # Align with data
    df["Stop_Loss"] = [None] + stop_losses
    df["Take_Profit"] = [None] + take_profits
    df["Pred_Gain"] = [None] + pred_gains
    return df

def backtest_strategy_riskreward_dfA(symbol,df,initial_balance=250):
    #df = get_stock_data_days(symbol,200) #get_minute_data(symbol)
    #df = calculate_indicators(df)
        # Evaluate stock health before trading
    
    #initial_balance = 5000
    balance = initial_balance
    position = 0
    trade_history = []
    df=pd.DataFrame(df)
    df.dropna()
    for i in range(1, len(df)):
        signal = df["Signal"].iloc[i]
        price = df["close"].iloc[i]
        stop_loss = df["Stop_Loss"].iloc[i]
        take_profit = df["Take_Profit"].iloc[i]
        #print(signal)
        # Extract the last 10 rows
        last_10_rows = df.tail(10)
        #last_10_rows["index"]=range(1,11)
        #print(last_10_rows)
        # Calculate the correlation between 'col1' and 'col2' in the last 10 rows
        #correlation = last_10_rows['close'].corr(last_10_rows["index"])
        #print(correlation)
        # and correlation>(-0.3)
        print("stop loss")
        print(i)
        #print(stop_loss)
        #print(price)
        if signal == "BUY" and balance >= price:
            print(balance)
            position = balance // price
            balance -= position * price
            trade_history.append(("BUY", price, position, balance))

        elif signal == "SELL" and position > 0:
            #if price >= take_profit:
            balance += position * price
            trade_history.append(("SELL", price, position, balance))
            position = 0

        # Apply Stop-Loss and Take-Profit
        if position > 0:
            if price <= df["Stop_Loss"].iloc[i]:  # Stop-Loss Triggered
                balance += position * df["Stop_Loss"].iloc[i]
                trade_history.append(("STOP-LOSS", df["Stop_Loss"].iloc[i], position, balance))
                position = 0

            elif price >= take_profit:  # Take-Profit Triggered
                balance += position * take_profit
                trade_history.append(("TAKE-PROFIT", take_profit, position, balance))
                position = 0
    print(balance)

    final_balance = balance + (position * df["close"].iloc[-1])
    total_pnl = final_balance - initial_balance
    win_trades = sum(1 for t in trade_history if t[0] in ["SELL", "TAKE-PROFIT"])

    total_trades = len(trade_history) // 2  # Buy/Sell pairs
    win_rate = (win_trades / total_trades) * 100 if total_trades > 0 else 0
    
    return {
        "symbol": symbol,
        "final_balance": final_balance,
        "total_pnl": round(total_pnl, 2),
        "win_rate": round(win_rate, 2),
        "trade_history":trade_history,
        "df":df
    }

def backtest_strategy_riskreward_symbol(symbol,startdate=date.today(),initial_balance=250,score_threshold=7,avoid_score_thresh=5,decline_score_thresh=3):
    #df = get_stock_data_days(symbol,200) #get_minute_data(symbol)
    #df = calculate_indicators(df)
        # Evaluate stock health before trading
    df=compute_multicriteria_scores(symbol,startdate)
    df=generate_trade_signals_backtest_fromscores(df,symbol,score_threshold,avoid_score_thresh,decline_score_thresh)
    print(df.head())
    #initial_balance = 5000
    balance = initial_balance
    position = 0
    trade_history = []
    #df=pd.DataFrame(df)
    #df.dropna()
    for i in range(1, len(df)):
        signal = df["Signal"].iloc[i]
        price = df["close"].iloc[i]
        stop_loss = df["Stop_Loss"].iloc[i]
        take_profit = df["Take_Profit"].iloc[i]
        #print(signal)
        # Extract the last 10 rows
        last_10_rows = df.tail(10)
        #last_10_rows["index"]=range(1,11)
        #print(last_10_rows)
        # Calculate the correlation between 'col1' and 'col2' in the last 10 rows
        #correlation = last_10_rows['close'].corr(last_10_rows["index"])
        #print(correlation)
        # and correlation>(-0.3)
        #print("stop loss")
        #print(i)
        #print(stop_loss)
        #print(price)
        if signal == "BUY" and balance >= price:
            print(balance)
            position = balance // price
            balance -= position * price
            trade_history.append(("BUY", price, position, balance))

        elif signal == "SELL" and position > 0:
            #if price >= take_profit:
            balance += position * price
            trade_history.append(("SELL", price, position, balance))
            position = 0

        # Apply Stop-Loss and Take-Profit
        if position > 0:
            if price <= df["Stop_Loss"].iloc[i]:  # Stop-Loss Triggered
                balance += position * df["Stop_Loss"].iloc[i]
                trade_history.append(("STOP-LOSS", df["Stop_Loss"].iloc[i], position, balance))
                position = 0

            elif price >= take_profit:  # Take-Profit Triggered
                balance += position * take_profit
                trade_history.append(("TAKE-PROFIT", take_profit, position, balance))
                position = 0
    print(balance)

    final_balance = balance + (position * df["close"].iloc[-1])
    total_pnl = final_balance - initial_balance
    win_trades = sum(1 for t in trade_history if t[0] in ["SELL", "TAKE-PROFIT"])

    total_trades = len(trade_history) // 2  # Buy/Sell pairs
    win_rate = (win_trades / total_trades) * 100 if total_trades > 0 else 0
    
    return {
        "symbol": symbol,
        "final_balance": final_balance,
        "total_pnl": round(total_pnl, 2),
        "win_rate": round(win_rate, 2),
        "trade_history":trade_history,
        "df":df
    }


# Backtesting function with risk-reward implementation
def backtest_strategy_riskreward(symbol,assettype="stocks",daily=False,start_date=None,initial_balance=250):
    #df = get_stock_data_days(symbol,200) #get_minute_data(symbol)
    #df = calculate_indicators(df)
        # Evaluate stock health before trading
    print("HEre")
    print(assettype)
    if not evaluate_stock_health(symbol,start_date,assettype):
        print(f"Skipping {symbol}: Poor stock health")
        #return None
    
    df = generate_trade_signals_backtest(symbol,assettype,daily,start_date)
    
    #initial_balance = 5000
    balance = initial_balance
    position = 0
    trade_history = []

    for i in range(1, len(df)):
        signal = df["Signal"].iloc[i]
        price = df["close"].iloc[i]
        stop_loss = df["Stop_Loss"].iloc[i]
        take_profit = df["Take_Profit"].iloc[i]
        #print(signal)
        # Extract the last 10 rows
        last_10_rows = df.tail(10)
        #last_10_rows["index"]=range(1,11)
        #print(last_10_rows)
        # Calculate the correlation between 'col1' and 'col2' in the last 10 rows
        #correlation = last_10_rows['close'].corr(last_10_rows["index"])
        #print(correlation)
        # and correlation>(-0.3)
        if signal == "BUY" and balance >= price:
            print(balance)
            position = balance // price
            balance -= position * price
            trade_history.append(("BUY", price, position, balance))

        elif signal == "SELL" and position > 0:
            #if price >= take_profit:
            balance += position * price
            trade_history.append(("SELL", price, position, balance))
            position = 0

        # Apply Stop-Loss and Take-Profit
        if position > 0:
            if price <= stop_loss:  # Stop-Loss Triggered
                balance += position * stop_loss
                trade_history.append(("STOP-LOSS", stop_loss, position, balance))
                position = 0

            elif price >= take_profit:  # Take-Profit Triggered
                balance += position * take_profit
                trade_history.append(("TAKE-PROFIT", take_profit, position, balance))
                position = 0
    print(balance)

    final_balance = balance + (position * df["close"].iloc[-1])
    total_pnl = final_balance - initial_balance
    win_trades = sum(1 for t in trade_history if t[0] in ["SELL", "TAKE-PROFIT"])

    total_trades = len(trade_history) // 2  # Buy/Sell pairs
    win_rate = (win_trades / total_trades) * 100 if total_trades > 0 else 0
    
    return {
        "symbol": symbol,
        "final_balance": final_balance,
        "total_pnl": round(total_pnl, 2),
        "win_rate": round(win_rate, 2),
        "trade_history":trade_history,
        "df":df
    }


# Backtesting function
def backtest_strategy(symbol,assettype="stocks"):
    #df = get_stock_data_days(symbol,200)

    #df = generate_trade_signals_backtest(symbol)
    if not evaluate_stock_health(symbol,assettype):
        print(f"Skipping {symbol}: Poor stock health")
        #return None
    
    df = generate_trade_signals_backtest(symbol,assettype)
    
    print(df)
    initial_balance = 10000
    balance = initial_balance
    position = 0
    trade_history = []
    

    for i in range(1, len(df)):
        signal = df["Signal"].iloc[i]
        price = df["close"].iloc[i]

        if signal == "BUY" and balance >= price:
            position = balance // price
            balance -= position * price
            trade_history.append(("BUY", price, position, balance))

        elif signal == "SELL" and position > 0:
            balance += position * price
            trade_history.append(("SELL", price, position, balance))
            position = 0

    final_balance = balance + (position * df["close"].iloc[-1])
    total_pnl = final_balance - initial_balance
    win_trades = sum(1 for t in trade_history if t[0] == "SELL" and t[2] > 0)
    total_trades = len(trade_history) // 2  # Buy/Sell pairs
    win_rate = (win_trades / total_trades) * 100 if total_trades > 0 else 0

    # Calculate max drawdown
    peak = initial_balance
    max_drawdown = 0
    for _, _, _, bal in trade_history:
        if bal > peak:
            peak = bal
        drawdown = (peak - bal) / peak
        max_drawdown = max(max_drawdown, drawdown)

    return {
        "symbol": symbol,
        "total_pnl": round(total_pnl, 2),
        "win_rate": round(win_rate, 2),
        "max_drawdown": round(max_drawdown * 100, 2)
    }
def monitor_and_adjust_sell_order(SYMBOL,buy_price, sell_order,quantity,stoplimit_threshold=1):
    """Monitors the price and adjusts the sell order accordingly."""
    #while True:
    current_price = get_latest_price_alpaca(SYMBOL)
    loss_price=stoplimit_threshold*buy_price*0.01


    price_increase_percent = ((current_price - buy_price) / buy_price) * 100

    if price_increase_percent > LIMIT_PERCENT:
        # Replace limit sell order with trailing stop order
        replace_with_trailing_stop_order(sell_order.id, SYMBOL,quantity)
        #break  # Exit loop after placing trailing stop order
    else:
        if price_increase_percent < buy_price and price_increase_percent>stoplimit_threshold:
            #replace_with_trailing_stop_order(sell_order.id, SYMBOL,quantity)
            place_limit_sell_order(symbol, profit_price, loss_price,quantity)
            #break
        else:
            place_limit_sell_order(symbol, profit_price, current_price,quantity)
            #break
        print(f"Current price {current_price} is within {LIMIT_PERCENT}% of buy price.")

        #time.sleep(CHECK_INTERVAL)
            
# Function to fetch 5-minute candlestick data
def get_stock_data(symbol):
    today = datetime.now(ZoneInfo("America/New_York")) #datetime.date.today()
    time.sleep(0.1)
    # Request 5-minute interval price data from market open
    request_params = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Minute,
        #start=today,
        adjustment="raw"
    )
    bars = historical_client.get_stock_bars(request_params)
    
    # Convert to DataFrame
    df = bars.df.loc[symbol].reset_index()
    #df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    df["timestamp"] = df["timestamp"].dt.tz_convert('America/New_York')
    return df


def eval_days_momentum(symbol):
    # Get today's date
    #
    today = datetime.now(ZoneInfo("America/New_York")) #datetime.date.today()

    # Request 5-minute interval price data from market open
    request_params = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Minute,
        #start=today,
        adjustment="raw"
    )
    bars = historical_client.get_stock_bars(request_params)

    # Convert to DataFrame
    df = bars.df.loc[symbol].reset_index()
    df["timestamp"] = df["timestamp"].dt.tz_convert('America/New_York') #tz_localize(None)  # Remove timezone for ease of use

    # Calculate Moving Averages
    df["SMA_20"] = ta.trend.sma_indicator(df["close"], window=20)
    df["EMA_20"] = ta.trend.ema_indicator(df["close"], window=20)

    # Calculate Momentum Indicators
    df["RSI_14"] = ta.momentum.rsi(df["close"], window=14)
    df["fastK"] = ta.momentum.stoch(df["high"], df["low"], df["close"], window=14, smooth_window=3)
    df["fastD"] = ta.momentum.stoch_signal(df["high"], df["low"], df["close"], window=14, smooth_window=3)
    df["ROC_10"] = ta.momentum.roc(df["close"], window=10)  # Rate of Change
    df["Momentum_10"] = df["close"].diff(10)  # Simple Momentum (Price Diff)

    # Show last few rows (latest data)
    print(df.tail())
    return(df)

def get_overall_signalv0409(symbol,assettype="stocks"):
     buy_signal=None
     sell_signal=None
     overall_signal=None
     df=pd.DataFrame()
     try:
        trade_signal=get_trade_signal_daily(symbol)
        stock_health=evaluate_stock_health(symbol,False)
        stock_health_daily=evaluate_stock_health(symbol,True)
        stock_trend=evaluate_stock_trend_daily_etfs(symbol)
        stock_score_long1, stock_score_long2=get_stock_score(symbol)
        stock_score_day1, stock_score_day2=get_stock_score(symbol,"Minute")
        gain_match=int(bool(re.search(r'\b:1\b', trade_signal)))
        buy_match=int(bool(re.search(r'\bBUY\b', trade_signal)))
        sell_match=int(bool(re.search(r'\bSELL\b', trade_signal)))
        down_match=int(bool(re.search(r'\b:down\b', trade_signal)))
        buy_signal=(
          (stock_score_day1<3 and stock_score_day1>1 and stock_score_long1>3 and stock_trend and stock_health and buy_match>0 and stock_score_day1>=stock_score_day2) or
        (stock_score_day1<3 and stock_score_day1>1 and stock_score_long1>3 and stock_trend and stock_health and gain_match>0 and stock_score_day1>=stock_score_day2) or
        (stock_score_day1<5 and (buy_match>0 or gain_match>0 or stock_score_day1>stock_score_day2) and stock_score_day1>1 and stock_score_long1>stock_score_long2) or
        (stock_score_day1<=6 and stock_score_day1>3 and stock_score_day2>3 and stock_score_long1>stock_score_long2 and stock_trend and stock_score_long1<2 and stock_score_day1>=stock_score_day2) or
        (stock_score_day1<6 and stock_score_day1>3 and stock_score_day2>3 and stock_score_long1<2 and stock_score_long1>stock_score_long2 and stock_score_day1>=stock_score_day2)
       # (stock_score_day1<3 and stock_score_day1>0 and stock_score_long1<3 and stock_trend and stock_score_day1<stock_score_day2 and stock_score_day2>2) or 
        #((stock_score_day1<1 and stock_score_day1<=stock_score_day2 and stock_score_long1<2 and stock_score_long1>0 and stock_trend==False and down_match>0))
        )

        
        sell_signal=(
          (stock_score_long1>4 and stock_score_day1>5 and sell_match>0) or (stock_score_long1>5 and stock_score_day1>5)
          )
        
        overall_signal="BUY" if buy_signal else "SELL" if sell_signal else "HOLD"
        
        dict1={'Symbol':symbol, 'Last':get_latest_price_alpaca(symbol),'Overall.Signal':overall_signal,'Technical.Indicators.Signal':trade_signal,
        'stock_trend':stock_trend,'stock_health':stock_health,'stock_score_long1':stock_score_long1,'stock_score_long2':stock_score_long2,
        'stock_score_current':stock_score_day1,'stock_score_prev':stock_score_day2,'gain_match':gain_match,'buy_match':buy_match,
        'sell_match':sell_match}
       
        df=pd.DataFrame(dict1,index=[0])
        #df.index = ['0']
     except Exception as e:
        print(e)
     
     return overall_signal,df

# --- Indicator Calculation ---
def compute_indicators(df):
    df['sma_5'] = df["close"].rolling(window=5).mean()
    df['sma_10'] = df["close"].rolling(window=10).mean()
    df['sma_20'] = df["close"].rolling(window=20).mean()
    df['sma_50'] = df["close"].rolling(window=50).mean()
    df['sma_120'] = df["close"].rolling(window=120).mean()
    df['daily_90thpercentile'] = df['close'].quantile(0.9)
    df['daily_50thpercentile'] = df['close'].quantile(0.5)
    df['daily_25thpercentile'] = df['close'].quantile(0.25)
    df['daily_10thpercentile'] = df['close'].quantile(0.1)
    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
    df['rsi_5'] = df['rsi'].rolling(5).mean()
    stoch = StochasticOscillator(df['high'], df['low'], df['close'], window=14)
    df['fastk'] = stoch.stoch()
    df['fastd'] = stoch.stoch_signal()
    df['fastk_5'] = df['fastk'].rolling(5).mean()
    df['roc'] = ROCIndicator(df['close'], window=12).roc()
    df['roc_5']=df['roc'].rolling(window=5).mean()
    df['roc_prev'] = df['roc'].shift(1)
    df['momentum'] = df['close'].diff(10)
    df['momentum_120'] = df['close'].diff(120)
    df['20_STD'] = df['close'].rolling(window=20).std()
    df['upper_band'] = df['sma_20'] + (2 * df['20_STD'])
    df['lower_band'] = df['sma_20'] - (2 * df['20_STD'])
    df['bb_width'] = 100*(df['upper_band']-df['lower_band'])/df['sma_20']  #BollingerBands(df['close']).bollinger_wband()
    df['volume_change'] = df['volume'].pct_change(periods=5) * 100
    df['avg_vol_10'] = df['volume'].rolling(10).mean()
    df['volume_change_ratio'] = df['volume'] / df['avg_vol_10']
    #bb = BollingerBands(df['close'], window=20, window_dev=2)
    #df['bb_width'] = bb.bollinger_hband() - bb.bollinger_lband()
    df['bb_width_5'] = df['bb_width'].rolling(5).mean()
    
    return df

# --- Recovery Detection Score ---
def generate_recovery_score(df):
    df['recovery_score'] = 0
    df['recovery_score'] += ((df['rsi'] > 30) & (df['rsi'] > df['rsi_5'])).astype(int)
    df['recovery_score'] += ((df['fastk'] < 30) & (df['fastk'] > df['fastd'])).astype(int)
    df['recovery_score'] += ((df['roc'] > 0) & (df['roc_prev'] < 0)).astype(int)
    df['recovery_score'] += (df['momentum'] > 0).astype(int)
    df['recovery_score'] += (df['bb_width'] > df['bb_width_5']).astype(int)
    df['recovery_score'] += (df['fastk'] > df['fastk_5']).astype(int)
    df['recovery_score'] += (df['close'] > df['sma_5']).astype(int)
    df['recovery_score'] += (df['close'] > df['sma_10']).astype(int)
    return df

# --- Momentum Detection Score ---
def generate_momentum_score(df):
    df['momentum_score'] = 0
    df['momentum_score'] += ((df['rsi'] > 50) & (df['rsi'] > df['rsi_5'])).astype(int)
    df['momentum_score'] += ((df['fastk'] > df['fastd']) & (df['fastk'] > 50)).astype(int)
    df['momentum_score'] += (df['roc'] > 0).astype(int)
    df['momentum_score'] += (df['momentum'] > 0).astype(int)
    df['momentum_score'] += (df['bb_width'] > df['bb_width_5']).astype(int)
    df['momentum_score'] += (df['fastk'] > df['fastk_5']).astype(int)
    df['momentum_score'] += (df['close'] > df['sma_5']).astype(int)
    df['momentum_score'] += ((df['close'] > df['sma_10']) & (df['sma_10'] > df['sma_20'])).astype(int)
    return df


# Recovery and momentum scoring
def generate_combined_buy_score(df):
  
    #max recovery: 8
    df['recovery_score'] = 0
    df['recovery_score'] += ((df['rsi'] > 30) & (df['rsi'] > df['rsi_5'])).astype(int)
    df['recovery_score'] += ((df['fastk'] > 20) & (df['fastk'] > df['fastk'].shift(1)) & (df['fastk'] > df['fastd']) & (df['fastk'] > df['fastk_5']) & (df['fastk_5'] <30)).astype(int)
    df['recovery_score'] += ((df['roc'] > df['roc_prev']) & (df['roc_prev'] < 0)).astype(int)
    df['recovery_score'] += ((0 > df['momentum'].shift(1)) & (df['momentum'] >=0)).astype(int)
    df['recovery_score'] += (df['bb_width'] > df['bb_width_5']).astype(int)
    df['recovery_score'] += (df['volume_change'] > df['volume_change'].shift(1)).astype(int)
    df['recovery_score'] += (df['close'] > df['sma_5']).astype(int)
    df['recovery_score'] += ((df['close'] > df['close'].shift(1)) & (df['close'] > df['close'].shift(2))).astype(int)

    df['momentum_score'] = 0
    df['momentum_score'] += ((df['rsi'] > 40) & (df['rsi'] <55) & (df['rsi_5'] > 40) & (df['rsi'] > df['rsi_5'])).astype(int)
    df['momentum_score'] += ((df['fastk'] > df['fastd']) & (df['fastk'] > 50) & (df['fastk_5'] <55) & (df['fastd'] <70) & (df['fastk'] < 70)).astype(int)
    df['momentum_score'] += ((df['roc'] > -0.5) & (df['roc'] > df['roc_5'])).astype(int)
    df['momentum_score'] += (df['momentum'] >= 0).astype(int)
    df['momentum_score'] += (df['bb_width'] > df['bb_width_5']).astype(int)
    df['momentum_score'] += (df['fastk'] > df['fastk_5']).astype(int)
    df['momentum_score'] += (df['volume_change'] > df['volume_change'].shift(1)).astype(int)
    df['momentum_score'] += ((df['close'] > df['sma_5']) & (df['close'] > df['sma_10']) & (df['sma_10'] > df['sma_20'])).astype(int)

    df['score'] = df[['recovery_score', 'momentum_score']].max(axis=1)
    return df
    
def generate_sell_score(df):
    df['avoid_score'] = 0
    df['avoid_score'] += ((df['rsi'] > 70)).astype(int)
    df['avoid_score'] += ((df['fastk'] > df['fastd']) & (df['fastk'] > 70)).astype(int)
    df['avoid_score'] += ((df['roc'] > 5)).astype(int)
    df['avoid_score'] += ((df['momentum'] > df['momentum'].rolling(10).mean())).astype(int)
    df['avoid_score'] += ((df['bb_width'] < df['bb_width_5'])).astype(int)
    df['avoid_score'] += ((df['close'] > df['sma_5']) & (df['close'] > df['sma_10'] * 1.02)).astype(int)
    return df

#max 7; below 5
def generate_early_decline_warning_score(df):
    df['decline_score'] = 0
    df['decline_score'] += ((df['rsi'] < 40) & (df['rsi'] < df['rsi_5'])).astype(int)
    df['decline_score'] += ((df['fastk'] < df['fastd']) & (df['fastk'] < 30)).astype(int)
    df['decline_score'] += (df['roc'] < -1).astype(int)
    df['decline_score'] += ((df['momentum'] < 0) & (df['momentum'] < df['momentum'].rolling(5).mean())).astype(int)
    df['decline_score'] += ((df['bb_width'] > df['bb_width_5']) & (df['close'] < df['close'].shift(1))).astype(int)
    df['decline_score'] += ((df['close'] < df['sma_5']) | (df['close'] < df['sma_10'])).astype(int)
    df['decline_score'] += (df['fastk'] < df['fastk_5']).astype(int)
    return df

def generate_buy_score(df):
    df['buy_score'] = 0
    df['buy_score'] += ((df['close'] < df['sma_20'])).astype(int)
    df['buy_score'] += ((df['close'] < df['sma_50'])).astype(int)
    df['buy_score'] += ((df['sma_20'] > df['sma_120'])).astype(int)
    df['buy_score'] += ((df['volume_change']>0)).astype(int)
    df['buy_score'] += ((df['momentum'] > 0)).astype(int)
    df['buy_score'] -= ((df['momentum'] < 0)).astype(int)
    df['buy_score'] += ((df['roc']>0)).astype(int)
    df['buy_score'] += ((df['volume_change'] >= 75)).astype(int)
    df['buy_score'] -= ((df['volume_change'] < 0)).astype(int)
    df['buy_score'] -= ((df['roc'] > 2)).astype(int)
    df['buy_score'] += ((df['bb_width'] < df['bb_width_5'])).astype(int)
    df['buy_score'] += ((df['bb_width'] > df['bb_width_5'])).astype(int)
    df['buy_score'] -= ((df['bb_width'] > 2*df['bb_width_5'])).astype(int)
    
        
    return df
#	8-10: Strong buy potential: High-confidence entry
#3 from price; 2 from momentum; 2 from volume; 3 from BB width
def calculate_buy_score(rec):
    score = 0
    
    rec=rec.iloc[-1]
    
    # Price relative to average
    if (rec['close'] < rec['sma_20']):
        score += 1
    if (rec['close'] < rec['sma_50']):
        score += 1
    if rec['sma_20'] > rec['sma_120']:
        score += 1

    # Momentum and Rate of Change
    if 0.2 < rec['roc'] < 1.5:
        score += 1
    if rec['momentum'] > 0:
        score += 1
    elif rec['momentum'] < 0:
        score -= 1

    # Volume
    if 0 < rec['volume_change'] < 100:
        score += 1
    elif rec['volume_change'] >= 100:
        score += 1
    elif rec['volume_change'] < 0:
        score -= 1

    # Caution flag on excessive ROC
    if rec['roc'] > 2:
        score -= 1
    
    # Bollinger Band Width (BBW) interpretation
    if rec['bb_width'] < rec['bb_width_5']:
        score += 1  # squeeze setup
    if rec['bb_width'] > rec['bb_width_5']:
        score += 1  # confirmed breakout
    if rec['bb_width'] > 2 * rec['bb_width_5'] and rec['volume_change'] < 0:
        score -= 1  # overextended + fading interest
    return score
  
def generate_momentum_buy_signal(df):
    df['buy_signal'] = df.apply(lambda row: (
         (row["rsi"] <= 30 and row["rsi_5"] > 40 and
        row["fastk"] > row["fastd"] and 0<row["fastk"]<20 and row["fastd"] > 0 and
        row["fastk_5"] >30 and
        row["roc_5"] >(-0.1) and
        row["bb_width"] > 0.04 and
        row["momentum"] > 0 and
        row["momentum_120"] > 5 and
        row["roc"] > 0 and
        row["volume_change_ratio"] > 1.2 and
        row["close"] >= row["vwap"]) or
        (45 <= row["rsi"] <= 65 and
        row["fastk"] > row["fastd"] and 50<row["fastk"]<80 and row["fastd"] > 50 and
        row["bb_width"] > 0.04 and
        row["momentum"] > 0 and
        row["roc"] > 0 and
        row["volume_change_ratio"] > 1.2 and
        row["close"] >= row["vwap"]) or 
        
        (45 <= row["rsi"] <= 65 and row["rsi"]>row["rsi_5"] and row["fastk"] > row["fastk_5"] and
        row["fastk"] > row["fastd"] and row["fastk"] > 40 and row["fastd"] < 25 and row["fastk_5"] < 25
        and row["bb_width"] > 1 and
        row["momentum"] < (-1) and
        row["roc"] <(-1) and
        row["volume_change_ratio"] > 1.5 and
        row["close"] >= row["vwap"] and row["close"]>row["open"] and row["close"]<row["high"] and row["recovery_score"]>4) or
        ((45 <= row["rsi"] <= 65 and row["rsi"]>row["rsi_5"] and row["fastk"] > row["fastk_5"] and
        row["fastk"] > row["fastd"] and row["fastk"] > 40 and row["fastd"] < 25 and row["fastk_5"] < 25
        and row["bb_width"] > 1 and
        row["momentum"] < (-1) and
        row["roc"] <(-1) and
        row["volume_change_ratio"] <1 and
        row["close"] >= row["vwap"] and row["close"]>row["open"] and row["close"]<row["high"] and row["recovery_score"]>4))
    ), axis=1) 
    return df

def generate_goodmomentum_decline_buy_signal(df):
    df['pricedrop_signal'] = df.apply(lambda row: (
        (row["rsi"] <= 30 and row["rsi_5"] <30 and
        row["fastk"] < row["fastd"] and 0<row["fastk"]<20 and row["fastd"]<20 and row["fastk_5"] > row["fastk"] and
        row["roc_5"] >(-0.1) and
        row["bb_width"] > 10 and
        row["momentum"] > (-0.5) and
        row["momentum_120"] > (-0.1) and
        row["roc"] <(-5) and row['volume']>1000 and
        row["volume_change"] > 100 and
        row["close"] < row["vwap"] and row["close"] < row["sma_5"] and row["close"] < row["sma_20"] and row["sma_50"]>row["sma_20"] and row["sma_50"]>row["sma_120"]) 
    ), axis=1) 
    return df

def generate_recovery_signal(df):
    df['recovery_signal'] = df.apply(lambda row: (
        (row["rsi"] <= 50 and row["rsi_5"] <50 and
        row["fastk"] < row["fastd"] and 0<row["fastk"]<40 and row["fastd"]<40 and row["fastk_5"] < row["fastk"] and
        row["roc_5"] >(-1) and
        row["bb_width"] > 0.4 and
        row["momentum"] > (0) and
        row["momentum_120"] > (1) and
        row["roc"] <(0) and row['volume']>1000 and
        row["volume_change"] > 10 and row["volume_change_ratio"] > 1 and
        row["close"] < row["vwap"] and row["close"] > row["sma_5"] and row["close"] < row["sma_20"] and row["sma_50"]>row["sma_120"]) 
    ), axis=1) 
    return df

def generate_rally_signal(df):
    df['rally_signal'] = df.apply(lambda row: (
        (row["avoid_score"]>4 
        and row["recovery_score"]>3 and row["momentum_score"]>4 and row["buy_score"]>2 and row["rsi"]>50 and row["rsi_5"]>50 and
        row["fastk"] > row["fastd"] and row["fastk"]>80 and row["fastd"]>80 and row["fastk_5"] < row["fastk"]
        and row["roc"] >(5) and
        row["bb_width"] > 5 and
        row["momentum"] > (0) and
        row["roc"]>row["roc_5"] and row['volume']>10000 and
        row["volume_change"] > 5 and row["volume_change_ratio"] > 1 and
        row["close"] > row["vwap"] and row["close"] > row["sma_5"] and row["sma_5"]>row["sma_10"] and row["sma_10"]>row["sma_20"]
        ) 
    ), axis=1) 
    return df
  
def get_multicriteria_score(symbol,score_threshold=7,avoid_score_thresh=5,decline_score_thresh=3):
    df = get_stock_data_date(symbol,date.today()) #get_minute_data(ticker)
    
    df = compute_indicators(df)
    #df.dropna()
    df = generate_combined_buy_score(df)
    df=generate_sell_score(df)
    df=generate_early_decline_warning_score(df)
    df=generate_buy_score(df)
    df=generate_momentum_buy_signal(df)
    df=generate_goodmomentum_decline_buy_signal(df)
    df=generate_recovery_signal(df)
    df=generate_rally_signal(df)
    #score = df['score'].iloc[-1]
    #decline_score = df['decline_score'].iloc[-1]
    #avoid_score=df['avoid_score'].iloc[-1]
    
    #buy_score=df['buy_score'].iloc[-1] #calculate_buy_score(df)
    
    latest=df.iloc[-1]
    #latest["Technical.Indicators.Signal"]=get_trade_signal_daily(symbol)
    #df.loc[df.index[-1], "Technical.Indicators.Signal"]=latest["Technical.Indicators.Signal"]
    #pred_gain=pred_gains_fromdfv2(latest)
    #or (buy_score>8)
    buy_signal= ((
        #(1 < latest["recovery_score"] < 3 and 1 < latest["momentum_score"] <3 and latest["avoid_score"]<1 and latest["decline_score"]<6 and latest["decline_score"]>6
        #and latest["buy_score"]>3 and latest["buy_score"]<6 and latest["rsi"]<30 and latest["fastk"]<5 and latest["volume_change"]>100) or
        #(3<latest["momentum_score"]<5 and 5 < latest["recovery_score"] < 7 and latest["avoid_score"]>0 and 2<latest["avoid_score"]<4 and latest["decline_score"]<1 and latest["buy_score"]>4 and latest["volume_change"]>100) or
        #(0<latest["momentum_score"]<2 and 0 < latest["recovery_score"] < 2 and latest["avoid_score"]==0 and 3<latest["decline_score"]<5 and 3<latest["buy_score"]<5 and latest["volume_change"]>300) or
        (latest["fastk"]>75 and latest["fastk_5"]<50 and latest["fastd"]<50 and latest["close"]>latest["sma_5"] and latest["close"]>latest["sma_20"] and latest["sma_20"]>latest["sma_5"]
        and latest["sma_5"]<latest["sma_10"] and latest["rsi"]>50 and latest["roc"]>0 and latest["roc_5"]<0 and latest["rsi_5"]<50 
        and 4<latest["momentum_score"]<6 and 5 < latest["recovery_score"] < 7 and 1<latest["avoid_score"]<3 and latest["decline_score"]<1 and 2<latest["buy_score"]<5 and latest["volume_change"]>300) or
        (latest["rsi"]<30 and latest["rsi_5"]<30 and latest["fastk"]<5 and latest["fastd"]>5 and latest["bb_width"]<latest["bb_width_5"] and latest["volume_change"]>40
        and 4<latest["decline_score"]<6 and 2<latest["buy_score"]<5 and 1 < latest["recovery_score"] < 3 and 1<latest["avoid_score"]<3 and 0<latest["momentum_score"]<2) or
        ( (50<latest["rsi"]<60 and latest["rsi_5"]<50 and 80<latest["fastk"]<90 and latest["fastk_5"]<55 and 45<latest["fastd"]<60 and latest["bb_width"]>latest["bb_width_5"] 
        and latest["volume_change"]>1000
        and latest["decline_score"]<1 and latest["buy_score"]>5 and latest["recovery_score"] >6 and 1<latest["avoid_score"]<3 and latest["momentum_score"]>4)) or 
        (( (60<latest["rsi"]<70 and 60<latest["rsi_5"]<70 and 70<latest["fastk"]<80 and latest["fastk_5"]>85 and latest["fastk"]<latest["fastd"]<95 and latest["bb_width"]>latest["bb_width_5"] 
        and latest["volume_change"]>(-1) and latest["momentum"]>0 and latest["roc_5"]>0 and latest["roc"]>latest["roc_5"] and latest["roc_prev"]>latest["roc"] and latest["roc_prev"]>latest["roc_5"]
        and latest["close"]<latest["sma_5"] and latest["close"]>latest["sma_10"] and 2<latest["decline_score"]<4 and 1<latest["buy_score"]<3
        and 0<latest["recovery_score"]<2 and 0<latest["avoid_score"]<2 and 2<latest["momentum_score"]<4))) or 
        (50<latest["rsi"]<65 and latest["rsi_5"]<50 and 40<latest["fastk"]<50 and latest["fastk_5"]<50 and 40<latest["fastd"]<50 and latest["bb_width"]>latest["bb_width_5"] 
        and latest["volume_change"]>200  and latest["roc"]>latest["roc_5"]
        and latest["decline_score"]<1 and latest["buy_score"]>5 and latest["recovery_score"] >6 and 0<latest["avoid_score"]<2 and latest["momentum_score"]>4) or
        ((40<latest["rsi"]<50 and latest["rsi_5"]>50 and latest["fastk"]<25 and latest["fastk_5"]>50 and latest["fastd"]>50 and latest["bb_width"]>latest["bb_width_5"] 
        and latest["volume_change"]>1000  and latest["roc"]<latest["roc_5"] and latest["roc"]>(-0.1) and latest["momentum"]>0
        and 2<latest["decline_score"]<5 and latest["buy_score"]>5 and 1<latest["recovery_score"]<3 and latest["avoid_score"]<1 and 2<latest["momentum_score"]<4)) or 
        (latest["buy_signal"] and 4<latest["buy_score"]<6 and latest["recovery_score"]>5 and latest["momentum_score"]>3) 
        or (latest["rsi"]<40 and latest["fastk"]<20 and latest["fastk_5"]>20 and latest["momentum_120"]>0 and latest["momentum"]>(-0.5) and latest["buy_score"]>=3
        and latest["decline_score"]>4 and latest['volume_change_ratio']>1 and 1 < latest["recovery_score"] < 5) 
        or (latest["rsi"]<40 and latest["fastk"]<20 and latest["fastk_5"]<latest["fastk"] and latest["momentum_120"]>0 and latest["momentum_score"]>2 and latest["buy_score"]>=3
        and latest["decline_score"]>3 and latest['volume_change_ratio']>1 and latest["recovery_score"]>1) 
        or (latest["rsi"]<40 and latest["fastk"]<20 and latest["fastk_5"]>latest["fastk"] and latest["momentum_score"]<2 and latest["buy_score"]>=1
        and latest["decline_score"]>4 and latest['volume_change_ratio']>1 and latest["recovery_score"]<2) 
        or (latest["rsi"]<40 and latest["fastk"]<50 and latest["fastk_5"]<latest["fastk"] and latest["buy_score"]>=1
        and latest['volume_change_ratio']>1)
        or (latest["rsi"]>50 and latest["fastk"]>50 and latest["fastk"]<latest["fastk_5"] and 2<latest["buy_score"]<7 and latest["decline_score"]<5 
        and latest["momentum"]>(0) and latest["momentum"]>latest["momentum_120"] and latest["momentum_score"]<4) 
        or (latest["rsi"]>50 and latest["fastk"]>50 and 2<latest["buy_score"]<5 and latest["decline_score"]<5 
        and latest["momentum"]>(0) and latest["momentum"]>latest["momentum_120"] and 2<latest["momentum_score"]<5 and 2<latest["recovery_score"]<5)
        or (latest["rsi"]>50 and latest["fastk"]>50 and 2<latest["buy_score"]<5 and latest["decline_score"]<5 
        and latest["momentum"]>(0) and latest["fastk"]>latest["fastk_5"] and 2<latest["momentum_score"]<5 and 2<latest["recovery_score"]<5)
        or (latest["rsi"]>40 and latest["fastk"]>30 and 2<latest["buy_score"]<4 and latest["decline_score"]<3 
        and latest["momentum"]>(0) and latest["momentum"]<latest["momentum_120"] and latest["fastk"]<latest["fastk_5"] and 3<latest["momentum_score"]<5 and 3<latest["recovery_score"]<6)
        or (latest["close"]<latest["daily_10thpercentile"] and latest["rsi"]<30 and latest["fastk"]<10 and latest["fastd"]<20 and latest["momentum_score"]<2 and latest["recovery_score"]<2)
        or (latest["close"]>latest["daily_90thpercentile"] and 50<latest["rsi"]<65 and latest["fastk"]<30 and latest["fastd"]>30 
        and latest["buy_score"]<2 and latest["avoid_score"]<2 and latest["momentum_score"]<2 and latest["recovery_score"]<2)
        or (latest["close"]>latest["daily_90thpercentile"] and 50<latest["rsi"]<65 and latest["fastk"]>50 
        and latest["buy_score"]<3 and latest["avoid_score"]<3 and latest["momentum_score"]>5 and 2<latest["recovery_score"]<5)
        
        #or (pred_gain>0)
        )) #and (latest["momentum_score"]<6 and latest['volume_change_ratio']>1 and latest['fastk']<70))
    #print(latest["buy_score"])  and 2<latest["momentum_score"]<5 and 2<latest["buy_score"]<6
    #print(latest["fastk"])
    #print(latest["fastk_5"])
    #print(latest["buy_signal"])
    #print(latest)
    buy_signal= buy_signal # or latest["pricedrop_signal"] or latest["recovery_signal"] #(pred_gain>0 and latest["buy_signal"] and 2<latest["buy_score"]<6 and 1<latest["momentum_score"]<5 and latest['volume_change_ratio']>1 and latest['fastk']<80)
    #if (score >= score_threshold and avoid_score<avoid_score_thresh and decline_score<decline_score_thresh and buy_score>4):
    if buy_signal:
        res=True
    else:
        res=False
    return res,df.iloc[-1]

def compute_multicriteria_scores(symbol,startdate=date.today()):
    df = get_stock_data_date(symbol,startdate) #get_minute_data(ticker)
    
    df = compute_indicators(df)
    #df = generate_mi_score(df)
    #df.dropna()
    df = generate_combined_buy_score(df)
    df=generate_sell_score(df)
    df=generate_early_decline_warning_score(df)
    df=generate_buy_score(df)
    df=generate_momentum_buy_signal(df)
    df=generate_goodmomentum_decline_buy_signal(df)
    df=generate_recovery_signal(df)
    df=generate_rally_signal(df)
    return df
 
def get_overall_signal(symbol,assettype="stocks"):
     buy_signal=None
     sell_signal=None
     overall_signal=None
     df=pd.DataFrame()
     try:
        trade_signal=None #get_trade_signal_daily(symbol)
        #stock_health=evaluate_stock_health(symbol,False)
        #stock_health_daily=evaluate_stock_health_daily(symbol)
        stock_trend=None #evaluate_stock_trend_daily_etfs(symbol)
        #stock_score_long1, stock_score_long2=get_stock_score(symbol)
        #stock_score_current, stock_score_prev=get_stock_score(symbol,"Minute")
        stock_score,df=get_multicriteria_score(symbol,7,4,2)
        
        #gain_match=int(bool(re.search(r'\b:1\b', trade_signal)))
        #buy_match=int(bool(re.search(r'\bBUY\b', trade_signal)))
        #sell_match=int(bool(re.search(r'\bSELL\b', trade_signal)))
        #down_match=int(bool(re.search(r'\b:down\b', trade_signal)))
        
        buy_signal=(#(stock_trend and buy_match>0) or 
          (stock_score) 
        
        #or (down_match<1 and stock_score_current>3 and stock_score_current>=stock_score_prev and stock_trend and stock_score_prev>3 
        #and stock_score_long1>=stock_score_long2 and stock_score_long1>3)
       
        )

        
        sell_signal=(
          (df['avoid_score']>5) #or (sell_match>0)
          )
        
        overall_signal="BUY" if buy_signal else "SELL" if sell_signal else "HOLD"
        
        dict1={'Symbol':symbol, 'Last':get_latest_price_alpaca(symbol),'Overall.Signal':overall_signal,'Technical.Indicators.Signal':trade_signal,
        'stock_trend':stock_trend,'stock_score':stock_score}
       
        df=pd.DataFrame(dict1,index=[0])
        #df.index = ['0']
     except Exception as e:
        print(e)
     
     return overall_signal,df
   
# Function to generate trade signals
def get_trade_signal_daily(symbol,assettype="stocks"):
    #print("Checking daily signal for "+symbol)
    signal="HOLD"
    try:
        time.sleep(0.1)
        df = calculate_indicators_histall(symbol,200,True,assettype)
        current_health=evaluate_stock_health(symbol)
        
        if current_health:
          
            
            #df = calculate_indicators(df)
    
            latest = df.iloc[-1]
            prev = df.iloc[-2]  # Previous candle

            #print(df.iloc[-2:,8:])
            #print(latest)
            indicators = calculate_indicators_hist(symbol)
            signal=get_signal(latest,prev,indicators)
            
            p1=pred_gains(symbol,date.today())
            #print("Gain pred: "+str(p1))
            if p1==1:
                signal=signal+":"+str(p1)
            else:
               signal=signal
            #print(signal)
        else:
          #print("Poor health")
          latest = df.iloc[-1]
          prev = df.iloc[-2]  # Previous candle
          indicators = calculate_indicators_hist(symbol)
          signal=get_signal(latest,prev,indicators)
          p1=pred_gains(symbol,date.today())
          #print("Gain pred: "+str(p1))
          if p1==1:
              signal=signal+":"+str(p1)+":down"
          else:
              signal=signal+":down"
          #print(signal)
    except Exception as e:
        print(e)
        signal="None"
        
    return signal #df #return df#"BUY" if buy_signal else "SELL" if sell_signal else "HOLD"
    

# Function to generate trade signals
def get_trade_signal(symbol,assettype="stocks"):
    print("Checking signal for "+symbol)
    if not evaluate_stock_health(symbol,assettype):
        print(f"Skipping {symbol}: Poor stock health")
        return "HOLD"
    df = calculate_indicators_histall(symbol,200,False,assettype)
    #df = calculate_indicators(df)
    indicators = calculate_indicators_hist(symbol)
    latest = df.iloc[-1]
    prev = df.iloc[-2]  # Previous candle

    #print(latest)
    signal=get_signal(latest,prev,indicators)
    
    return signal #"BUY" if buy_signal else "SELL" if sell_signal else "HOLD"


# Function to place trades
def execute_trade(symbol,action, amount_to_invest=500,
                  pct_thresh_buy=5,pct_thresh_sell=5,pct_thresh_stoploss=3,
                  ordertype="bracket",trail_pct=1):  # Default 10 shares
    if action == "BUY":
       # order = MarketOrderRequest(
        #    symbol="SOXL",
       #     qty=qty,
       #     side=OrderSide.BUY,
       #     time_in_force=TimeInForce.DAY
        #)
        
        trade_calls_buy(symbol,last_price=get_latest_price_alpaca(symbol),
                        amount_to_invest=amount_to_invest,pct_thresh_buy=pct_thresh_buy,
                        pct_thresh_sell=pct_thresh_sell,
                        pct_thresh_stoploss=pct_thresh_stoploss,ordertype=ordertype)
        #trading_client.submit_order(order)
        print(f"BUY Order placed for {qty} shares of SOXL")

    elif action == "SELL":
      #  order = MarketOrderRequest(
       #     symbol="SOXL",
       #     qty=qty,
       #     side=OrderSide.SELL,
        #    time_in_force=TimeInForce.DAY
       # )
        trade_calls_sell(symbol,sell_pct_thresh=pct_thresh_sell,trail_pct=trail_pct,
                         stoplimit_pct_thresh=pct_thresh_stoploss,ordertype=ordertype,CHECK_INTERVAL=10)
     #   trading_client.submit_order(order)
        print(f"SELL Order placed for {qty} shares of SOXL")

def live_trading_loop(symbol):
    i=0
    # **Live Trading Loop**
    while i<3:
        print(f"\nChecking market conditions at {datetime.datetime.now()}...")

        # Fetch latest market data
        #df = get_market_data(symbol)

        # Compute indicators
        #df = calculate_indicators(df)

        # Get trade signal
        signal = get_trade_signal(symbol)

        # Execute trade if needed
        if signal in ["BUY", "SELL"]:
            execute_trade(symbol,action=signal, amount_to_invest=500,
                  pct_thresh_buy=5,pct_thresh_sell=5,pct_thresh_stoploss=1,
                  ordertype="bracket",trail_pct=1)
        else:
            print("No trade signal. Holding position.")

        # Wait for the next 5-minute interval
        time.sleep(300)  # 300 seconds = 5 minutes
        i=i+1

##############Options Trading##########################

def calculate_indicators_option(option_data):
    option_data["SMA_50"] = talib.SMA(option_data["close"], timeperiod=50)
    option_data["EMA_20"] = talib.EMA(option_data["close"], timeperiod=20)
    
    option_data["stoch_k"], option_data["stoch_d"] = talib.STOCH(
        option_data["high"], option_data["low"], option_data["close"], 
        fastk_period=14, slowk_period=3, slowd_period=3
    )
    return option_data

def get_option_contracts(symbol, expiry_date=None):
    """
    Fetches available option contracts for a given stock symbol.
    Filters contracts based on expiry date (if provided).
    """
    request = GetOptionContractsRequest(underlying_symbol=symbol)
    contracts = trade_client.get_option_contracts(request)
    #print(contracts[0].symbol)
    #for c in contracts:
    #  print(pd.DataFrame(c[1]))
    # Convert contracts to a list of dictionaries
    contracts_data = [
        
        {
            "symbol": c.symbol,
            "strike_price": c.strike_price,
            "expiration_date": c.expiration_date,
            "option_type": c.type
        }
        for c in contracts.option_contracts
    ]
    print(contracts_data)
    # Filter by expiry date if specified
    if expiry_date:
        contracts_data = [c for c in contracts_data if c["expiration_date"] == expiry_date]

    return contracts_data

def place_iron_condor(symbol, expiry_date):
    """
    Places an Iron Condor options order on Alpaca.
    Uses ATM strikes for demo; adjust logic for dynamic selection.
    """
    options = get_option_contracts(symbol, expiry_date)

    # Find ATM Strikes (Modify as Needed)
    atm_call = min([o for o in options if o["option_type"] == "call"], key=lambda x: abs(x["strike_price"] - 500))
    atm_put = min([o for o in options if o["option_type"] == "put"], key=lambda x: abs(x["strike_price"] - 500))

    # Define Wing Strikes
    lower_put = next(o for o in options if o["strike_price"] < atm_put["strike_price"] and o["option_type"] == "put")
    higher_call = next(o for o in options if o["strike_price"] > atm_call["strike_price"] and o["option_type"] == "call")

    # Define Order Legs
    order_legs = [
        {"symbol": atm_call["symbol"], "qty": 1, "side": OrderSide.SELL},  # Short Call
        {"symbol": higher_call["symbol"], "qty": 1, "side": OrderSide.BUY},  # Long Call
        {"symbol": atm_put["symbol"], "qty": 1, "side": OrderSide.SELL},  # Short Put
        {"symbol": lower_put["symbol"], "qty": 1, "side": OrderSide.BUY},  # Long Put
    ]

    # Create Order Request
    order_request = LimitOrderRequest(
        order_class=OptionOrderClass.BRACKET,
        side=OrderSide.BUY,  # Buying the Iron Condor as a debit strategy
        type=OptionOrderType.LIMIT,
        limit_price=1.50,  # Modify limit price based on market conditions
        time_in_force=TimeInForce.DAY,
        legs=order_legs,
    )

    # Submit Order
    try:
        order = trade_client.submit_order(order_request)
        print(f"✅ Order Submitted: {order}")
    except Exception as e:
        print(f"❌ Order Failed: {e}")

# Example: Place Iron Condor for SPY with expiry on 2025-04-19
#place_iron_condor("SPY", "2025-04-19")

  
def get_option_chain(symbol):
    options_client = OptionHistoricalDataClient(API_KEY, SECRET_KEY)
    request_params = OptionChainRequest(underlying=symbol)
    option_chain = options_client.get_option_chain(request_params)
    return option_chain

#option_chain = get_option_chain("AAPL")
#print(option_chain)



def get_top_50_sp500():
    url = "https://www.slickcharts.com/sp500"
    response = requests.get(url)
    df = pd.read_html(response.text)[0]  # Read HTML table
    top_50_symbols = df["Symbol"][:50].tolist()
    return top_50_symbols

#top_50_symbols = get_top_50_sp500()
#print(top_50_symbols)

def place_option_trade(symbol, qty, side, limit_price):
    order = LimitOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
        time_in_force=TimeInForce.GTC,
        limit_price=limit_price
    )

    response = trade_client.submit_order(order)
    return response

# Example: Buy 1 contract of AAPL $150 Call at $5.00
#trade_response = place_option_trade("AAPL230616C150", qty=1, side="buy", limit_price=5.00)
#print(trade_response)

def should_enter_trade(option_data):
    # Example: Enter trade if RSI is below 30 (oversold)
    rsi = calculate_rsi(option_data)  # Implement RSI function
    if rsi < 30:
        return True
    return False

def should_exit_trade(option_data):
    # Example: Exit if RSI > 70 (overbought)
    rsi = calculate_rsi(option_data)
    if rsi > 70:
        return True
    return False


def place_wings_trade(symbol, expiry, strike, wings_width=5):
    # Example: Buy an Iron Condor for AAPL
    call_buy = f"{symbol}{expiry}C{strike + wings_width}"  # Buy higher strike call
    call_sell = f"{symbol}{expiry}C{strike}"  # Sell at-the-money call
    put_buy = f"{symbol}{expiry}P{strike - wings_width}"  # Buy lower strike put
    put_sell = f"{symbol}{expiry}P{strike}"  # Sell at-the-money put

    for contract in [call_buy, call_sell, put_buy, put_sell]:
        place_option_trade(contract, qty=1, side="buy" if "B" in contract else "sell", limit_price=1.00)

# Example: Iron Condor on AAPL with strikes at $150 and wings at $5
#place_wings_trade("AAPL", "240419", 150, wings_width=5)


def automated_options_trading(symbols,expiry_date=None):
    options_client = OptionHistoricalDataClient(api_key, secret_key, url_override = data_api_url)
    #top_50_symbols = get_top_50_sp500()
    symbols=["AAPL"]
    for symbol in symbols:
        #option_chain = get_option_chain(symbol)
        option_chain=pd.DataFrame(get_option_contracts(symbol, expiry_date))
        
        #print(pd.DataFrame(option_chain))
        # Select the nearest expiry & at-the-money strike
        expiry = option_chain.iloc[0]["expiration_date"]
        atm_strike = option_chain.strike_price[len(option_chain.strike_price) // 2]

        # get options historical bars by symbol

        # Fetch market data
        option_data = options_client.get_option_bars(OptionBarsRequest(symbol_or_symbols=option_chain.iloc[0]["symbol"], timeframe=TimeFrame(amount = 1, unit = TimeFrameUnit.Hour), limit=50))
        option_data = df=calculate_indicators_histall(symbol,1500,False,"stocks") #calculate_indicators_option()(option_data)
        print(option_data[-1:])
        # Trading conditions
        if option_data["SMA_50"].iloc[-1] > option_data["EMA_20"].iloc[-1] and option_data["stoch_k"].iloc[-1] < 20:
            place_wings_trade(symbol, expiry, atm_strike, wings_width=5)

        time.sleep(60)  # Avoid rate limits


def get_historical_options_data(symbol, expiry, strike, option_type, start_date, end_date):
    """
    Fetch historical price data for an option contract.
    """
    option_symbol = f"{symbol}{expiry}{option_type}{strike}"
    request_params = OptionBarsRequest(
        symbol=option_symbol,
        timeframe="1D",
        start=start_date,
        end=end_date,
        limit=100
    )
    
    option_data = options_client.get_option_bars(request_params)
    df = pd.DataFrame(option_data)
    
    return df

def backtest_strategy(symbol, expiry, strike, option_type, start_date, end_date):
    """
    Backtest options trading strategy on historical data.
    """
    df = get_historical_options_data(symbol, expiry, strike, option_type, start_date, end_date)

    if df.empty:
        return "No data available for this contract."

    # Calculate Moving Averages & Stochastics
    df["SMA_50"] = df["close"].rolling(window=50).mean()
    df["EMA_20"] = df["close"].ewm(span=20, adjust=False).mean()

    df["stoch_k"], df["stoch_d"] = talib.STOCH(
        df["high"], df["low"], df["close"], 
        fastk_period=14, slowk_period=3, slowd_period=3
    )

    # Simulate Trading
    cash = 10000  # Starting capital
    position = 0
    trade_log = []

    for i in range(1, len(df)):
        if df["SMA_50"].iloc[i] > df["EMA_20"].iloc[i] and df["stoch_k"].iloc[i] < 20:
            # Buy Signal
            position = cash / df["close"].iloc[i]  # Buy as many contracts as possible
            cash = 0
            trade_log.append(("BUY", df.index[i], df["close"].iloc[i]))

        elif position > 0 and df["stoch_k"].iloc[i] > 80:
            # Sell Signal
            cash = position * df["close"].iloc[i]
            position = 0
            trade_log.append(("SELL", df.index[i], df["close"].iloc[i]))

    # Final portfolio value
    final_value = cash + (position * df["close"].iloc[-1] if position > 0 else 0)
    total_return = ((final_value - 10000) / 10000) * 100

    return {"Final Portfolio Value": final_value, "Total Return (%)": total_return, "Trade Log": trade_log}

def plot_backtest(df, trade_log):
    """
    Plot the backtest results with buy/sell signals.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["close"], label="Option Price", color="blue")

    for trade in trade_log:
        if trade[0] == "BUY":
            plt.scatter(trade[1], trade[2], color="green", marker="^", label="Buy Signal")
        else:
            plt.scatter(trade[1], trade[2], color="red", marker="v", label="Sell Signal")

    plt.xlabel("Date")
    plt.ylabel("Option Price")
    plt.title("Backtest Results")
    plt.legend()
    plt.show()

def get_trade_history():
    orders = trade_client.get_orders(
        GetOrdersRequest(status=QueryOrderStatus.CLOSED,
                         limit=500
                        # , symbols=["SOXL"]
                        )
    )
    trade_data = [
        {
            "symbol": o.symbol,
            "side": o.side.value,
            "qty": o.qty,
            "price": o.filled_avg_price,
            "amount": float(o.qty)*float(o.filled_avg_price),
            "timestamp": o.filled_at #.dt.tz_convert('America/New_York')
        }
        for o in orders
        if o.filled_at is not None  # Ignore unfilled orders
    ]

    return pd.DataFrame(trade_data)

# Fetch current positions
def get_positions():
    positions = trade_client.get_all_positions()
    position_data = [
        {
            "Symbol": p.symbol,
            "Qty": p.qty,
            "Market value": p.market_value,
            "Current Price":p.current_price,
            "Avg. Entry Price":p.avg_entry_price,
            "Lastday price":p.lastday_price,
            "Unrealized_pl ($)": p.unrealized_pl,
            "Unrealized_pl (%)": round(float(p.unrealized_plpc),2)*100
            
        }
        for p in positions
    ]
    return pd.DataFrame(position_data)

# Fetch portfolio history
def get_portfolio_history():
   # positions = broker_client.get_portfolio_history_for_account('43c7bbfc-bca9-4c49-bd3b-19f67cba92bd')
    
    #position_data = [
     #   {
      #      "equity": p.equity,
       #     "profit_loss": p.qtprofit_loss,
        #    "profit_loss_pct": p.profit_loss_pct,
         #   "cashflow":p.cashflow
            
        #}
        #for p in positions
    #]
    res=trade_client.get_account()
    
    position_data = [
        {
            "equity": p.equity,
            "cash": p.cash,
            "last_equity": p.last_equity,
            "cashflow":p.cashflow
            
        }
        for p in res
    ]
    return position_data

