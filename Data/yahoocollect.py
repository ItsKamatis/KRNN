import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Path to the Nasdaq-100 CSV file
csv_file = os.path.expanduser("~/.quantlib/data/nasdaq100/nasdaq100.csv")

# Load the list of Nasdaq-100 companies and tickers
nasdaq_df = pd.read_csv(csv_file)

# Create the directory to store the stock data if it doesn't exist
stock_data_directory = os.path.expanduser("~/.quantlib/data/nasdaq100/stocks")
if not os.path.exists(stock_data_directory):
    os.makedirs(stock_data_directory)

# Define the date range: 6 years ago to today - 1
end_date = datetime.today() - timedelta(days=1)
start_date = end_date - timedelta(days=6 * 365)

# Function to download stock data for each ticker
def download_stock_data(ticker, company):
    # Define the file path for the stock data
    stock_file_path = os.path.join(stock_data_directory, f"{ticker}.csv")

    # Download the daily data from Yahoo Finance
    print(f"Downloading data for {company} ({ticker})...")
    stock_data = yf.download(ticker, start=start_date, end=end_date)

    # If data is found, save it to the CSV file
    if not stock_data.empty:
        stock_data.to_csv(stock_file_path)
        print(f"Data for {ticker} saved to {stock_file_path}")
    else:
        print(f"No data found for {ticker}")

# Check if any stock data files already exist
existing_files = [f for f in os.listdir(stock_data_directory) if f.endswith('.csv')]
if existing_files:
    user_input = input("Stock data files already exist. Overwrite? (Y/N): ").strip().lower()
    if user_input != 'y':
        print("Process cancelled.")
        exit()
    else:
        print("Overwriting existing stock data files...")

# Loop through each ticker in the Nasdaq-100 list and download its data
for index, row in nasdaq_df.iterrows():
    ticker = row['Ticker']
    company = row['Company']
    download_stock_data(ticker, company)

print("Data collection complete.")
