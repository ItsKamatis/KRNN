# KRNNDATA.py

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Set the directories for your data
historical_data_dir = os.path.expanduser("~/.quantlib/data/nasdaq100/stocks_normalized/")
gbmp_data_dir = os.path.expanduser("~/.quantlib/data/nasdaq100/GBMP_SIMS/")

# Define the sequence length for your model
SEQ_LEN = 30  # You can adjust this value as needed

# Initialize lists to collect data from all stocks
X_all = []
y_all = []

# Initialize scalers
feature_scaler = StandardScaler()
target_scaler = StandardScaler()

# List all CSV files in the historical data directory
stock_files = [f for f in os.listdir(historical_data_dir) if f.endswith('.csv')]

# Loop over each stock file
for stock_file in stock_files:
    stock_symbol = os.path.splitext(stock_file)[0]  # Get the stock symbol from the filename
    print(f"Processing {stock_symbol}...")

    # Paths to the historical and GBMP data files
    historical_file_path = os.path.join(historical_data_dir, stock_file)
    gbmp_file_path = os.path.join(gbmp_data_dir, f"{stock_symbol}.csv")

    # Check if the corresponding GBMP file exists
    if not os.path.isfile(gbmp_file_path):
        print(f"GBMP simulation file for {stock_symbol} not found. Skipping this stock.")
        continue

    try:
        # Load historical stock data
        df_stock = pd.read_csv(historical_file_path, parse_dates=['Date'])
        df_stock.sort_values('Date', inplace=True)  # Ensure data is sorted by date

        # Select relevant columns and fill any missing values
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        df_stock = df_stock[['Date'] + features].dropna()

        # Load GBMP simulation data
        df_gbmp = pd.read_csv(gbmp_file_path)
        # For simplicity, use the first simulation path as the label
        y_gbmp = df_gbmp.iloc[:, 1].values  # Assuming the first column is 'TimeStep'

        # Ensure the lengths match
        data_values = df_stock[features].values

        # Normalize the features
        data_values_scaled = feature_scaler.fit_transform(data_values)

        # Scale the target variable (y_gbmp)
        y_gbmp_scaled = target_scaler.fit_transform(y_gbmp.reshape(-1, 1)).flatten()

        # Number of samples we can create
        num_samples = len(y_gbmp_scaled)

        # Ensure we have enough data to create sequences
        if len(data_values_scaled) < SEQ_LEN + num_samples:
            print(f"Not enough data for {stock_symbol} to create sequences. Skipping this stock.")
            continue

        # Create sequences of features
        X = []
        for i in range(len(data_values_scaled) - SEQ_LEN - num_samples + 1, len(data_values_scaled) - SEQ_LEN + 1):
            X.append(data_values_scaled[i:i+SEQ_LEN])

        X = np.array(X)  # Shape: [num_samples, seq_len, input_dim]
        y = y_gbmp_scaled  # Labels are scaled and aligned with sequences

        # Verify data shapes
        assert X.shape[0] == y.shape[0], f"Mismatch between features and labels for {stock_symbol}"

        # Collect data
        X_all.append(X)
        y_all.append(y)

        print(f"Processed data for {stock_symbol}")

    except Exception as e:
        print(f"An error occurred while processing {stock_symbol}: {e}")

# Combine data from all stocks
if X_all and y_all:
    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    print(f"Total samples collected: {X_all.shape[0]}")
else:
    print("No data was processed. Please check your data files.")
    exit()

# Save the scalers for use during model training and prediction
# You can save them to disk if needed, or pass them to the model
# For this example, we'll assume they're available in memory
