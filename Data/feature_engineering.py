import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import joblib  # For saving the scaler
from ta import add_all_ta_features  # For technical indicators
from ta.utils import dropna
import ta

# Directories for input and output
input_dir = os.path.expanduser("~/.quantlib/data/nasdaq100/stocks")
output_dir = os.path.expanduser("~/.quantlib/data/nasdaq100/processed_data")

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define the columns to keep
columns_to_keep = [
    'Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
    # Selected Indicators
    'trend_sma_slow', 'trend_ema_slow',  # Moving Averages
    'momentum_rsi',  # RSI
    'trend_macd', 'trend_macd_signal', 'trend_macd_diff',  # MACD components
    'volatility_atr',  # ATR
    'volatility_bbm', 'volatility_bbh', 'volatility_bbl', 'volatility_bbp',  # Bollinger Bands components
    'volume_obv', 'volume_vpt',  # Volume Indicators
    'trend_adx', 'trend_adx_pos', 'trend_adx_neg',  # ADX
    'HL_Range', 'OC_Range',  # Custom Features
    # Time Features (optional, encode if used)
    # 'Day_of_Week', 'Month'
]

# Scale the new features (excluding categorical like 'Day_of_Week' and 'Month')
features_to_scale = [
    col for col in columns_to_keep if col not in ['Date', 'Day_of_Week', 'Month']
]

# Collect all processed data for fitting the scalers
all_processed_data = {}
stock_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

# First loop: Process data and collect for scaler fitting
for filename in stock_files:
    stock_symbol = os.path.splitext(filename)[0]
    file_path = os.path.join(input_dir, filename)
    # Read the CSV file
    df = pd.read_csv(file_path)
    # Process the data (add technical indicators)
    def add_technical_indicators(df):
        # Ensure 'Date' is datetime
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values('Date', inplace=True)

        # Drop rows with NaN values
        df = dropna(df)

        # Calculate selected technical indicators
        # Moving Averages
        df['trend_sma_slow'] = df['Close'].rolling(window=50).mean()
        df['trend_ema_slow'] = df['Close'].ewm(span=50, adjust=False).mean()

        # RSI
        df['momentum_rsi'] = ta.momentum.RSIIndicator(close=df['Close'], window=14, fillna=True).rsi()

        # MACD
        macd = ta.trend.MACD(close=df['Close'], fillna=True)
        df['trend_macd'] = macd.macd()
        df['trend_macd_signal'] = macd.macd_signal()
        df['trend_macd_diff'] = macd.macd_diff()

        # ATR
        df['volatility_atr'] = ta.volatility.AverageTrueRange(
            high=df['High'], low=df['Low'], close=df['Close'], window=14, fillna=True
        ).average_true_range()

        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(close=df['Close'], window=20, fillna=True)
        df['volatility_bbm'] = bollinger.bollinger_mavg()
        df['volatility_bbh'] = bollinger.bollinger_hband()
        df['volatility_bbl'] = bollinger.bollinger_lband()
        df['volatility_bbp'] = bollinger.bollinger_pband()

        # OBV
        df['volume_obv'] = ta.volume.OnBalanceVolumeIndicator(
            close=df['Close'], volume=df['Volume'], fillna=True
        ).on_balance_volume()

        # VPT
        df['volume_vpt'] = ta.volume.VolumePriceTrendIndicator(
            close=df['Close'], volume=df['Volume'], fillna=True
        ).volume_price_trend()

        # ADX
        adx = ta.trend.ADXIndicator(
            high=df['High'], low=df['Low'], close=df['Close'], window=14, fillna=True
        )
        df['trend_adx'] = adx.adx()
        df['trend_adx_pos'] = adx.adx_pos()
        df['trend_adx_neg'] = adx.adx_neg()

        # Calculate Volume (log-transformed)
        df['Volume'] = np.log1p(df['Volume'])

        # Custom Features
        df['HL_Range'] = df['High'] - df['Low']
        df['OC_Range'] = df['Open'] - df['Close']

        # Time Features
        # df['Day_of_Week'] = df['Date'].dt.dayofweek
        # df['Month'] = df['Date'].dt.month

        return df

    # Process the data
    df_processed = add_technical_indicators(df)

    # Drop any new NaN values resulting from indicators
    df_processed.dropna(inplace=True)

    # Keep only the desired columns and create a copy
    df_processed = df_processed[columns_to_keep].copy()

    # Store the processed DataFrame
    all_processed_data[stock_symbol] = df_processed

# Concatenate all processed data to fit the scaler
combined_processed_data = pd.concat(all_processed_data.values(), ignore_index=True)

# Fit the initial scaler on the original features
feature_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
scaler = StandardScaler()
scaler.fit(combined_processed_data[feature_columns])

# Save the scaler for future use
scaler_path = os.path.join(output_dir, 'scaler.pkl')
joblib.dump(scaler, scaler_path)
print(f"Scaler saved to {scaler_path}")

# Fit the additional scaler on the new features
additional_scaler = StandardScaler()
additional_scaler.fit(combined_processed_data[features_to_scale])

# Save the additional scaler
additional_scaler_path = os.path.join(output_dir, 'additional_scaler.pkl')
joblib.dump(additional_scaler, additional_scaler_path)
print(f"Additional scaler saved to {additional_scaler_path}")

# Second loop: Scale and save processed data
for stock_symbol, df_processed in all_processed_data.items():
    output_path = os.path.join(output_dir, f"{stock_symbol}.csv")
    # Ensure columns are of type float64
    df_processed[feature_columns] = df_processed[feature_columns].astype('float64')
    df_processed[features_to_scale] = df_processed[features_to_scale].astype('float64')
    # Scale the features using the fitted scalers
    df_processed.loc[:, feature_columns] = scaler.transform(df_processed[feature_columns])
    df_processed.loc[:, features_to_scale] = additional_scaler.transform(df_processed[features_to_scale])
    # Save the processed data to the output directory
    df_processed.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

print("Feature engineering complete.")
