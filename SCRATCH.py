import yaml
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from krnn_new import KRNN
from sklearn.metrics import mean_squared_error

# Load the configuration file
config_path = 'config.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Access configuration parameters
data_config = config['data']
model_config = config['model']
training_config = config['training']
device_config = config['device']
output_config = config['output']

# Data parameters
historical_data_dir = os.path.expanduser(data_config['historical_data_dir'])
gbmp_data_dir = os.path.expanduser(data_config['gbmp_data_dir'])
seq_len = data_config['seq_len']
batch_size = data_config['batch_size']
test_size = data_config['test_size']
valid_size = data_config['valid_size']

# Data processing

# Define the sequence length for your model
SEQ_LEN = seq_len  # Use the value from the config

# Initialize lists to collect data from all stocks
X_all = []
y_all = []

# List all CSV files in the historical data directory
stock_files = [f for f in os.listdir(historical_data_dir) if f.endswith('.csv')]

# Collect raw data without scaling
data_values_list = []
y_gbmp_list = []

# First Loop: Collect data for scaling
for stock_file in stock_files:
    stock_symbol = os.path.splitext(stock_file)[0]  # Get the stock symbol from the filename
    print(f"Collecting data for {stock_symbol}...")

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

        # Select relevant columns and drop missing values
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        df_stock = df_stock[['Date'] + features].dropna()

        # Load GBMP simulation data
        df_gbmp = pd.read_csv(gbmp_file_path)
        y_gbmp = df_gbmp.iloc[:, 1].values  # Assuming the first column is 'TimeStep'

        # Check for negative values
        if np.any(y_gbmp < 0):
            print(f"Negative values found in y_gbmp for {stock_symbol}. Skipping this stock.")
            continue

        # Apply log transformation to y_gbmp
        y_gbmp_log = np.log1p(y_gbmp)

        # Collect data for scaling later
        data_values = df_stock[features].values
        data_values_list.append(data_values)
        y_gbmp_list.append(y_gbmp_log)

    except Exception as e:
        print(f"An error occurred while processing {stock_symbol}: {e}")

# Combine all data values and target values for scaling
all_data_values = np.concatenate(data_values_list, axis=0)
all_y_gbmp_log = np.concatenate(y_gbmp_list, axis=0)

# Fit scalers on all data
feature_scaler = StandardScaler()
target_scaler = StandardScaler()

feature_scaler.fit(all_data_values)
target_scaler.fit(all_y_gbmp_log.reshape(-1, 1))

# Second Loop: Process each stock with the fitted scalers
for stock_file in stock_files:
    stock_symbol = os.path.splitext(stock_file)[0]
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
        df_stock.sort_values('Date', inplace=True)

        # Select relevant columns and drop missing values
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        df_stock = df_stock[['Date'] + features].dropna()

        # Load GBMP simulation data
        df_gbmp = pd.read_csv(gbmp_file_path)
        y_gbmp = df_gbmp.iloc[:, 1].values

        # Check for negative values
        if np.any(y_gbmp < 0):
            print(f"Negative values found in y_gbmp for {stock_symbol}. Skipping this stock.")
            continue

        # Apply log transformation to y_gbmp
        y_gbmp_log = np.log1p(y_gbmp)

        data_values = df_stock[features].values

        # Normalize the features and scale the target variable using the global scalers
        data_values_scaled = feature_scaler.transform(data_values)
        y_gbmp_scaled = target_scaler.transform(y_gbmp_log.reshape(-1, 1)).flatten()

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

# Update feature dimension in the model configuration
model_config['fea_dim'] = X_all.shape[2]

# Data splitting
X_train_val, X_test, y_train_val, y_test_scaled = train_test_split(
    X_all, y_all, test_size=test_size, shuffle=False
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_val, y_train_val, test_size=valid_size, shuffle=False
)

# Instantiate the model with updated dimensions
model = KRNN(
    fea_dim=model_config['fea_dim'],
    cnn_dim=256,  # Increased dimension
    cnn_kernel_size=model_config['cnn_kernel_size'],
    rnn_dim=256,  # Increased dimension
    rnn_dups=model_config['rnn_dups'],
    rnn_layers=model_config['rnn_layers'],
    dropout=model_config['dropout'],
    n_epochs=training_config['n_epochs'],
    lr=training_config['lr'],
    early_stop=training_config['early_stop'],
    batch_size=batch_size,
    loss=training_config['loss'],
    optimizer=training_config['optimizer'],
    GPU=device_config['GPU'],
    seed=training_config['seed'],
)

# Train the model
evals_result = {}
model.fit(
    X_train, y_train,
    X_valid, y_valid,
    evals_result=evals_result,
    save_path=output_config['model_save_path']
)

# Make predictions on the test set
predictions_scaled = model.predict(X_test)

# Inverse transform the predictions and actual values
predictions_log = target_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
y_test_log = target_scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()

# Inverse log transformation
predictions = np.expm1(predictions_log)
y_test = np.expm1(y_test_log)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f"Test MSE: {mse}")

# Plot predictions vs. actual values
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(y_test[:200], label='Actual')
plt.plot(predictions[:200], label='Predicted')
plt.legend()
plt.title('KRNN Model Predictions vs Actual Values (First 200 Samples)')
plt.show()

# After training
plt.figure(figsize=(10, 5))
plt.plot(evals_result['train'], label='Train Loss')
plt.plot(evals_result['valid'], label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

print("Training complete.")
