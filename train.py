import yaml
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import random
import matplotlib.pyplot as plt

from krnn_new import KRNN  # Ensure this module is correctly implemented

# Load the configuration file
config_path = 'config.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Access configuration parameters
data_config = config.get('data', {})
model_config = config.get('model', {})
training_config = config.get('training', {})
device_config = config.get('device', {})
output_config = config.get('output', {})

# Set random seed for reproducibility
seed = training_config.get('seed', 42)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Data parameters with default values if not specified
seq_len = data_config.get('seq_len', 10)
batch_size = data_config.get('batch_size', 32)
test_size = data_config.get('test_size', 0.2)
valid_size = data_config.get('valid_size', 0.1)

# Directories
processed_data_dir = os.path.expanduser("~/.quantlib/data/nasdaq100/processed_data/")

# List all CSV files in the processed data directory
stock_files = [f for f in os.listdir(processed_data_dir) if f.endswith('.csv')]

# Initialize lists to collect data from all stocks
X_all = []
y_all = []

# Loop over each stock file
for stock_file in stock_files:
    stock_symbol = os.path.splitext(stock_file)[0]
    print(f"Processing {stock_symbol}...")
    file_path = os.path.join(processed_data_dir, stock_file)
    try:
        # Load the processed data
        df = pd.read_csv(file_path, parse_dates=['Date'])
        df.sort_values('Date', inplace=True)

        # Ensure that the DataFrame has enough data
        if len(df) < seq_len + 1:
            print(f"Not enough data for {stock_symbol}. Skipping.")
            continue

        # Define the feature columns (excluding 'Date' and 'Target' if present)
        feature_columns = df.columns.difference(['Date', 'Target'])

        # Create the target variable (predicting the next day's close price)
        df['Target'] = df['Close'].shift(-1)
        df.dropna(inplace=True)  # Drop the last row with NaN target

        # Extract features and target
        X = df[feature_columns].values
        y = df['Target'].values

        # Create sequences
        def create_sequences(X, y, seq_length):
            xs = []
            ys = []
            for i in range(len(X) - seq_length):
                x_seq = X[i:(i + seq_length)]
                y_seq = y[i + seq_length]
                xs.append(x_seq)
                ys.append(y_seq)
            return np.array(xs), np.array(ys)

        X_sequences, y_sequences = create_sequences(X, y, seq_len)

        # Collect the sequences
        X_all.append(X_sequences)
        y_all.append(y_sequences)

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

# Ensure that feature dimension matches the data
fea_dim = X_all.shape[2]  # Number of features

# Update 'fea_dim' in model configuration
model_config['fea_dim'] = fea_dim

# Split the data into training, validation, and test sets
# Calculate indices for splitting
test_split_idx = int(len(X_all) * (1 - test_size))
valid_split_idx = int(test_split_idx * (1 - valid_size / (1 - test_size)))

# Split the data
X_train = X_all[:valid_split_idx]
y_train = y_all[:valid_split_idx]

X_valid = X_all[valid_split_idx:test_split_idx]
y_valid = y_all[valid_split_idx:test_split_idx]

X_test = X_all[test_split_idx:]
y_test = y_all[test_split_idx:]

print(f"Training samples: {len(X_train)}, Validation samples: {len(X_valid)}, Test samples: {len(X_test)}")

# Instantiate the model
model = KRNN(
    fea_dim=model_config.get('fea_dim', fea_dim),
    cnn_dim=model_config.get('cnn_dim', 256),
    cnn_kernel_size=model_config.get('cnn_kernel_size', 3),
    rnn_dim=model_config.get('rnn_dim', 256),
    rnn_dups=model_config.get('rnn_dups', 3),
    rnn_layers=model_config.get('rnn_layers', 2),
    dropout=model_config.get('dropout', 0.5),
    n_epochs=training_config.get('n_epochs', 100),
    lr=training_config.get('lr', 0.0005),
    early_stop=training_config.get('early_stop', 15),
    loss=training_config.get('loss', 'mse'),
    optimizer=training_config.get('optimizer', 'adam'),
    batch_size=batch_size,
    GPU=device_config.get('GPU', -1),
    seed=training_config.get('seed', 42)
)

# Train the model
evals_result = {}
model.fit(
    X_train, y_train,
    X_valid, y_valid,
    evals_result=evals_result,
    save_path=output_config.get('model_save_path', 'krnn_model.pth')
)

# Evaluate the model on the test set
test_predictions = model.predict(X_test)

# Calculate MSE
mse = mean_squared_error(y_test, test_predictions)
print(f"Test MSE: {mse:.4f}")

# Plot predictions vs. actual values
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual')
plt.plot(test_predictions, label='Predicted')
plt.legend()
plt.title('KRNN Model Predictions vs. Actual Values')
plt.show()

# Plot training and validation loss over epochs
plt.figure(figsize=(10, 5))
plt.plot(evals_result['train'], label='Train Loss')
plt.plot(evals_result['valid'], label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
