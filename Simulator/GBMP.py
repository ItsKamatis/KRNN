import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

# Determine the device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set input and output directories
input_dir = os.path.expanduser("~/.quantlib/data/nasdaq100/stocks_normalized/")
output_dir = os.path.expanduser("~/.quantlib/data/nasdaq100/GBMP_SIMS/")

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Parameters for the GBM simulation
initial_value = 100000  # Initial stock price (hypothetical value)
time_horizon = 0.5      # 6 months = 0.5 years
time_steps = 126        # Assuming 252 trading days in a year, 6 months = 126 trading days
num_paths = 10          # Number of simulated paths

# Function to simulate Geometric Brownian Motion using PyTorch
def simulate_gbm_pytorch(initial_value, mu, sigma, time_horizon, time_steps, num_paths, device):
    dt = time_horizon / time_steps  # Time increment
    dt_tensor = torch.tensor(dt, dtype=torch.float64, device=device)  # Convert dt to tensor
    t = torch.linspace(0, time_horizon, time_steps + 1, device=device, dtype=torch.float64)  # Time vector

    # Convert mu and sigma to torch tensors on the specified device
    mu_tensor = torch.tensor(mu, dtype=torch.float64, device=device)
    sigma_tensor = torch.tensor(sigma, dtype=torch.float64, device=device)

    # Precompute drift and diffusion
    drift = (mu_tensor - 0.5 * sigma_tensor ** 2) * dt_tensor
    diffusion = sigma_tensor * torch.sqrt(dt_tensor)

    # Initialize the tensor for paths on the specified device
    paths = torch.zeros(num_paths, time_steps + 1, dtype=torch.float64, device=device)
    paths[:, 0] = initial_value

    # Generate random shocks on the specified device
    rand_shocks = torch.randn(num_paths, time_steps, dtype=torch.float64, device=device)

    # Simulate paths
    for t_step in range(1, time_steps + 1):
        # Calculate the next price
        paths[:, t_step] = paths[:, t_step - 1] * torch.exp(drift + diffusion * rand_shocks[:, t_step - 1])

    # Move the paths back to CPU before converting to NumPy
    return paths.cpu().numpy()

# List all CSV files in the input directory
stock_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

# Loop over each stock file
for stock_file in stock_files:
    stock_symbol = os.path.splitext(stock_file)[0]  # Get the stock symbol from the filename
    file_path = os.path.join(input_dir, stock_file)
    print(f"Processing {stock_symbol}...")

    try:
        # Load stock data from the normalized data folder
        df_stock = pd.read_csv(file_path)

        # Calculate daily returns
        df_stock['Returns'] = df_stock['Close'].pct_change()

        # Estimate mu (average daily return)
        mu = df_stock['Returns'].mean()

        # Estimate sigma (volatility as the standard deviation of returns)
        sigma = df_stock['Returns'].std()

        # Print estimated values of mu and sigma
        print(f"Estimated mu for {stock_symbol}: {mu}, Estimated sigma: {sigma}")

        # Simulate GBM for the stock using PyTorch
        simulated_paths = simulate_gbm_pytorch(
            initial_value, mu, sigma, time_horizon, time_steps, num_paths, device
        )

        # Save the simulated paths to a CSV file
        output_file_path = os.path.join(output_dir, f"{stock_symbol}.csv")

        # Convert the simulated paths to a DataFrame
        simulated_df = pd.DataFrame(simulated_paths.T)
        simulated_df.index.name = 'TimeStep'

        # Save to CSV
        simulated_df.to_csv(output_file_path)

        print(f"Simulated data for {stock_symbol} saved to {output_file_path}")


    except Exception as e:
        print(f"An error occurred while processing {stock_symbol}: {e}")

print("GBM simulation complete.")