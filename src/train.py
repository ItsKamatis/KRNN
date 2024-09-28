# src/train.py

import yaml
import os
import pandas as pd
import numpy as np
import torch
import random
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from models.krnn_new import KRNN  # Correct import path
import logging


def load_config(config_path='config.yaml'):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(logs_dir='logs'):
    """
    Set up logging configuration to log messages to both a file and the console.

    Args:
        logs_dir (str): Directory where log files will be stored.

    Returns:
        logging.Logger: Configured logger.
    """
    os.makedirs(logs_dir, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(os.path.join(logs_dir, 'train.log'))
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger


class TimeSeriesDataset(Dataset):
    """
    Custom Dataset for Time Series Data.
    """
    def __init__(self, X, y, augment=False):
        """
        Args:
            X (numpy.ndarray): Features, shape (num_samples, seq_len, num_features)
            y (numpy.ndarray): Targets, shape (num_samples,)
            augment (bool): Whether to apply data augmentation
        """
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]

        if self.augment:
            # Example: Add jitter noise
            noise = torch.randn_like(X) * 0.01
            X = X + noise

        return X, y


def load_data(processed_data_dir, seq_len, features_to_use, target_column, logger):
    """
    Load and preprocess data from CSV files.

    Args:
        processed_data_dir (str): Directory containing processed CSV files.
        seq_len (int): Sequence length for time series.
        features_to_use (list or None): List of feature column names to use.
        target_column (str): Name of the target column.
        logger (logging.Logger): Logger for logging information.

    Returns:
        tuple: Tuple containing features and targets as NumPy arrays.
    """
    stock_files = [f for f in os.listdir(processed_data_dir) if f.endswith('.csv')]
    X_all = []
    y_all = []

    for stock_file in stock_files:
        stock_symbol = os.path.splitext(stock_file)[0]
        logger.info(f"Processing {stock_symbol}...")
        file_path = os.path.join(processed_data_dir, stock_file)
        try:
            df = pd.read_csv(file_path, parse_dates=['Date'])
            df.sort_values('Date', inplace=True)
            if len(df) < seq_len + 1:
                logger.warning(f"Not enough data for {stock_symbol}. Skipping.")
                continue
            if features_to_use is None:
                feature_columns = df.columns.difference(['Date', target_column])
            else:
                feature_columns = features_to_use
            df['Target'] = df[target_column].shift(-1)
            df.dropna(inplace=True)
            X = df[feature_columns].values
            y = df['Target'].values
            X_sequences, y_sequences = create_sequences(X, y, seq_len)
            X_all.append(X_sequences)
            y_all.append(y_sequences)
        except Exception as e:
            logger.error(f"An error occurred while processing {stock_symbol}: {e}")

    if X_all and y_all:
        X_all = np.concatenate(X_all, axis=0)
        y_all = np.concatenate(y_all, axis=0)
        logger.info(f"Total samples collected: {X_all.shape[0]}")
        return X_all, y_all
    else:
        logger.error("No data was processed. Please check your data files.")
        exit()


def create_sequences(X, y, seq_length):
    """
    Create sequences of data for time series.

    Args:
        X (numpy.ndarray): Feature data.
        y (numpy.ndarray): Target data.
        seq_length (int): Length of each sequence.

    Returns:
        tuple: Tuple containing sequences and corresponding targets.
    """
    xs = []
    ys = []
    for i in range(len(X) - seq_length):
        x_seq = X[i:(i + seq_length)]
        y_seq = y[i + seq_length]
        xs.append(x_seq)
        ys.append(y_seq)
    return np.array(xs), np.array(ys)


def scale_data(X_train, X_valid, X_test, y_train, y_valid, y_test):
    """
    Scale features and targets using StandardScaler.

    Args:
        X_train (numpy.ndarray): Training features.
        X_valid (numpy.ndarray): Validation features.
        X_test (numpy.ndarray): Test features.
        y_train (numpy.ndarray): Training targets.
        y_valid (numpy.ndarray): Validation targets.
        y_test (numpy.ndarray): Test targets.

    Returns:
        tuple: Scaled features and targets along with the target scaler.
    """
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_valid_reshaped = X_valid.reshape(-1, X_valid.shape[-1])
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])

    X_train_scaled = feature_scaler.fit_transform(X_train_reshaped).reshape(X_train.shape)
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()

    X_valid_scaled = feature_scaler.transform(X_valid_reshaped).reshape(X_valid.shape)
    y_valid_scaled = target_scaler.transform(y_valid.reshape(-1, 1)).flatten()

    X_test_scaled = feature_scaler.transform(X_test_reshaped).reshape(X_test.shape)
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()

    return X_train_scaled, X_valid_scaled, X_test_scaled, y_train_scaled, y_valid_scaled, y_test_scaled, target_scaler


def train_and_evaluate(model_config, training_config, device_config, train_loader,
                       valid_loader, test_loader, plots_dir,
                       fold, logger):
    """
    Train and evaluate the KRNN model for a specific fold.

    Args:
        model_config (dict): Configuration for the model.
        training_config (dict): Configuration for training.
        device_config (dict): Configuration for device usage.
        train_loader (DataLoader): DataLoader for training data.
        valid_loader (DataLoader): DataLoader for validation data.
        test_loader (DataLoader): DataLoader for test data.
        plots_dir (str): Directory to save plots.
        fold (int): Current fold number.
        logger (logging.Logger): Logger for logging information.

    Returns:
        tuple: Mean Squared Error, Mean Absolute Error, R^2 Score.
    """
    model = KRNN(
        fea_dim=model_config.get('fea_dim'),
        cnn_dim=model_config.get('cnn_dim'),
        cnn_kernel_size=model_config.get('cnn_kernel_size'),
        rnn_dim=model_config.get('rnn_dim'),
        rnn_dups=model_config.get('rnn_dups'),
        rnn_layers=model_config.get('rnn_layers'),
        dropout=model_config.get('dropout'),
        n_epochs=training_config.get('n_epochs'),
        lr=training_config.get('lr'),
        early_stop=training_config.get('early_stop'),
        loss=training_config.get('loss'),
        optimizer=training_config.get('optimizer'),
        batch_size=training_config.get('batch_size'),
        GPU=device_config.get('GPU'),
        seed=training_config.get('seed')
    )

    device = next(model.parameters()).device
    logger.info(f"Fold {fold} - Training on device: {device}")
    print(f"Fold {fold} - Training on device: {device}")  # Optional

    evals_result = {}
    logger.info(f"Fold {fold} - Starting training...")
    model.fit(
        train_loader, valid_loader,
        evals_result=evals_result,
        save_path=None  # Optionally save the model per fold
    )

    # Evaluation
    test_predictions = model.predict(test_loader)

    # Inverse transform
    test_predictions_inv = training_config['target_scaler'].inverse_transform(
        test_predictions.reshape(-1, 1)).flatten()
    y_test_inv = training_config['target_scaler'].inverse_transform(
        test_loader.dataset.y.numpy().reshape(-1, 1)).flatten()

    # Evaluation metrics
    mse = mean_squared_error(y_test_inv, test_predictions_inv)
    mae = mean_absolute_error(y_test_inv, test_predictions_inv)
    r2 = r2_score(y_test_inv, test_predictions_inv)

    logger.info(
        f"Fold {fold} - Test MSE: {mse:.6f}, MAE: {mae:.6f}, R^2 Score: {r2:.6f}"
    )

    # Plotting Predictions vs Actual
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(y_test_inv, label='Actual')
        plt.plot(test_predictions_inv, label='Predicted')
        plt.legend()
        plt.title(f'KRNN Model Predictions vs. Actual Values - Fold {fold}')
        plot_path = os.path.join(plots_dir, f'predictions_vs_actual_fold_{fold}.png')
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Saved predictions plot to: {plot_path}")
    except Exception as e:
        logger.error(f"Failed to save predictions plot for fold {fold}: {e}")

    # Plotting Training and Validation Loss
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(evals_result['train'], label='Train Loss')
        plt.plot(evals_result['valid'], label='Validation Loss')
        plt.legend()
        plt.title(f'Training and Validation Loss over Epochs - Fold {fold}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        loss_plot_path = os.path.join(plots_dir, f'loss_over_epochs_fold_{fold}.png')
        plt.savefig(loss_plot_path)
        plt.close()
        logger.info(f"Saved loss plot to: {loss_plot_path}")
    except Exception as e:
        logger.error(f"Failed to save loss plot for fold {fold}: {e}")

    return mse, mae, r2


def validate_config(config):
    """
    Validate the presence of required configuration sections.

    Args:
        config (dict): Configuration dictionary.

    Raises:
        ValueError: If required sections are missing.
    """
    required_sections = ['data', 'model', 'training', 'device', 'output']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section '{section}' in config.yaml")

    if config['training'].get('batch_size') is None:
        raise ValueError("Batch size must be specified in the training section.")


def main():
    """Main function to execute the training pipeline."""
    config = load_config()

    # Validate configuration
    try:
        validate_config(config)
    except ValueError as ve:
        print(f"Configuration Error: {ve}")
        exit()

    # Access configuration parameters
    data_config = config.get('data', {})
    model_config = config.get('model', {})
    training_config = config.get('training', {})
    device_config = config.get('device', {})
    output_config = config.get('output', {})

    # Setup logging
    logs_dir = os.path.expanduser(output_config.get('logs_dir', 'logs'))
    os.makedirs(logs_dir, exist_ok=True)
    logger = setup_logging(logs_dir)
    logger.info("Logging setup complete.")

    # Set random seed
    set_seed(training_config.get('seed', 42))
    logger.info("Random seed set.")

    # Load and prepare data
    X_all, y_all = load_data(
        processed_data_dir=os.path.expanduser(
            data_config.get('processed_data_dir',
                            "C:/Users/colme/.quantlib/data/nasdaq100/processed_data/")),
        seq_len=data_config.get('seq_len', 10),
        features_to_use=data_config.get('features_to_use', None),
        target_column=data_config.get('target_column', 'Close'),
        logger=logger
    )

    # Update 'fea_dim' in model configuration
    fea_dim = X_all.shape[2]
    model_config['fea_dim'] = fea_dim
    logger.info(f"Feature dimension set to {fea_dim}.")

    # Implement Time-Series Cross-Validation
    n_splits = data_config.get('n_splits', 5)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    logger.info(f"Using TimeSeriesSplit with {n_splits} splits.")

    fold = 0
    cv_results = []

    # Output directories for plots and models
    plots_dir = os.path.expanduser(output_config.get('plots_dir', 'plots'))
    models_dir = os.path.expanduser(output_config.get('models_dir', 'models'))
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    logger.info(f"Plots will be saved to: {plots_dir}")
    logger.info(f"Models will be saved to: {models_dir}")

    for train_val_index, test_index in tscv.split(X_all):
        fold += 1
        logger.info(f"Starting fold {fold}...")
        X_train_val, X_test = X_all[train_val_index], X_all[test_index]
        y_train_val, y_test = y_all[train_val_index], y_all[test_index]

        # Further split train_val into training and validation sets
        valid_size_adjusted = int(len(X_train_val) * data_config.get('valid_size', 0.1))
        X_train = X_train_val[:-valid_size_adjusted]
        y_train = y_train_val[:-valid_size_adjusted]

        X_valid = X_train_val[-valid_size_adjusted:]
        y_valid = y_train_val[-valid_size_adjusted:]

        logger.info(
            f"Fold {fold} - Training samples: {len(X_train)}, Validation samples: {len(X_valid)}, Test samples: {len(X_test)}"
        )

        # Scale data within the fold
        X_train_scaled, X_valid_scaled, X_test_scaled, y_train_scaled, y_valid_scaled, y_test_scaled, target_scaler = scale_data(
            X_train, X_valid, X_test, y_train, y_valid, y_test
        )
        logger.info(f"Fold {fold} - Data scaled.")

        # Store scaler for inverse transformation
        training_config['target_scaler'] = target_scaler

        # Create Datasets
        train_dataset = TimeSeriesDataset(X_train_scaled, y_train_scaled, augment=True)
        valid_dataset = TimeSeriesDataset(X_valid_scaled, y_valid_scaled, augment=False)
        test_dataset = TimeSeriesDataset(X_test_scaled, y_test_scaled, augment=False)

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=training_config.get('batch_size', 32),
            shuffle=True,
            num_workers=training_config.get('num_workers', 4),  # Updated to use config
            pin_memory=True
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=training_config.get('batch_size', 32),
            shuffle=False,
            num_workers=training_config.get('num_workers', 4),  # Updated to use config
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=training_config.get('batch_size', 32),
            shuffle=False,
            num_workers=training_config.get('num_workers', 4),  # Updated to use config
            pin_memory=True
        )

        # Train and evaluate the model
        mse, mae, r2 = train_and_evaluate(
            model_config=model_config,
            training_config=training_config,
            device_config=device_config,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            plots_dir=plots_dir,
            fold=fold,
            logger=logger
        )

        cv_results.append({
            'fold': fold,
            'mse': mse,
            'mae': mae,
            'r2': r2
        })

    # Summarize cross-validation results
    avg_mse = np.mean([result['mse'] for result in cv_results])
    avg_mae = np.mean([result['mae'] for result in cv_results])
    avg_r2 = np.mean([result['r2'] for result in cv_results])

    logger.info(f"Average Test MSE over {n_splits} folds: {avg_mse:.6f}")
    logger.info(f"Average Test MAE over {n_splits} folds: {avg_mae:.6f}")
    logger.info(f"Average Test R^2 Score over {n_splits} folds: {avg_r2:.6f}")

    print(f"Average Test MSE over {n_splits} folds: {avg_mse:.6f}")
    print(f"Average Test MAE over {n_splits} folds: {avg_mae:.6f}")
    print(f"Average Test R^2 Score over {n_splits} folds: {avg_r2:.6f}")


if __name__ == "__main__":
    main()
