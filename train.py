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
from models.krnn_new import KRNN  # Ensure this module is correctly implemented
import logging


def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(logs_dir='logs'):
    os.makedirs(logs_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(logs_dir, 'train.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger()
    return logger


def load_data(processed_data_dir, seq_len, features_to_use, target_column,
              logger):
    stock_files = [f for f in os.listdir(processed_data_dir) if
                   f.endswith('.csv')]
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
                feature_columns = df.columns.difference(['Date', 'Target'])
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
            logger.error(
                f"An error occurred while processing {stock_symbol}: {e}")

    if X_all and y_all:
        X_all = np.concatenate(X_all, axis=0)
        y_all = np.concatenate(y_all, axis=0)
        logger.info(f"Total samples collected: {X_all.shape[0]}")
        return X_all, y_all
    else:
        logger.error("No data was processed. Please check your data files.")
        exit()


def create_sequences(X, y, seq_length):
    xs = []
    ys = []
    for i in range(len(X) - seq_length):
        x_seq = X[i:(i + seq_length)]
        y_seq = y[i + seq_length]
        xs.append(x_seq)
        ys.append(y_seq)
    return np.array(xs), np.array(ys)


def scale_data(X_train, X_valid, X_test, y_train, y_valid, y_test):
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_valid_reshaped = X_valid.reshape(-1, X_valid.shape[-1])
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])

    X_train_scaled = feature_scaler.fit_transform(X_train_reshaped).reshape(
        X_train.shape)
    y_train_scaled = target_scaler.fit_transform(
        y_train.reshape(-1, 1)).flatten()

    X_valid_scaled = feature_scaler.transform(X_valid_reshaped).reshape(
        X_valid.shape)
    y_valid_scaled = target_scaler.transform(y_valid.reshape(-1, 1)).flatten()

    X_test_scaled = feature_scaler.transform(X_test_reshaped).reshape(
        X_test.shape)
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()

    return X_train_scaled, X_valid_scaled, X_test_scaled, y_train_scaled, y_valid_scaled, y_test_scaled, target_scaler


def train_and_evaluate(model_config, training_config, device_config, X_train,
                       y_train, X_valid, y_valid, X_test, y_test, plots_dir,
                       fold, logger):
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

    evals_result = {}
    logger.info(f"Fold {fold} - Starting training...")
    model.fit(
        X_train, y_train,
        X_valid, y_valid,
        evals_result=evals_result,
        save_path=None  # Optionally save the model per fold
    )

    test_predictions = model.predict(X_test)

    # Inverse transform
    test_predictions_inv = training_config['target_scaler'].inverse_transform(
        test_predictions.reshape(-1, 1)).flatten()
    y_test_inv = training_config['target_scaler'].inverse_transform(
        y_test.reshape(-1, 1)).flatten()

    # Debugging: Inspect statistics
    logger.info(
        f"Fold {fold} - Y_test stats: min={y_test_inv.min()}, max={y_test_inv.max()}, mean={y_test_inv.mean():.4f}, std={y_test_inv.std():.4f}")
    logger.info(
        f"Fold {fold} - Predictions stats: min={test_predictions_inv.min()}, max={test_predictions_inv.max()}, mean={test_predictions_inv.mean():.4f}, std={test_predictions_inv.std():.4f}")

    # Evaluation metrics
    mse = mean_squared_error(y_test_inv, test_predictions_inv)
    mae = mean_absolute_error(y_test_inv, test_predictions_inv)
    r2 = r2_score(y_test_inv, test_predictions_inv)

    logger.info(
        f"Fold {fold} - Test MSE: {mse:.6f}, MAE: {mae:.6f}, R^2 Score: {r2:.6f}")

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_inv, label='Actual')
    plt.plot(test_predictions_inv, label='Predicted')
    plt.legend()
    plt.title(f'KRNN Model Predictions vs. Actual Values - Fold {fold}')
    plt.savefig(
        os.path.join(plots_dir, f'predictions_vs_actual_fold_{fold}.png'))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(evals_result['train'], label='Train Loss')
    plt.plot(evals_result['valid'], label='Validation Loss')
    plt.legend()
    plt.title(f'Training and Validation Loss over Epochs - Fold {fold}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(plots_dir, f'loss_over_epochs_fold_{fold}.png'))
    plt.close()

    return mse, mae, r2


def main():
    config = load_config()

    # Access configuration parameters
    data_config = config.get('data', {})
    model_config = config.get('model', {})
    training_config = config.get('training', {})
    device_config = config.get('device', {})
    output_config = config.get('output', {})

    # Setup logging
    logs_dir = output_config.get('logs_dir', '~/.quantlib/data/nasdaq100/logs')
    logs_dir = os.path.expanduser(logs_dir)
    os.makedirs(logs_dir, exist_ok=True)
    logger = setup_logging(logs_dir)

    # Set random seed
    set_seed(training_config.get('seed', 42))
    logger.info("Random seed set.")

    # Load and prepare data
    X_all, y_all = load_data(
        processed_data_dir=os.path.expanduser(
            data_config.get('processed_data_dir',
                            "~/.quantlib/data/nasdaq100/processed_data/")),
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

    for train_val_index, test_index in tscv.split(X_all):
        fold += 1
        logger.info(f"Starting fold {fold}...")
        X_train_val, X_test = X_all[train_val_index], X_all[test_index]
        y_train_val, y_test = y_all[train_val_index], y_all[test_index]

        # Further split train_val into training and validation sets
        valid_size_adjusted = int(
            len(X_train_val) * data_config.get('valid_size', 0.1))
        X_train = X_train_val[:-valid_size_adjusted]
        y_train = y_train_val[:-valid_size_adjusted]

        X_valid = X_train_val[-valid_size_adjusted:]
        y_valid = y_train_val[-valid_size_adjusted:]

        logger.info(
            f"Fold {fold} - Training samples: {len(X_train)}, Validation samples: {len(X_valid)}, Test samples: {len(X_test)}")

        # Scale data within the fold
        X_train_scaled, X_valid_scaled, X_test_scaled, y_train_scaled, y_valid_scaled, y_test_scaled, target_scaler = scale_data(
            X_train, X_valid, X_test, y_train, y_valid, y_test
        )
        logger.info(f"Fold {fold} - Data scaled.")

        # Store scalers for inverse transformation
        training_config['target_scaler'] = target_scaler

        # Train and evaluate the model
        mse, mae, r2 = train_and_evaluate(
            model_config=model_config,
            training_config=training_config,
            device_config=device_config,
            X_train=X_train_scaled,
            y_train=y_train_scaled,
            X_valid=X_valid_scaled,
            y_valid=y_valid_scaled,
            X_test=X_test_scaled,
            y_test=y_test_scaled,
            plots_dir='plots_dir',
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
