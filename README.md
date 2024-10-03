# KRNN Stock Price Prediction

## Overview

KRNN Stock Price Prediction is a project that leverages the **K Rare-class Nearest Neighbour Classification (KRNN)** algorithm to predict stock prices using time-series data from the NASDAQ 100. Inspired by the research paper "[KRNN: k Rare-class Nearest Neighbour Classification](https://www.sciencedirect.com/science/article/abs/pii/S0031320316302369)" by Xiuzhen Zhang et al., this project implements a modified version of Microsoft's Qlib [`pytorch_krnn.py`](https://github.com/microsoft/qlib) to address the challenges of imbalanced classification in financial data.
### THIS VERSION IS NOW OUTDATED AND INCOMPLETE. 
Since last update a new and improved version of the project has been under 
construction. The new version is currently under testing and will be released. 
## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Results](#results)
- [Credits](#credits)
- [License](#license)
- [Contact](#contact)

## Features

- **Time-Series Cross-Validation:** Utilizes `TimeSeriesSplit` to maintain temporal order in training and testing.
- **Data Scaling:** Implements proper scaling within each cross-validation fold to prevent data leakage.
- **Model Training and Evaluation:** Trains the KRNN model and evaluates it using MSE, MAE, and R² metrics.
- **Plot Generation:** Saves plots of predictions vs. actual values and training/validation loss over epochs.
- **Logging:** Comprehensive logging to monitor the training process and debug if necessary.

## Installation

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/yourusername/krnn-stock-prediction.git
    cd krnn-stock-prediction
    ```

2. **Create a Virtual Environment:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Prepare Your Data:

Ensure that your processed CSV data files are located in the directory specified in `config.yaml` under `data -> processed_data_dir`.

### Configure Parameters:

Modify the `config.yaml` file to adjust parameters such as sequence length, batch size, model hyperparameters, and output paths.

### Run the Training Script:

```bash
python train.py
```

The script will perform cross-validation, train the KRNN model, evaluate its performance, and save the resulting plots. 

## Configuration
The project is configured via the config.yaml file. Below is an overview of the key parameters:
```bash
# config.yaml

## Data parameters
data:
  processed_data_dir: "~/.quantlib/data/nasdaq100/processed_data/"
  seq_len: 10            # Sequence length
  batch_size: 32         # Batch size
  test_size: 0.2         # Fraction of data for testing
  valid_size: 0.1        # Fraction of training data for validation
  features_to_use: null  # List of features to use; null means use all
  target_column: "Close" # Target variable for prediction
  n_splits: 5            # Number of cross-validation splits

## Model parameters
model:
  fea_dim: null         # Will be set dynamically based on the data
  cnn_dim: 256          # CNN layer dimensions
  cnn_kernel_size: 3    # CNN kernel size
  rnn_dim: 256          # RNN layer dimensions
  rnn_dups: 3           # Number of parallel RNNs
  rnn_layers: 2         # Number of RNN layers
  dropout: 0.5          # Dropout rate

## Training parameters
training:
  n_epochs: 100         # Number of training epochs
  lr: 0.0005            # Learning rate
  early_stop: 15        # Early stopping patience
  loss: "mse"           # Loss function
  optimizer: "adam"     # Optimizer
  seed: 42              # Random seed for reproducibility

## Device configuration
device:
  GPU: 0                # GPU ID to use; set to -1 for CPU

## Output paths
output:
  model_save_path: "krnn_model.pth"             # Path to save the trained model
  plots_dir: "~/.quantlib/data/nasdaq100/plots" # Directory to save plots
  logs_dir: "~/.quantlib/data/nasdaq100/logs"   # Directory to save logs
```
## Results
#### Upon running the training script, the following outputs are generated:

***Model Performance Metrics:*** MSE, MAE, and R² scores for each fold and their averages.

***Plots:*** Predictions vs. Actual Values: 
Visual comparison of model predictions against true values.

***Training and Validation Loss over Epochs:*** Tracks the loss reduction during training.

These results are saved in the directories specified in config.yaml.

## Credits 
This project builds upon and is inspired by the following works and tools:

***Microsoft Qlib:*** 
Utilized the pytorch_krnn.py implementation for KRNN.

***Research Paper:*** Inspired by the paper:

##### KRNN: k Rare-class Nearest Neighbour Classification

Xiuzhen Zhang, Yuxuan Li, Ramamohanarao Kotagiri, Lifang Wu, Zahir Tari, Mohamed Cheriet

****Affiliations:****

RMIT University, Australia
The University of Melbourne, Australia
Beijing University of Technology, PR China
The University of Quebec (ETS), Canada

****Abstract:****

Imbalanced classification is a challenging problem. Re-sampling and cost-sensitive learning are global strategies for generality-oriented algorithms such as the decision tree, targeting inter-class imbalance. We research local strategies for the specificity-oriented learning algorithms like the k Nearest Neighbour (KNN) to address the within-class imbalance issue of positive data sparsity. We propose an algorithm k Rare-class Nearest Neighbour, or KRNN, by directly adjusting the induction bias of KNN. We propose to form dynamic query neighbourhoods, and to further adjust the positive posterior probability estimation to bias classification towards the rare class. We conducted extensive experiments on thirty real-world and artificial datasets to evaluate the performance of KRNN. Our experiments showed that KRNN significantly improved KNN for classification of the rare class, and often outperformed re-sampling and cost-sensitive learning strategies with generality-oriented base learners.

ChatGPT (OpenAI): Leveraged various versions (o1-preview, o1-mini, and 4) for code development and project guidance.

## License 
This project is licensed under the MIT License. See the [LICENSE](https://github.com/ItsKamatis/QuantLIB/blob/master/LICENSE) file for details.

## Contact
Anru Joshua Colmenar

[Rutgers University](https://www.business.rutgers.edu/)

Email: [aj.colmenar@rutgers.edu](mailto:aj.colmenar@rutgers.edu)

