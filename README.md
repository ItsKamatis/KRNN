# KRNN Stock Price Prediction

## Overview

KRNN Stock Price Prediction is a project that leverages the **K Rare-class Nearest Neighbour Classification (KRNN)** algorithm to predict stock prices using time-series data from the NASDAQ 100. Inspired by the research paper "[KRNN: k Rare-class Nearest Neighbour Classification](#)" by Xiuzhen Zhang et al., this project implements a modified version of Microsoft's Qlib [`pytorch_krnn.py`](https://github.com/microsoft/qlib) to address the challenges of imbalanced classification in financial data.

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


