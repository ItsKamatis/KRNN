# src/models/krnn_new.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import copy


class CNNKRNNEncoder(nn.Module):
    def __init__(
            self,
            cnn_input_dim,
            cnn_output_dim,
            cnn_kernel_size,
            rnn_output_dim,
            rnn_dup_num,
            rnn_layers,
            dropout
    ):
        super().__init__()
        self.cnn = nn.Conv1d(
            in_channels=cnn_input_dim,
            out_channels=cnn_output_dim,
            kernel_size=cnn_kernel_size,
            padding='same'
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(
            input_size=cnn_output_dim,
            hidden_size=rnn_output_dim,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=dropout if rnn_layers > 1 else 0
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, cnn_input_dim)
        x = x.permute(0, 2,
                      1)  # Convert to (batch_size, cnn_input_dim, seq_len)
        x = self.cnn(x)  # (batch_size, cnn_output_dim, seq_len)
        x = self.relu(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, cnn_output_dim)
        output, _ = self.rnn(x)  # (batch_size, seq_len, rnn_output_dim)
        return output


class KRNNModel(nn.Module):
    def __init__(
            self,
            fea_dim,
            cnn_dim=64,
            cnn_kernel_size=3,
            rnn_dim=64,
            rnn_dups=1,
            rnn_layers=1,
            dropout=0.2,
    ):
        super().__init__()
        self.encoder = CNNKRNNEncoder(
            cnn_input_dim=fea_dim,
            cnn_output_dim=cnn_dim,
            cnn_kernel_size=cnn_kernel_size,
            rnn_output_dim=rnn_dim,
            rnn_dup_num=rnn_dups,
            rnn_layers=rnn_layers,
            dropout=dropout,
        )
        self.out_fc = nn.Linear(rnn_dim, 1)

    def forward(self, x):
        x = self.encoder(x)
        # Use the last time step's output
        x = x[:, -1, :]  # (batch_size, rnn_dim)
        out = self.out_fc(x)  # (batch_size, 1)
        return out.squeeze(1)  # (batch_size,)


class KRNN:
    """KRNN Model"""

    def __init__(
            self,
            fea_dim=6,
            cnn_dim=64,
            cnn_kernel_size=3,
            rnn_dim=64,
            rnn_dups=3,
            rnn_layers=2,
            dropout=0.0,
            n_epochs=200,
            lr=0.001,
            metric="loss",
            batch_size=64,
            early_stop=20,
            loss="mse",
            optimizer="adam",
            GPU=-1,
            seed=None,
    ):
        # Set logger
        self.logger = logging.getLogger(__name__)
        self.logger.info("KRNN pytorch version...")

        # Set hyperparameters
        self.fea_dim = fea_dim
        self.cnn_dim = cnn_dim
        self.cnn_kernel_size = cnn_kernel_size
        self.rnn_dim = rnn_dim
        self.rnn_dups = rnn_dups
        self.rnn_layers = rnn_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.metric = metric
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.optimizer_name = optimizer.lower()
        self.loss_name = loss
        self.seed = seed

        # Set device
        if torch.cuda.is_available() and GPU >= 0:
            self.device = torch.device(f"cuda:{GPU}")
        else:
            self.device = torch.device("cpu")
        self.logger.info(f"KRNN initialized on device: {self.device}")

        # Set random seed
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

        # Initialize the model
        self.krnn_model = KRNNModel(
            fea_dim=self.fea_dim,
            cnn_dim=self.cnn_dim,
            cnn_kernel_size=self.cnn_kernel_size,
            rnn_dim=self.rnn_dim,
            rnn_dups=self.rnn_dups,
            rnn_layers=self.rnn_layers,
            dropout=self.dropout,
        ).to(self.device)

        # Set optimizer
        if self.optimizer_name == "adam":
            self.train_optimizer = optim.Adam(
                self.krnn_model.parameters(), lr=self.lr, weight_decay=1e-5
            )
        elif self.optimizer_name == "sgd":
            self.train_optimizer = optim.SGD(
                self.krnn_model.parameters(), lr=self.lr, weight_decay=1e-5
            )
        else:
            raise NotImplementedError(
                f"Optimizer {self.optimizer_name} is not supported!"
            )

        # Initialize loss function
        if self.loss_name == "mse":
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss `{self.loss_name}`")

        self.fitted = False

    def parameters(self):
        """Return the parameters of the internal KRNNModel."""
        return self.krnn_model.parameters()

    def train_epoch(self, train_loader):
        self.krnn_model.train()
        total_loss = 0.0
        num_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.train_optimizer.zero_grad()
            outputs = self.krnn_model(X_batch)
            loss = self.loss_fn(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.krnn_model.parameters(), 3.0)
            self.train_optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def evaluate(self, valid_loader):
        self.krnn_model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for X_batch, y_batch in valid_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                outputs = self.krnn_model(X_batch)
                loss = self.loss_fn(outputs, y_batch)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def fit(self, train_loader, valid_loader, evals_result=None,
            save_path=None):
        if evals_result is None:
            evals_result = {}
        best_score = float("inf")
        best_epoch = 0
        best_param = None
        stop_steps = 0

        evals_result["train"] = []
        evals_result["valid"] = []

        for epoch in range(self.n_epochs):
            train_loss = self.train_epoch(train_loader)
            evals_result["train"].append(train_loss)

            valid_loss = self.evaluate(valid_loader)
            evals_result["valid"].append(valid_loss)
            self.logger.info(
                f"Epoch {epoch + 1}: Train Loss={train_loss:.6f}, Valid Loss={valid_loss:.6f}"
            )

            if valid_loss < best_score:
                best_score = valid_loss
                best_epoch = epoch
                best_param = copy.deepcopy(self.krnn_model.state_dict())
                stop_steps = 0
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("Early stopping triggered.")
                    break

        if best_param is not None:
            self.krnn_model.load_state_dict(best_param)

        self.fitted = True

        if save_path is not None:
            torch.save(self.krnn_model.state_dict(), save_path)

    def predict(self, test_loader):
        if not self.fitted:
            raise ValueError("Model is not fitted yet!")

        self.krnn_model.eval()
        preds = []

        with torch.no_grad():
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(self.device)
                outputs = self.krnn_model(X_batch)
                preds.append(outputs.cpu().numpy())

        return np.concatenate(preds)
