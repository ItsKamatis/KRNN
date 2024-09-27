import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import copy

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CNNEncoderBase(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size

        # Ensure same length output by setting appropriate padding
        self.conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=output_dim,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        x = x.permute(0, 2, 1)  # [batch_size, input_dim, seq_len]
        y = self.conv(x)        # [batch_size, output_dim, seq_len]
        y = y.permute(0, 2, 1)  # [batch_size, seq_len, output_dim]
        return y


class KRNNEncoderBase(nn.Module):
    def __init__(self, input_dim, output_dim, dup_num, rnn_layers, dropout):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dup_num = dup_num
        self.rnn_layers = rnn_layers
        self.dropout = dropout

        self.rnn_modules = nn.ModuleList()
        for _ in range(dup_num):
            self.rnn_modules.append(
                nn.GRU(
                    input_size=input_dim,
                    hidden_size=output_dim,
                    num_layers=rnn_layers,
                    dropout=dropout if rnn_layers > 1 else 0,
                    batch_first=True,
                )
            )

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        hids = []
        for rnn in self.rnn_modules:
            h, _ = rnn(x)  # [batch_size, seq_len, output_dim]
            hids.append(h)

        # Stack and average the outputs from the parallel RNNs
        hids = torch.stack(hids, dim=-1)  # [batch_size, seq_len, output_dim, dup_num]
        hids = hids.mean(dim=-1)          # [batch_size, seq_len, output_dim]
        return hids


class CNNKRNNEncoder(nn.Module):
    def __init__(
        self,
        cnn_input_dim,
        cnn_output_dim,
        cnn_kernel_size,
        rnn_output_dim,
        rnn_dup_num,
        rnn_layers,
        dropout,
    ):
        super().__init__()
        self.cnn_encoder = CNNEncoderBase(cnn_input_dim, cnn_output_dim, cnn_kernel_size)
        self.krnn_encoder = KRNNEncoderBase(
            input_dim=cnn_output_dim,
            output_dim=rnn_output_dim,
            dup_num=rnn_dup_num,
            rnn_layers=rnn_layers,
            dropout=dropout,
        )

    def forward(self, x):
        cnn_out = self.cnn_encoder(x)
        krnn_out = self.krnn_encoder(cnn_out)
        return krnn_out


class KRNNModel(nn.Module):
    def __init__(
        self,
        fea_dim,
        cnn_dim,
        cnn_kernel_size,
        rnn_dim,
        rnn_dups,
        rnn_layers,
        dropout,
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
        # x: [batch_size, seq_len, input_dim]
        encode = self.encoder(x)
        out = self.out_fc(encode[:, -1, :]).squeeze()
        return out


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
        self.logger = logger
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

        self.fitted = False

    def mse(self, pred, label):
        loss = (pred - label) ** 2
        return torch.mean(loss)

    def loss_fn(self, pred, label):
        if self.loss_name == "mse":
            return self.mse(pred, label)
        else:
            raise ValueError(f"Unknown loss `{self.loss_name}`")

    def metric_fn(self, pred, label):
        if self.metric in ("", "loss"):
            return -self.loss_fn(pred, label)
        else:
            raise ValueError(f"Unknown metric `{self.metric}`")

    def train_epoch(self, X_train, y_train):
        self.krnn_model.train()
        total_loss = 0.0
        num_batches = 0

        indices = np.arange(len(X_train))
        np.random.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            X_batch = torch.from_numpy(X_train[batch_indices]).float().to(self.device)
            y_batch = torch.from_numpy(y_train[batch_indices]).float().to(self.device)

            self.train_optimizer.zero_grad()
            outputs = self.krnn_model(X_batch)
            loss = self.loss_fn(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.krnn_model.parameters(), 3.0)
            self.train_optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def evaluate(self, X_eval, y_eval):
        self.krnn_model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for i in range(0, len(X_eval), self.batch_size):
                X_batch = torch.from_numpy(X_eval[i : i + self.batch_size]).float().to(self.device)
                y_batch = torch.from_numpy(y_eval[i : i + self.batch_size]).float().to(self.device)

                outputs = self.krnn_model(X_batch)
                loss = self.loss_fn(outputs, y_batch)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, evals_result=None, save_path=None):
        if evals_result is None:
            evals_result = {}
        best_score = float("inf")
        best_epoch = 0
        best_param = None
        stop_steps = 0

        evals_result["train"] = []
        if X_valid is not None and y_valid is not None:
            evals_result["valid"] = []

        for epoch in range(self.n_epochs):
            train_loss = self.train_epoch(X_train, y_train)
            evals_result["train"].append(train_loss)

            if X_valid is not None and y_valid is not None:
                valid_loss = self.evaluate(X_valid, y_valid)
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
            else:
                self.logger.info(f"Epoch {epoch + 1}: Train Loss={train_loss:.6f}")

        if best_param is not None:
            self.krnn_model.load_state_dict(best_param)

        self.fitted = True

        if save_path is not None:
            torch.save(self.krnn_model.state_dict(), save_path)

    def predict(self, X):
        if not self.fitted:
            raise ValueError("Model is not fitted yet!")

        self.krnn_model.eval()
        preds = []

        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                X_batch = torch.from_numpy(X[i : i + self.batch_size]).float().to(self.device)
                outputs = self.krnn_model(X_batch)
                preds.append(outputs.cpu().numpy())

        return np.concatenate(preds)
