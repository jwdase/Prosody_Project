import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence


class CNNLanguageDetector(nn.Module):
    """
    CNN Neural Network
    """

    def __init__(self, num_classes, input_shape):
        super().__init__()

        # Convolution Layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # Normalize Layers
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)

        # Pooling Features
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout(p=0.3)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, *input_shape)
            out = self.pool(self.relu(self.conv1(dummy)))
            out = self.pool(self.relu(self.conv2(out)))
            flat_dim = out.view(1, -1).shape[1]

        # Neuron Layers
        self.fc1 = nn.Linear(flat_dim, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):

        # Apply 2D Convolution
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))

        # Flatten Dimensions
        x = x.view(x.size(0), -1)

        # Dense Layers
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)

        return x


class CNNRNNLanguageDetector(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_shape: tuple[int,int],    # (freq_bins, time_steps)
        rnn_hidden: int = 128,
        rnn_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3
    ):
        super().__init__()

        # 1) CNN feature extractor
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.relu  = nn.ReLU()
        self.pool  = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(dropout)

        # figure out size after conv/pool
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *input_shape)
            out = self.pool(self.relu(self.bn1(self.conv1(dummy))))
            out = self.pool(self.relu(self.bn2(self.conv2(out))))
            _, C, Fp, Tp = out.shape

        # 2) RNN on the sequence of Tp feature‑vectors of size (C × Fp)
        self.rnn = nn.LSTM(
            input_size  = C * Fp,
            hidden_size = rnn_hidden,
            num_layers  = rnn_layers,
            batch_first = True,
            bidirectional=bidirectional,
            dropout      = dropout if rnn_layers>1 else 0.0
        )

        # 3) Final classifier
        rnn_directions = 2 if bidirectional else 1
        self.classifier = nn.Sequential(
            nn.Linear(rnn_hidden * rnn_directions, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, freq_bins, time_steps]
        returns: [B, num_classes]
        """
        B = x.size(0)

        # ---- CNN stack ----
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        # now x.shape == [B, C, F′, T′]

        # ---- reshape to sequence ----
        x = x.permute(0, 3, 1, 2)       # [B, T′, C, F′]
        x = x.contiguous().view(B, x.size(1), -1)  
        # [B, T′, C × F′]

        # ---- RNN ----
        rnn_out, (h_n, _) = self.rnn(x)
        # h_n: [num_layers * num_directions, B, rnn_hidden]

        if self.rnn.bidirectional:
            # concat last forward & backward hidden states
            h_fw = h_n[-2]   # final layer, forward direction
            h_bw = h_n[-1]   # final layer, backward direction
            h_last = torch.cat([h_fw, h_bw], dim=1)  # [B, hidden*2]
        else:
            h_last = h_n[-1]  # [B, hidden]

        # ---- classify ----
        return self.classifier(h_last)

# ------------------------------ Make Model ----------------------


class VarCNNRNNLanguageDetector(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_shape: tuple[int, int],  # (freq_bins, time_steps)
        rnn_hidden: int = 128,
        rnn_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3
    ):
        super().__init__()

        # CNN feature extractor
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(16)
        self.relu  = nn.ReLU()
        self.pool  = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(dropout)

        # Determine output size after conv/pool
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *input_shape)
            out = self.pool(self.relu(self.bn1(self.conv1(dummy))))
            out = self.pool(self.relu(self.bn2(self.conv2(out))))
            _, C, Fp, Tp = out.shape

        self.Fp = Fp
        self.C = C

        # RNN layer
        self.rnn = nn.LSTM(
            input_size  = C * Fp,
            hidden_size = rnn_hidden,
            num_layers  = rnn_layers,
            batch_first = True,
            bidirectional = bidirectional,
            dropout = dropout
        )

        rnn_directions = 2 if bidirectional else 1
        self.classifier = nn.Sequential(
            nn.Linear(rnn_hidden * rnn_directions, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        x:       [B, 1, F, T]  — padded spectrograms
        lengths: [B]           — actual (unpadded) time steps BEFORE CNN
        """
        B = x.size(0)

        # CNN feature extraction
        x = self.dropout(self.pool(self.relu(self.bn1(self.conv1(x)))))
        x = self.dropout(self.pool(self.relu(self.bn2(self.conv2(x)))))
        # x shape: [B, C, Fp, Tp]

        # Adjust lengths for CNN pooling (2x MaxPool2d(2) → total downsample by 4)
        lengths = torch.clamp(torch.div(lengths, 4, rounding_mode='trunc'), min=1)
        lengths = torch.clamp(lengths, min=1)  # 👈 ensures no zero-length

        # Prepare input for RNN: [B, T', C, F'] → [B, T', C*F']
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, T', C, F']
        x = x.view(B, x.size(1), -1)            # [B, T', C*F']

        lengths = torch.clamp(lengths, max=x.size(1), min=1)

        # Debug: check for NaN/inf
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("Input to LSTM contains NaN or inf")

        # Pack and send through RNN
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)

        packed_out, (h_n, _) = self.rnn(packed)

        # Get final hidden state(s)
        if self.rnn.bidirectional:
            h_fw = h_n[-2]  # forward
            h_bw = h_n[-1]  # backward
            h_last = torch.cat([h_fw, h_bw], dim=1)  # [B, hidden*2]
        else:
            h_last = h_n[-1]  # [B, hidden]

        return self.classifier(h_last)

# -------------- Model 2

class VarCNNRNNLanguageDetector2(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_shape: tuple[int, int],  # (freq_bins, time_steps)
        rnn_hidden: int = 128,
        rnn_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3
    ):
        super().__init__()

        # CNN feature extractor
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(16)
        self.relu  = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Determine output size after conv/pool
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *input_shape)
            out = self.relu(self.bn1(self.conv1(dummy)))
            out = self.relu(self.bn2(self.conv2(out)))
            _, C, Fp, Tp = out.shape

        self.Fp = Fp
        self.C = C

        # RNN layer
        self.rnn = nn.LSTM(
            input_size  = C * Fp,
            hidden_size = rnn_hidden,
            num_layers  = rnn_layers,
            batch_first = True,
            bidirectional = bidirectional,
            dropout = dropout
        )

        rnn_directions = 2 if bidirectional else 1
        self.classifier = nn.Sequential(
            nn.Linear(rnn_hidden * rnn_directions, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        x:       [B, 1, F, T]  — padded spectrograms
        lengths: [B]           — actual (unpadded) time steps BEFORE CNN
        """
        B = x.size(0)

        # CNN feature extraction
        x = self.dropout(self.relu(self.bn1(self.conv1(x))))
        x = self.dropout(self.relu(self.bn2(self.conv2(x))))
        # x shape: [B, C, Fp, Tp]

        # Adjust lengths for CNN pooling (2x MaxPool2d(2) → total downsample by 4)
        # lengths = torch.clamp(torch.div(lengths, 4, rounding_mode='trunc'), min=1)
        lengths = torch.clamp(lengths, min=1)  # 👈 ensures no zero-length

        # Prepare input for RNN: [B, T', C, F'] → [B, T', C*F']
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, T', C, F']
        x = x.view(B, x.size(1), -1)            # [B, T', C*F']

        lengths = torch.clamp(lengths, max=x.size(1), min=1)

        # Pack and send through RNN
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)

        packed_out, (h_n, _) = self.rnn(packed)

        # Get final hidden state(s)
        if self.rnn.bidirectional:
            h_fw = h_n[-2]  # forward
            h_bw = h_n[-1]  # backward
            h_last = torch.cat([h_fw, h_bw], dim=1)  # [B, hidden*2]
        else:
            h_last = h_n[-1]  # [B, hidden]

        return self.classifier(h_last)



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10_000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # shape [1, max_len, d_model]

    def forward(self, x):
        # x: [B, T, D]
        return x + self.pe[:, :x.size(1), :]


class VarCNNTransformerLanguageDetector(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_shape: tuple[int, int],  # (freq_bins, time_steps)
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.3
    ):
        super().__init__()

        # CNN feature extractor
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.relu  = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # No pooling

        # Compute flattened size after CNN
        freq_bins, _ = input_shape
        self.flattened_dim = 32 * freq_bins  # no pooling, 32 channels

        # Transformer encoder
        self.pos_encoder = PositionalEncoding(self.flattened_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.flattened_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_dim, self.flattened_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.flattened_dim, num_classes)
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        x:       [B, 1, F, T]
        lengths: [B] (before CNN)
        """
        B = x.size(0)

        # CNN feature extraction
        x = self.relu(self.bn1(self.conv1(x)))  # [B, 16, F, T]
        x = self.relu(self.bn2(self.conv2(x)))  # [B, 32, F, T]
        x = self.dropout(x)

        # Prepare for transformer: [B, 32, F, T] → [B, T, 32*F]
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, T, C, F]
        x = x.view(B, x.size(1), -1)            # [B, T, D]

        # Adjust lengths (no pooling used here, but keep if you add one)
        # If MaxPool2d(2) was used twice, you'd do: lengths = lengths // 4

        # Padding mask: True = masked (i.e., padding)
        T = x.size(1)
        mask = torch.arange(T, device=lengths.device)[None, :] >= lengths[:, None]

        # Positional encoding
        x = self.pos_encoder(x)  # [B, T, D]

        # Transformer
        x = self.transformer(x, src_key_padding_mask=mask)  # [B, T, D]

        # Aggregate: mean over non-padded time steps
        lengths_clamped = lengths.clamp(min=1).unsqueeze(1)
        summed = x.sum(dim=1)                          # [B, D]
        meaned = summed / lengths_clamped              # [B, D]

        # Classifier
        return self.classifier(meaned)                 # [B, num_classes]
