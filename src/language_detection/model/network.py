from torch import nn
import torch

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

# class MyCNNRNN(nn.Module):
#     def __init__(self, num_classes, input_shape):
#         super().__init__()

#         # Constants
#         dropout = .3

#         # Convolution Layers
#         self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

#         # Normalize Layers
#         self.bn1 = nn.BatchNorm2d(16)
#         self.bn2 = nn.BatchNorm2d(32)

#         # Pooling Features
#         self.relu = nn.ReLU()
#         self.pool = nn.MaxPool2d(kernel_size=2)
#         self.dropout = nn.Dropout(p=dropout)

#         with torch.no_grad():
#             dummy = torch.zeros(1, 1, *input_shape)
#             out = self.pool(self.relu(self.bn1(self.conv1(dummy))))
#             out = self.pool(self.relu(self.bn1(self.conv1(out))))
#             _, C, Fp, Tp = out.shape

#         # Fp --> Frequency pooling Dimension
#         # Tp --> Time pooling Dimensions

#         # Defining RNN
#         self.rnn = nn.LSTM(
#             input_size = C * Fp,
#             hidden_size = 128,
#             num_layers = 2,
#             batch_first = True,
#             bidirectional = True,
#             dropout = dropout,
#         )

#         ## Explanation of all the features
#         # input_size --> Says how many dimensions there are to input vectors
#         # C * Fp

#         # hidden_size --> 

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


from torch.nn.utils.rnn import pack_padded_sequence

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
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
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
            dropout = 0.0
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
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
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
