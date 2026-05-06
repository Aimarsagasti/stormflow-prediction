"""Standard TCN (Bai et al. 2018) for iter19.

Independent from `src/models/tcn.py` (TwoStageTCN v1, deprecated).
Implements the original paper architecture without two-stage, without a
classifier, and without a hard switch: residual blocks with two dilated causal
convolutions (WeightNorm + ReLU + Dropout each), 1x1 skip when channels do not
match, residual sum + ReLU.

Causal padding through Chomp1d: after a conv1d with
`padding=(kernel_size-1)*dilation`, the last `padding` steps are trimmed.
This preserves temporal length and guarantees that output at step t depends
only on steps <= t (no temporal leakage).

Receptive field with kernel_size=3 and dilations=[1,2,4,8]:
    RF = 1 + 2 * (kernel_size - 1) * sum(dilations)
       = 1 + 2 * 2 * (1+2+4+8)
       = 1 + 4 * 15
       = 61 steps
with T=72 available steps, the last position sees 61 steps of effective context.

Forward: x (B, T, F) -> permute to (B, F, T) -> blocks -> last position ->
Linear(C, 1) -> y_hat (B,).
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    """Trim the last `chomp_size` steps from the temporal dimension.

    Standard Bai 2018 trick to obtain causal convolution from a conv1d
    with symmetric padding of `(kernel_size-1)*dilation`.
    """

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size <= 0:
            return x
        return x[..., : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Bai 2018 residual block: two dilated causal convs + skip.

    Args:
        in_channels: input channels (F in the first block, hidden after that).
        out_channels: output channels (hidden_channels in all blocks).
        kernel_size: temporal kernel size (3 in the spec).
        dilation: dilation of this block (1, 2, 4, 8 in the spec).
        dropout: dropout probability applied after each ReLU.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Skip: 1x1 conv if channels change, identity otherwise.
        self.downsample: Optional[nn.Conv1d] = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )
        self.relu_out = nn.ReLU()
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu_out(out + res)


class TCNClean(nn.Module):
    """Standard Bai 2018 TCN for scalar regression at one horizon.

    Args:
        in_channels: F (number of input channels). For iter19 = 11.
        hidden_channels: C (internal channels shared by all blocks).
            The spec uses 32 (baseline) or 64 (A3 variant).
        kernel_size: temporal kernel size. The spec fixes it at 3.
        num_blocks: number of residual blocks. The spec fixes it at 4.
        dilations: sequence of per-block dilations. If None, uses
            [2**i for i in range(num_blocks)] = [1, 2, 4, 8] for 4 blocks.
        dropout: dropout probability. The spec fixes it at 0.1.

    Expected forward:
        x: tensor (B, T, F) with normalized sequence.
        Permutes to (B, F, T), passes through the blocks (which preserve T),
        takes the last temporal position, and applies a linear layer C -> 1.
        Returns y_hat (B,) in normalized target space.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        kernel_size: int = 3,
        num_blocks: int = 4,
        dilations: Optional[Sequence[int]] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if dilations is None:
            dilations = [2 ** i for i in range(num_blocks)]
        else:
            dilations = list(dilations)
        if len(dilations) != num_blocks:
            raise ValueError(
                f"len(dilations)={len(dilations)} does not match num_blocks={num_blocks}"
            )

        layers = []
        prev_ch = in_channels
        for d in dilations:
            layers.append(
                TemporalBlock(
                    in_channels=prev_ch,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=int(d),
                    dropout=dropout,
                )
            )
            prev_ch = hidden_channels
        self.blocks = nn.Sequential(*layers)

        self.linear = nn.Linear(hidden_channels, 1)
        nn.init.kaiming_normal_(self.linear.weight, nonlinearity="linear")
        nn.init.zeros_(self.linear.bias)

        # Internal metadata useful for logging / experiment serialization.
        self.config = {
            "in_channels": int(in_channels),
            "hidden_channels": int(hidden_channels),
            "kernel_size": int(kernel_size),
            "num_blocks": int(num_blocks),
            "dilations": [int(d) for d in dilations],
            "dropout": float(dropout),
            "receptive_field": 1 + 2 * (kernel_size - 1) * int(sum(dilations)),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) -> (B, F, T)
        x = x.permute(0, 2, 1).contiguous()
        out = self.blocks(x)              # (B, C, T)
        out = out[:, :, -1]                # (B, C) last temporal position
        y = self.linear(out).squeeze(-1)   # (B,)
        return y

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
