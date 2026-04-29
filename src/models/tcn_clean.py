"""TCN estandar (Bai et al. 2018) para iter19.

Independiente de `src/models/tcn.py` (TwoStageTCN v1, deprecated).
Implementa la arquitectura del paper original sin two-stage, sin clasificador,
sin switch duro: bloques residuales con dos convoluciones causales dilatadas
(WeightNorm + ReLU + Dropout cada una), skip 1x1 cuando los canales no
coinciden, suma residual + ReLU.

Padding causal mediante Chomp1d: tras una conv1d con
`padding=(kernel_size-1)*dilation`, se recortan los ultimos `padding` pasos.
Esto preserva la longitud temporal y garantiza que la salida en el paso t solo
depende de pasos <= t (sin leakage temporal).

Receptive field con kernel_size=3 y dilations=[1,2,4,8]:
    RF = 1 + 2 * (kernel_size - 1) * sum(dilations)
       = 1 + 2 * 2 * (1+2+4+8)
       = 1 + 4 * 15
       = 61 pasos
con T=72 pasos disponibles, la ultima posicion ve 61 pasos de contexto efectivo.

Forward: x (B, T, F) -> permuta a (B, F, T) -> bloques -> ultima posicion ->
Linear(C, 1) -> y_hat (B,).
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    """Recorta los ultimos `chomp_size` pasos de la dimension temporal.

    Truco estandar de Bai 2018 para conseguir convolucion causal a partir
    de una conv1d con padding simetrico en `(kernel_size-1)*dilation`.
    """

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size <= 0:
            return x
        return x[..., : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Bloque residual de Bai 2018: dos conv causales dilatadas + skip.

    Args:
        in_channels: canales de entrada (F en el primer bloque, hidden despues).
        out_channels: canales de salida (hidden_channels en todos los bloques).
        kernel_size: tamano del kernel temporal (3 en la spec).
        dilation: dilatacion de este bloque (1, 2, 4, 8 en la spec).
        dropout: probabilidad de dropout aplicada tras cada ReLU.
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

        # Skip: 1x1 conv si los canales cambian, identidad si no.
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
    """TCN estandar Bai 2018 para regresion escalar a un horizonte.

    Args:
        in_channels: F (numero de canales de entrada). Para iter19 = 11.
        hidden_channels: C (canales internos comunes a todos los bloques).
            La spec usa 32 (baseline) o 64 (variante A3).
        kernel_size: tamano del kernel temporal. La spec fija 3.
        num_blocks: numero de bloques residuales. La spec fija 4.
        dilations: secuencia de dilataciones por bloque. Si None, usa
            [2**i for i in range(num_blocks)] = [1, 2, 4, 8] para 4 bloques.
        dropout: probabilidad de dropout. La spec fija 0.1.

    Forward esperado:
        x: tensor (B, T, F) con secuencia normalizada.
        Permuta a (B, F, T), pasa por los bloques (que preservan T),
        toma la ultima posicion temporal y aplica una capa lineal C -> 1.
        Devuelve y_hat (B,) en espacio normalizado del target.
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
                f"len(dilations)={len(dilations)} no coincide con num_blocks={num_blocks}"
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

        # Metadata interna util para logging / serializacion del experimento.
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
        out = out[:, :, -1]                # (B, C) ultima posicion temporal
        y = self.linear(out).squeeze(-1)   # (B,)
        return y

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
