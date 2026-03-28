"""Composite loss functions for stormflow peak-focused training."""

from __future__ import annotations  # Permite anotaciones modernas con buena compatibilidad

import torch  # Provee operaciones tensoriales para calculo de perdidas
from torch import nn  # Incluye clases base para construir modulos de perdida


class CompositeLoss(nn.Module):
    """Composite loss: weighted Huber + asymmetric underprediction + peak MSE."""

    def __init__(
        self,
        p95_threshold: float,
        p99_threshold: float,
        p999_threshold: float,
        huber_weight: float = 0.55,
        asym_weight: float = 0.30,
        peak_weight: float = 0.15,
        huber_beta: float = 1.0,
    ) -> None:
        super().__init__()  # Inicializa clase base para registrar el modulo correctamente
        self.p95_threshold = float(p95_threshold)  # Guarda umbral P95 para activar componente de picos
        self.p99_threshold = float(p99_threshold)  # Guarda umbral P99 para penalizacion asimetrica intermedia
        self.p999_threshold = float(p999_threshold)  # Guarda umbral P99.9 para penalizacion asimetrica extrema
        self.huber_weight = float(huber_weight)  # Guarda peso global de la componente Weighted Huber
        self.asym_weight = float(asym_weight)  # Guarda peso global de la componente asimetrica
        self.peak_weight = float(peak_weight)  # Guarda peso global de la componente Peak MSE
        self.huber_base = nn.SmoothL1Loss(reduction="none", beta=huber_beta)  # Define Huber elemento a elemento para poder ponderar por muestra

    def _weighted_huber(self, y_pred: torch.Tensor, y_true: torch.Tensor, sample_weights: torch.Tensor) -> torch.Tensor:
        huber_per_sample = self.huber_base(y_pred, y_true)  # Calcula error Huber por muestra y dimension
        weighted_huber = huber_per_sample * sample_weights  # Aplica peso de magnitud proveniente del DataLoader
        return weighted_huber.mean()  # Promedia sobre batch para obtener escalar de la componente

    def _asymmetric_underprediction(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        under_error = torch.relu(y_true - y_pred)  # Conserva solo infraestimaciones y anula sobreestimaciones
        asym_factor = torch.full_like(y_true, 2.0)  # Inicializa factor base 2x para valores por debajo de P99
        asym_factor = torch.where(y_true >= self.p99_threshold, torch.full_like(y_true, 4.0), asym_factor)  # Sube a 4x cuando y_true esta en zona >= P99
        asym_factor = torch.where(y_true >= self.p999_threshold, torch.full_like(y_true, 6.0), asym_factor)  # Sube a 6x cuando y_true esta en zona >= P99.9
        asym_loss = (under_error ** 2) * asym_factor  # Penaliza cuadraticamente la infraestimacion con factor segun severidad
        return asym_loss.mean()  # Promedia sobre batch para obtener escalar estable

    def _peak_mse(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        peak_mask = y_true >= self.p95_threshold  # Selecciona solo muestras de pico para esta componente
        if peak_mask.any():  # Verifica si hay picos en el batch para evitar media sobre tensor vacio
            peak_error = y_pred[peak_mask] - y_true[peak_mask]  # Calcula residuo solo en zona de picos relevantes
            return (peak_error ** 2).mean()  # Calcula MSE de picos como escalar
        return torch.zeros((), device=y_true.device, dtype=y_true.dtype)  # Retorna cero en batches sin picos para no contaminar la perdida

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, sample_weights: torch.Tensor) -> torch.Tensor:
        if y_pred.shape != y_true.shape:  # Valida que prediccion y target tengan misma forma (batch, 1)
            raise ValueError("y_pred and y_true must have the same shape")  # Mensaje claro para detectar errores de modelo/dataloader
        if sample_weights.shape != y_true.shape:  # Valida que pesos por muestra coincidan con forma del target
            raise ValueError("sample_weights must have the same shape as y_true")  # Mensaje claro para detectar errores de batching

        huber_component = self._weighted_huber(y_pred=y_pred, y_true=y_true, sample_weights=sample_weights)  # Calcula componente 1 Weighted Huber
        asym_component = self._asymmetric_underprediction(y_pred=y_pred, y_true=y_true)  # Calcula componente 2 asimetrica por infraestimacion
        peak_component = self._peak_mse(y_pred=y_pred, y_true=y_true)  # Calcula componente 3 MSE enfocado en picos >= P95

        total_loss = (  # Combina componentes segun pesos definidos en la propuesta
            self.huber_weight * huber_component  # Aplica peso 0.55 a Weighted Huber
            + self.asym_weight * asym_component  # Aplica peso 0.30 a penalizacion asimetrica
            + self.peak_weight * peak_component  # Aplica peso 0.15 a Peak MSE
        )
        return total_loss  # Devuelve escalar final de perdida para backpropagation
