"""Composite loss functions for stormflow peak-focused training."""

from __future__ import annotations  # Permite anotaciones modernas con buena compatibilidad

from typing import Dict, Optional  # Define tipos para parametros de normalizacion opcionales

import numpy as np  # Permite convertir umbrales reales a la escala normalizada del entrenamiento
import torch  # Provee operaciones tensoriales para calculo de perdidas
from torch import nn  # Incluye clases base para construir modulos de perdida

from src.pipeline.normalize import normalize_target_values  # Reutiliza la conversion oficial de umbrales al espacio normalizado


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
        norm_params: Optional[Dict[str, object]] = None,
        thresholds_are_normalized: bool = False,
    ) -> None:
        super().__init__()  # Inicializa clase base para registrar el modulo correctamente

        if norm_params is not None and not thresholds_are_normalized:  # Convierte umbrales reales MGD al espacio de entrenamiento si hace falta
            normalized_thresholds = normalize_target_values(  # Usa util oficial del pipeline para respetar log1p y Min-Max del target
                values=np.asarray([p95_threshold, p99_threshold, p999_threshold], dtype=float),  # Empaqueta umbrales crudos para transformarlos juntos
                norm_params=norm_params,  # Pasa parametros de normalizacion que definen la escala del entrenamiento
            )
            p95_threshold = float(normalized_thresholds[0])  # Sustituye P95 por su equivalente en escala normalizada
            p99_threshold = float(normalized_thresholds[1])  # Sustituye P99 por su equivalente en escala normalizada
            p999_threshold = float(normalized_thresholds[2])  # Sustituye P99.9 por su equivalente en escala normalizada

        self.p95_threshold = float(p95_threshold)  # Guarda umbral P95 ya alineado con la escala de y_true como fallback diagnostico
        self.p99_threshold = float(p99_threshold)  # Guarda umbral P99 ya alineado con la escala de y_true como fallback diagnostico
        self.p999_threshold = float(p999_threshold)  # Guarda umbral P99.9 ya alineado con la escala de y_true como fallback diagnostico
        self.huber_weight = float(huber_weight)  # Guarda peso global de la componente Weighted Huber
        self.asym_weight = float(asym_weight)  # Guarda peso global de la componente asimetrica
        self.peak_weight = float(peak_weight)  # Guarda peso global de la componente Peak MSE
        self.huber_base = nn.SmoothL1Loss(reduction="none", beta=huber_beta)  # Define Huber elemento a elemento para poder ponderar por muestra

    def _weighted_huber(self, y_pred: torch.Tensor, y_true: torch.Tensor, sample_weights: torch.Tensor) -> torch.Tensor:
        huber_per_sample = self.huber_base(y_pred, y_true)  # Calcula error Huber por muestra y dimension
        weighted_huber = huber_per_sample * sample_weights  # Aplica peso de magnitud proveniente del DataLoader
        return weighted_huber.mean()  # Promedia sobre batch para obtener escalar de la componente

    def _asymmetric_underprediction(self, y_pred: torch.Tensor, y_true: torch.Tensor, sample_weights: torch.Tensor) -> torch.Tensor:
        under_error = torch.relu(y_true - y_pred)  # Conserva solo infraestimaciones y anula sobreestimaciones
        asym_factor = torch.full_like(y_true, 2.0)  # Usa 2x como factor base para muestras por debajo de P99
        asym_factor = torch.where(sample_weights >= 10.0, torch.full_like(y_true, 4.0), asym_factor)  # Escala a 4x cuando la muestra pertenece a la cola >= P99
        asym_factor = torch.where(sample_weights >= 20.0, torch.full_like(y_true, 6.0), asym_factor)  # Escala a 6x cuando la muestra pertenece a la cola >= P99.9
        asym_loss = (under_error ** 2) * asym_factor * sample_weights  # Integra severidad y peso de magnitud en la penalizacion de infraestimacion
        return asym_loss.mean()  # Promedia sobre batch para obtener escalar estable

    def _peak_mse(self, y_pred: torch.Tensor, y_true: torch.Tensor, sample_weights: torch.Tensor) -> torch.Tensor:
        peak_mask = sample_weights >= 4.0  # Usa sample_weight como criterio robusto para detectar muestras >= P95
        if peak_mask.any():  # Verifica si hay picos en el batch para evitar media sobre tensor vacio
            peak_error = y_pred[peak_mask] - y_true[peak_mask]  # Calcula residuo solo en zona de picos relevantes
            peak_weights = sample_weights[peak_mask]  # Recupera pesos de magnitud solo para las muestras de pico seleccionadas
            return ((peak_error ** 2) * peak_weights).mean()  # Calcula MSE ponderado de picos como escalar
        return torch.zeros((), device=y_true.device, dtype=y_true.dtype)  # Retorna cero en batches sin picos para no contaminar la perdida

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, sample_weights: torch.Tensor) -> torch.Tensor:
        if y_pred.shape != y_true.shape:  # Valida que prediccion y target tengan misma forma (batch, 1)
            raise ValueError("y_pred and y_true must have the same shape")  # Mensaje claro para detectar errores de modelo/dataloader
        if sample_weights.shape != y_true.shape:  # Valida que pesos por muestra coincidan con forma del target
            raise ValueError("sample_weights must have the same shape as y_true")  # Mensaje claro para detectar errores de batching

        huber_component = self._weighted_huber(y_pred=y_pred, y_true=y_true, sample_weights=sample_weights)  # Calcula componente 1 Weighted Huber
        asym_component = self._asymmetric_underprediction(y_pred=y_pred, y_true=y_true, sample_weights=sample_weights)  # Calcula componente 2 asimetrica con severidad derivada del peso de muestra
        peak_component = self._peak_mse(y_pred=y_pred, y_true=y_true, sample_weights=sample_weights)  # Calcula componente 3 MSE enfocado en muestras >= P95

        total_loss = (  # Combina componentes segun pesos definidos en la propuesta
            self.huber_weight * huber_component  # Aplica peso 0.55 a Weighted Huber
            + self.asym_weight * asym_component  # Aplica peso 0.30 a penalizacion asimetrica
            + self.peak_weight * peak_component  # Aplica peso 0.15 a Peak MSE
        )
        return total_loss  # Devuelve escalar final de perdida para backpropagation
