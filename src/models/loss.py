"""Composite loss functions for stormflow peak-focused training."""

from __future__ import annotations  # Permite anotaciones modernas con buena compatibilidad

from typing import Dict, Optional  # Define tipos para parametros de normalizacion opcionales

import numpy as np  # Permite convertir umbrales reales a la escala normalizada del entrenamiento
import torch  # Provee operaciones tensoriales para calculo de perdidas
from torch import nn  # Incluye clases base para construir modulos de perdida

from src.pipeline.normalize import normalize_target_values  # Reutiliza la conversion oficial de umbrales al espacio normalizado


class CompositeLoss(nn.Module):
    """Composite loss: weighted huber + asymmetric underprediction + peak mse."""

    def __init__(
        self,
        p95_threshold: float,
        p99_threshold: float,
        p999_threshold: float,
        huber_weight: float = 0.25,
        asym_weight: float = 0.20,
        peak_weight: float = 0.20,
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

        self.p95_threshold = float(p95_threshold)  # Guarda umbral P95 ya alineado con la escala de y_true
        self.p99_threshold = float(p99_threshold)  # Guarda umbral P99 ya alineado con la escala de y_true
        self.p999_threshold = float(p999_threshold)  # Guarda umbral P99.9 ya alineado con la escala de y_true
        self.huber_weight = float(huber_weight)  # Guarda peso global de la componente robusta principal
        self.asym_weight = float(asym_weight)  # Guarda peso global de la penalizacion de infraestimacion
        self.peak_weight = float(peak_weight)  # Guarda peso global de la componente Peak MSE
        self.huber_base = nn.SmoothL1Loss(reduction="none", beta=huber_beta)  # Define Huber elemento a elemento para poder ponderar por muestra

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute the mean over a boolean mask, returning zero if empty."""
        if mask.any():  # Verifica si existe al menos una muestra activa antes de promediar
            return values[mask].mean()  # Calcula media solo sobre las posiciones activas del subconjunto
        return torch.zeros((), device=values.device, dtype=values.dtype)  # Retorna cero escalar cuando el subconjunto esta vacio

    def _weighted_huber(self, y_pred: torch.Tensor, y_true: torch.Tensor, sample_weights: torch.Tensor) -> torch.Tensor:
        huber_per_sample = self.huber_base(y_pred, y_true)  # Calcula error Huber por muestra y dimension
        weighted_huber = huber_per_sample * sample_weights  # Aplica peso de muestra proveniente del DataLoader
        return weighted_huber.mean()  # Promedia sobre batch para obtener escalar de la componente

    def _asymmetric_underprediction(self, y_pred: torch.Tensor, y_true: torch.Tensor, sample_weights: torch.Tensor) -> torch.Tensor:
        focus_mask = y_true >= self.p95_threshold  # Enfoca la penalizacion solo en cola alta definida por el target real
        under_error = torch.relu(y_true - y_pred)  # Conserva solo infraestimaciones y anula sobreestimaciones
        severity_factor = torch.ones_like(y_true) * 2.0  # Usa un factor base de 2x para muestras entre P95 y P99
        severity_factor = torch.where(y_true >= self.p99_threshold, torch.full_like(y_true, 4.0), severity_factor)  # Escala a 4x en muestras de cola muy alta
        severity_factor = torch.where(y_true >= self.p999_threshold, torch.full_like(y_true, 6.0), severity_factor)  # Escala a 6x en muestras extremas criticas
        asym_values = (under_error ** 2) * severity_factor * sample_weights  # Combina magnitud del error, severidad y ponderacion por muestra
        return self._masked_mean(asym_values, focus_mask)  # Promedia solo sobre muestras en cola alta para evitar ruido de baseflow

    def _peak_mse(self, y_pred: torch.Tensor, y_true: torch.Tensor, sample_weights: torch.Tensor) -> torch.Tensor:
        peak_mask = y_true >= self.p95_threshold  # Selecciona picos usando directamente el target real y no los pesos
        squared_error = (y_pred - y_true) ** 2  # Calcula error cuadratico por muestra para enfatizar desviaciones grandes
        peak_values = squared_error * sample_weights  # Pondera el error cuadratico para mantener prioridad operacional en cola alta
        return self._masked_mean(peak_values, peak_mask)  # Calcula media solo en muestras de pico relevantes

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, sample_weights: torch.Tensor) -> torch.Tensor:
        if y_pred.shape != y_true.shape:  # Valida que prediccion y target tengan misma forma (batch, 1)
            raise ValueError("y_pred and y_true must have the same shape")  # Mensaje claro para detectar errores de modelo o batching
        if sample_weights.shape != y_true.shape:  # Valida que pesos por muestra coincidan con forma del target
            raise ValueError("sample_weights must have the same shape as y_true")  # Mensaje claro para detectar desalineacion de DataLoader

        huber_component = self._weighted_huber(y_pred=y_pred, y_true=y_true, sample_weights=sample_weights)  # Calcula componente base robusta ponderada
        asym_component = self._asymmetric_underprediction(y_pred=y_pred, y_true=y_true, sample_weights=sample_weights)  # Calcula penalizacion asimetrica de infraestimacion en cola alta
        peak_component = self._peak_mse(y_pred=y_pred, y_true=y_true, sample_weights=sample_weights)  # Calcula Peak MSE ponderado solo sobre >= P95

        total_loss = (  # Combina los tres terminos solicitados para priorizar captura de picos sin perder robustez global
            self.huber_weight * huber_component  # Mantiene base robusta para toda la distribucion
            + self.asym_weight * asym_component  # Penaliza fuertemente infraestimaciones en eventos severos
            + self.peak_weight * peak_component  # Refuerza ajuste de magnitud en cola alta
        )
        return total_loss  # Devuelve escalar final de perdida para backpropagation