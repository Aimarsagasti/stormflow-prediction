"""Composite loss functions for stormflow peak-focused training."""

from __future__ import annotations  # Permite anotaciones modernas con buena compatibilidad

from typing import Dict, Optional  # Define tipos para parametros de normalizacion opcionales

import numpy as np  # Permite convertir umbrales reales a la escala normalizada del entrenamiento
import torch  # Provee operaciones tensoriales para calculo de perdidas
from torch import nn  # Incluye clases base para construir modulos de perdida

from src.pipeline.normalize import normalize_target_values  # Reutiliza la conversion oficial de umbrales al espacio normalizado


class CompositeLoss(nn.Module):
    """Composite loss: weighted huber + asymmetric underprediction + peak mse + base overprediction penalty."""

    def __init__(
        self,
        p95_threshold: float,
        p99_threshold: float,
        p999_threshold: float,
        baseflow_threshold: float = 0.5,
        huber_weight: float = 0.25,
        asym_weight: float = 0.20,
        peak_weight: float = 0.20,
        base_over_weight: float = 0.12,
        tail_focus_weight: float = 2.0,
        huber_beta: float = 1.0,
        norm_params: Optional[Dict[str, object]] = None,
        thresholds_are_normalized: bool = False,
    ) -> None:
        super().__init__()  # Inicializa clase base para registrar el modulo correctamente

        if norm_params is not None and not thresholds_are_normalized:  # Convierte umbrales reales MGD al espacio de entrenamiento si hace falta
            normalized_thresholds = normalize_target_values(  # Usa util oficial del pipeline para respetar log1p y Min-Max del target
                values=np.asarray([p95_threshold, p99_threshold, p999_threshold, baseflow_threshold], dtype=float),  # Empaqueta umbrales crudos para transformarlos juntos
                norm_params=norm_params,  # Pasa parametros de normalizacion que definen la escala del entrenamiento
            )
            p95_threshold = float(normalized_thresholds[0])  # Sustituye P95 por su equivalente en escala normalizada
            p99_threshold = float(normalized_thresholds[1])  # Sustituye P99 por su equivalente en escala normalizada
            p999_threshold = float(normalized_thresholds[2])  # Sustituye P99.9 por su equivalente en escala normalizada
            baseflow_threshold = float(normalized_thresholds[3])  # Sustituye umbral de baseflow por su equivalente en escala normalizada

        self.p95_threshold = float(p95_threshold)  # Guarda umbral P95 ya alineado con la escala de y_true
        self.p99_threshold = float(p99_threshold)  # Guarda umbral P99 ya alineado con la escala de y_true
        self.p999_threshold = float(p999_threshold)  # Guarda umbral P99.9 ya alineado con la escala de y_true
        self.baseflow_threshold = float(baseflow_threshold)  # Guarda umbral de baseflow donde se debe suprimir sobreestimacion
        self.huber_weight = float(huber_weight)  # Guarda peso global de la componente robusta principal
        self.asym_weight = float(asym_weight)  # Guarda peso global de la penalizacion de infraestimacion
        self.peak_weight = float(peak_weight)  # Guarda peso global de la componente Peak MSE
        self.base_over_weight = float(base_over_weight)  # Guarda peso de la penalizacion explicita de sobreestimacion en baseflow
        self.tail_focus_weight = float(tail_focus_weight)  # Guarda cuanto se amplifica de forma continua la cola alta dentro del termino peak
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
        tail_range = max(self.p999_threshold - self.p95_threshold, 1e-6)  # Define un rango minimo estable para medir posicion relativa dentro de la cola alta
        tail_position = torch.clamp((y_true - self.p95_threshold) / tail_range, min=0.0, max=1.0)  # Estima que tan cerca esta cada muestra del extremo superior de la cola
        tail_factor = 1.0 + (self.tail_focus_weight * tail_position)  # Amplifica de forma continua la perdida cuanto mas extrema es la muestra real
        peak_values = squared_error * sample_weights * tail_factor  # Aumenta gradiente en la cola alta sin tocar muestras de baseflow
        return self._masked_mean(peak_values, peak_mask)  # Calcula media solo en muestras de pico relevantes

    def _base_overprediction_penalty(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        base_mask = y_true <= self.baseflow_threshold  # Selecciona muestras en regimen base donde la falsa alarma es mas costosa operativamente
        over_error = torch.relu(y_pred - y_true)  # Conserva solo sobreestimaciones para no castigar ligera subestimacion en base
        threshold_floor = max(self.baseflow_threshold, 1e-6)  # Evita division por cero si el umbral configurado es extremadamente pequeno
        near_zero_factor = torch.clamp((self.baseflow_threshold - y_true) / threshold_floor, min=0.0, max=1.0)  # Incrementa penalizacion cuanto mas cerca de cero este el target real
        suppression_values = (over_error ** 2) * (1.0 + near_zero_factor)  # Penaliza cuadraticamente salidas infladas y refuerza el castigo cerca de cero
        return self._masked_mean(suppression_values, base_mask)  # Promedia solo sobre muestras base para no interferir con la cola alta

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, sample_weights: torch.Tensor) -> torch.Tensor:
        if y_pred.shape != y_true.shape:  # Valida que prediccion y target tengan misma forma (batch, 1)
            raise ValueError("y_pred and y_true must have the same shape")  # Mensaje claro para detectar errores de modelo o batching
        if sample_weights.shape != y_true.shape:  # Valida que pesos por muestra coincidan con forma del target
            raise ValueError("sample_weights must have the same shape as y_true")  # Mensaje claro para detectar desalineacion de DataLoader

        huber_component = self._weighted_huber(y_pred=y_pred, y_true=y_true, sample_weights=sample_weights)  # Calcula componente base robusta ponderada
        asym_component = self._asymmetric_underprediction(y_pred=y_pred, y_true=y_true, sample_weights=sample_weights)  # Calcula penalizacion asimetrica de infraestimacion en cola alta
        peak_component = self._peak_mse(y_pred=y_pred, y_true=y_true, sample_weights=sample_weights)  # Calcula Peak MSE ponderado con enfasis continuo dentro de la cola alta
        base_over_component = self._base_overprediction_penalty(y_pred=y_pred, y_true=y_true)  # Calcula penalizacion de falsas alarmas cuando el target esta en baseflow

        total_loss = (  # Combina los terminos solicitados para priorizar captura de picos sin perder control en baseflow
            self.huber_weight * huber_component  # Mantiene base robusta para toda la distribucion
            + self.asym_weight * asym_component  # Penaliza fuertemente infraestimaciones en eventos severos
            + self.peak_weight * peak_component  # Refuerza ajuste de magnitud en cola alta
            + self.base_over_weight * base_over_component  # Suprime sobreestimacion sistematica en regimen base sin tocar la arquitectura
        )
        return total_loss  # Devuelve escalar final de perdida para backpropagation


class TwoStageLoss(nn.Module):
    """Loss para entrenamiento two-stage con clasificacion y regresion separadas."""

    def __init__(
        self,
        event_threshold: float = 0.5,
        norm_params: Dict[str, object] | None = None,
        cls_weight: float = 0.3,
        reg_weight: float = 0.7,
    ) -> None:
        super().__init__()  # Inicializa clase base para registrar el modulo correctamente
        if norm_params is None:  # Verifica que existan parametros de normalizacion para desnormalizar y_true
            raise ValueError("norm_params must be provided for TwoStageLoss")  # Falla temprano si falta informacion critica

        self.event_threshold = float(event_threshold)  # Guarda umbral real en MGD para definir evento
        self.norm_params = norm_params  # Conserva parametros de normalizacion para desnormalizar dentro de la loss
        self.cls_weight = float(cls_weight)  # Guarda peso del componente de clasificacion en la perdida total
        self.reg_weight = float(reg_weight)  # Guarda peso del componente de regresion en la perdida total
        self.huber_base = nn.SmoothL1Loss(reduction="none")  # Define Huber elemento a elemento para poder ponderar por muestra
        self.last_cls_loss: torch.Tensor | None = None  # Guarda ultima loss de clasificacion para logging en el trainer
        self.last_reg_loss: torch.Tensor | None = None  # Guarda ultima loss de regresion para logging en el trainer

    def _denormalize_target(self, y_true: torch.Tensor) -> torch.Tensor:
        target_col = str(self.norm_params["target_col"])  # Recupera nombre del target para acceder a sus parametros
        target_mean = float(self.norm_params["mean"][target_col])  # Extrae media usada en z-score del target
        target_std = float(self.norm_params["std"][target_col])  # Extrae desviacion usada en z-score del target
        y_real = (y_true * target_std) + target_mean  # Revierte z-score al espacio transformado previo al escalado
        if target_col in self.norm_params.get("log1p_columns", []):  # Verifica si el target usa log1p
            y_real = torch.expm1(y_real)  # Revierte log1p para volver a MGD reales
        y_real = torch.clamp(y_real, min=0.0)  # Impone no negatividad por consistencia fisica del stormflow
        return y_real  # Devuelve valores reales en MGD para definir eventos

    @staticmethod
    def _compute_pos_weight(event_label: torch.Tensor) -> torch.Tensor:
        positives = event_label.sum()  # Cuenta positivos del batch para balancear el BCE
        negatives = event_label.numel() - positives  # Cuenta negativos del batch para balancear el BCE
        if positives > 0:  # Evita division por cero cuando el batch no tiene eventos
            return negatives / positives  # Calcula pos_weight como ratio neg/pos segun practica habitual
        return torch.tensor(1.0, device=event_label.device, dtype=event_label.dtype)  # Fallback neutro si no hay positivos

    def _weighted_bce(self, cls_prob: torch.Tensor, event_label: torch.Tensor) -> torch.Tensor:
        eps = 1e-7  # Define epsilon para evitar log(0) en BCE
        cls_prob = torch.clamp(cls_prob, min=eps, max=1.0 - eps)  # Limita probabilidades para estabilidad numerica
        pos_weight = self._compute_pos_weight(event_label)  # Calcula pos_weight del batch para compensar desbalance
        bce_values = (  # Implementa BCE con pos_weight manual porque usamos probabilidades ya pasadas por sigmoid
            -(pos_weight * event_label * torch.log(cls_prob))  # Penaliza falsos negativos con mayor peso
            - ((1.0 - event_label) * torch.log(1.0 - cls_prob))  # Penaliza falsos positivos con peso normal
        )
        return bce_values.mean()  # Devuelve BCE promedio del batch

    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
        y_true: torch.Tensor,
        sample_weights: torch.Tensor,
    ) -> torch.Tensor:
        if "cls_prob" not in model_output or "reg_value" not in model_output:  # Verifica que el modelo entregue ambas salidas
            raise KeyError("model_output must contain 'cls_prob' and 'reg_value'")  # Falla claro si falta alguna salida

        cls_prob = model_output["cls_prob"]  # Extrae probabilidades de evento desde la cabeza de clasificacion
        reg_value = model_output["reg_value"]  # Extrae magnitud predicha desde la cabeza de regresion

        if cls_prob.shape != y_true.shape:  # Valida forma esperada para clasificacion (batch, 1)
            raise ValueError("cls_prob and y_true must have the same shape")  # Mensaje claro para depurar salidas mal formadas
        if reg_value.shape != y_true.shape:  # Valida forma esperada para regresion (batch, 1)
            raise ValueError("reg_value and y_true must have the same shape")  # Mensaje claro para depurar salidas mal formadas
        if sample_weights.shape != y_true.shape:  # Valida pesos por muestra alineados con el target
            raise ValueError("sample_weights must have the same shape as y_true")  # Mensaje claro para detectar desalineacion del loader

        y_true_real = self._denormalize_target(y_true)  # Desnormaliza y_true para definir eventos en MGD reales
        event_label = (y_true_real > self.event_threshold).float()  # Construye etiqueta binaria de evento con umbral real

        cls_loss = self._weighted_bce(cls_prob=cls_prob, event_label=event_label)  # Calcula BCE con pos_weight dinamico

        event_mask = (event_label == 1.0).squeeze(1)  # Crea mascara booleana para filtrar solo muestras con evento
        if event_mask.any():  # Verifica si hay eventos en el batch antes de calcular la regresion
            reg_pred = reg_value[event_mask]  # Selecciona predicciones de magnitud solo en eventos
            reg_true = y_true[event_mask]  # Selecciona targets normalizados solo en eventos
            reg_weights = sample_weights[event_mask]  # Selecciona pesos por muestra solo en eventos
            huber_values = self.huber_base(reg_pred, reg_true)  # Calcula error Huber por muestra para regresion
            under_mask = reg_pred < reg_true  # Detecta infraestimaciones para penalizarlas mas fuerte
            huber_values = torch.where(under_mask, huber_values * 3.0, huber_values)  # Amplifica errores por infraestimacion
            
            reg_loss = (huber_values * reg_weights).mean()  # Pondera por sample_weights y promedia sobre eventos
        else:  # Si no hay eventos en el batch, no se puede entrenar el regresor
            reg_loss = torch.zeros((), device=y_true.device, dtype=y_true.dtype)  # Usa cero escalar para no afectar el total

        total_loss = (  # Combina perdidas de clasificacion y regresion con pesos configurables
            (self.cls_weight * cls_loss)  # Controla impacto de la cabeza de clasificacion
            + (self.reg_weight * reg_loss)  # Controla impacto de la cabeza de regresion
        )

        self.last_cls_loss = cls_loss.detach()  # Guarda loss de clasificacion para logging externo sin gradiente
        self.last_reg_loss = reg_loss.detach()  # Guarda loss de regresion para logging externo sin gradiente
        return total_loss  # Devuelve perdida total para backpropagation
