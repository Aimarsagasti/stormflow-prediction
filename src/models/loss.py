"""Composite loss functions for stormflow peak-focused training."""

from __future__ import annotations  # Permite anotaciones modernas con buena compatibilidad

from typing import Dict, Optional  # Define tipos para parametros de normalizacion y salidas opcionales

import numpy as np  # Permite convertir umbrales reales a la escala normalizada del entrenamiento
import torch  # Provee operaciones tensoriales para calculo de perdidas
from torch import nn  # Incluye clases base para construir modulos de perdida
from torch.nn import functional as F  # Aporta primitivas numericamente estables para BCE y activaciones

from src.pipeline.normalize import normalize_target_values  # Reutiliza la conversion oficial de umbrales al espacio normalizado


class CompositeLoss(nn.Module):
    """Composite loss: robust regression + event BCE + peak/base penalties."""

    def __init__(
        self,
        p95_threshold: float,
        p99_threshold: float,
        p999_threshold: float,
        base_threshold: float = 0.5,
        huber_weight: float = 0.30,
        asym_weight: float = 0.20,
        peak_weight: float = 0.20,
        event_weight: float = 0.20,
        false_positive_weight: float = 0.10,
        huber_beta: float = 1.0,
        event_pos_weight: float = 1.0,
        norm_params: Optional[Dict[str, object]] = None,
        thresholds_are_normalized: bool = False,
    ) -> None:
        super().__init__()  # Inicializa clase base para registrar el modulo correctamente

        if norm_params is not None and not thresholds_are_normalized:  # Convierte umbrales reales MGD al espacio de entrenamiento si hace falta
            normalized_thresholds = normalize_target_values(  # Usa util oficial del pipeline para respetar log1p y Min-Max del target
                values=np.asarray([base_threshold, p95_threshold, p99_threshold, p999_threshold], dtype=float),  # Empaqueta umbrales crudos para transformarlos juntos
                norm_params=norm_params,  # Pasa parametros de normalizacion que definen la escala del entrenamiento
            )
            base_threshold = float(normalized_thresholds[0])  # Sustituye umbral base por su equivalente en escala normalizada
            p95_threshold = float(normalized_thresholds[1])  # Sustituye P95 por su equivalente en escala normalizada
            p99_threshold = float(normalized_thresholds[2])  # Sustituye P99 por su equivalente en escala normalizada
            p999_threshold = float(normalized_thresholds[3])  # Sustituye P99.9 por su equivalente en escala normalizada

        self.base_threshold = float(base_threshold)  # Guarda umbral base ya alineado con la escala de y_true
        self.p95_threshold = float(p95_threshold)  # Guarda umbral P95 ya alineado con la escala de y_true
        self.p99_threshold = float(p99_threshold)  # Guarda umbral P99 ya alineado con la escala de y_true
        self.p999_threshold = float(p999_threshold)  # Guarda umbral P99.9 ya alineado con la escala de y_true
        self.huber_weight = float(huber_weight)  # Guarda peso global de la componente robusta principal
        self.asym_weight = float(asym_weight)  # Guarda peso global de la penalizacion de infraestimacion
        self.peak_weight = float(peak_weight)  # Guarda peso global de la componente Peak MSE
        self.event_weight = float(event_weight)  # Guarda peso global de la supervision de evento
        self.false_positive_weight = float(false_positive_weight)  # Guarda peso global de la penalizacion de falsas alarmas
        self.huber_base = nn.SmoothL1Loss(reduction="none", beta=huber_beta)  # Define Huber elemento a elemento para poder ponderar por muestra
        self.register_buffer(  # Registra el peso positivo de BCE como buffer para moverlo junto al modulo
            "event_pos_weight_tensor",  # Define nombre del buffer persistente del modulo
            torch.tensor([float(event_pos_weight)], dtype=torch.float32),  # Guarda un tensor 1D compatible con BCEWithLogits
        )

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute the mean over a boolean mask, returning zero if empty."""
        if mask.any():  # Verifica si existe al menos una muestra activa antes de promediar
            return values[mask].mean()  # Calcula media solo sobre las posiciones activas del subconjunto
        return torch.zeros((), device=values.device, dtype=values.dtype)  # Retorna cero escalar cuando el subconjunto esta vacio

    def _resolve_outputs(self, model_output: torch.Tensor | Dict[str, torch.Tensor]) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Extract regression prediction and event logit from either tensor or dict output."""
        if isinstance(model_output, dict):  # Soporta el nuevo modelo multitarea que retorna varias salidas
            if "stormflow_prediction" not in model_output:  # Valida clave requerida para la tarea de regresion principal
                raise ValueError("model_output dict must include 'stormflow_prediction'")  # Lanza error claro si la salida principal falta
            y_pred = model_output["stormflow_prediction"]  # Recupera la prediccion final ya gateada por evento
            event_logit = model_output.get("event_logit")  # Recupera logit de evento cuando el modelo expone la rama auxiliar
            return y_pred, event_logit  # Devuelve ambas piezas para calcular la loss compuesta
        return model_output, None  # Mantiene compatibilidad con modelos antiguos de salida unica

    def _weighted_huber(self, y_pred: torch.Tensor, y_true: torch.Tensor, sample_weights: torch.Tensor) -> torch.Tensor:
        huber_per_sample = self.huber_base(y_pred, y_true)  # Calcula error Huber por muestra y dimension
        weighted_huber = huber_per_sample * sample_weights  # Aplica peso de magnitud proveniente del DataLoader
        return weighted_huber.mean()  # Promedia sobre batch para obtener escalar de la componente

    def _asymmetric_underprediction(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        sample_weights: torch.Tensor,
        event_targets: torch.Tensor,
    ) -> torch.Tensor:
        under_error = torch.relu(y_true - y_pred)  # Conserva solo infraestimaciones y anula sobreestimaciones
        asym_factor = torch.full_like(y_true, 2.0)  # Usa 2x como factor base para muestras por debajo de P99
        asym_factor = torch.where(sample_weights >= 12.0, torch.full_like(y_true, 4.0), asym_factor)  # Escala a 4x cuando la muestra pertenece a la cola >= P99
        asym_factor = torch.where(sample_weights >= 20.0, torch.full_like(y_true, 6.0), asym_factor)  # Escala a 6x cuando la muestra pertenece a la cola >= P99.9
        focus_mask = (event_targets >= 0.5) | (sample_weights >= 6.0)  # Enfoca la penalizacion donde importa mas el timing/magnitud de evento
        asym_values = (under_error ** 2) * asym_factor * sample_weights  # Integra severidad y peso de magnitud en la penalizacion de infraestimacion
        return self._masked_mean(asym_values, focus_mask)  # Promedia solo sobre muestras de evento o cola alta para evitar ruido en baseflow

    def _peak_mse(self, y_pred: torch.Tensor, y_true: torch.Tensor, sample_weights: torch.Tensor) -> torch.Tensor:
        peak_mask = sample_weights >= 6.0  # Usa el peso de muestra como criterio robusto para detectar ventanas >= P95
        peak_error = (y_pred - y_true) ** 2  # Calcula error cuadratico por muestra para usarlo en el subconjunto de picos
        peak_values = peak_error * sample_weights  # Repondera el error cuadratico segun severidad del target futuro
        return self._masked_mean(peak_values, peak_mask)  # Calcula MSE ponderado solo sobre muestras de pico relevantes

    def _false_positive_penalty(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        event_targets: torch.Tensor,
    ) -> torch.Tensor:
        base_mask = (event_targets < 0.5) & (y_true <= self.base_threshold)  # Selecciona no-eventos o casi-cero donde las falsas alarmas son el fallo dominante
        over_error = torch.relu(y_pred - y_true)  # Conserva solo sobreestimaciones para castigar falsas activaciones en baseflow
        penalty_values = over_error ** 2  # Usa penalizacion cuadratica para que errores grandes cerca de cero cuesten mucho mas
        return self._masked_mean(penalty_values, base_mask)  # Promedia solo sobre el subconjunto base/no-evento relevante

    def _event_bce(self, event_logit: Optional[torch.Tensor], event_targets: torch.Tensor) -> torch.Tensor:
        if event_logit is None:  # Mantiene compatibilidad con modelos antiguos que no exponen logit de evento
            return torch.zeros((), device=event_targets.device, dtype=event_targets.dtype)  # Retorna cero para no afectar la loss total
        bce_per_sample = F.binary_cross_entropy_with_logits(  # Calcula BCE estable numercamente para la rama de evento
            event_logit,  # Usa logit crudo en vez de probabilidad para mejor estabilidad numerica
            event_targets,  # Usa etiqueta booleana de evento alineada al horizonte objetivo
            pos_weight=self.event_pos_weight_tensor.to(event_logit.device),  # Compensa el desbalance natural entre eventos y no-eventos
            reduction="none",  # Mantiene perdida por muestra para poder promediar explicitamente
        )
        return bce_per_sample.mean()  # Promedia sobre batch para obtener escalar estable de supervision auxiliar

    def forward(
        self,
        model_output: torch.Tensor | Dict[str, torch.Tensor],
        y_true: torch.Tensor,
        sample_weights: torch.Tensor,
        event_targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        y_pred, event_logit = self._resolve_outputs(model_output=model_output)  # Extrae prediccion continua y logit de evento desde la salida del modelo
        if y_pred.shape != y_true.shape:  # Valida que prediccion y target tengan misma forma (batch, 1)
            raise ValueError("y_pred and y_true must have the same shape")  # Mensaje claro para detectar errores de modelo/dataloader
        if sample_weights.shape != y_true.shape:  # Valida que pesos por muestra coincidan con forma del target
            raise ValueError("sample_weights must have the same shape as y_true")  # Mensaje claro para detectar errores de batching

        if event_targets is None:  # Ofrece fallback razonable si el llamador aun no pasa la rama auxiliar explicitamente
            event_targets = (y_true > self.base_threshold).to(dtype=y_true.dtype)  # Aproxima evento a partir del target cuando no existe supervision explicita
        if event_targets.shape != y_true.shape:  # Valida que la supervision de evento tenga la misma estructura que el target continuo
            raise ValueError("event_targets must have the same shape as y_true")  # Lanza error claro si el DataLoader esta desalineado

        huber_component = self._weighted_huber(y_pred=y_pred, y_true=y_true, sample_weights=sample_weights)  # Calcula componente 1 de regresion robusta ponderada
        asym_component = self._asymmetric_underprediction(  # Calcula componente 2 enfocada en infraestimaciones de eventos y colas altas
            y_pred=y_pred,  # Pasa prediccion final gateada del modelo
            y_true=y_true,  # Pasa target continuo alineado al horizonte
            sample_weights=sample_weights,  # Pasa pesos por severidad para reforzar extremos
            event_targets=event_targets,  # Pasa etiqueta de evento para enfocar la penalizacion donde importa operativamente
        )
        peak_component = self._peak_mse(y_pred=y_pred, y_true=y_true, sample_weights=sample_weights)  # Calcula componente 3 MSE enfocado en muestras >= P95
        event_component = self._event_bce(event_logit=event_logit, event_targets=event_targets)  # Calcula componente 4 BCE para activar/desactivar eventos correctamente
        false_positive_component = self._false_positive_penalty(  # Calcula componente 5 para castigar falsas alarmas cerca de cero
            y_pred=y_pred,  # Usa la prediccion final ya gateada del modelo
            y_true=y_true,  # Usa target continuo real para medir sobreestimacion
            event_targets=event_targets,  # Usa bandera de evento para limitar la penalizacion a base/no-evento
        )

        total_loss = (  # Combina componentes segun pesos definidos para equilibrar baseflow, evento y extremos
            self.huber_weight * huber_component  # Mantiene una base robusta para toda la distribucion del target
            + self.asym_weight * asym_component  # Refuerza que infraestimar eventos severos cueste mas que sobreestimar
            + self.peak_weight * peak_component  # Mantiene foco adicional en la cola alta del stormflow
            + self.event_weight * event_component  # Supervisa explicitamente la activacion correcta de eventos futuros
            + self.false_positive_weight * false_positive_component  # Castiga el patron observado de falsas alarmas cerca de cero
        )
        return total_loss  # Devuelve escalar final de perdida para backpropagation
