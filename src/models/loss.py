"""Composite loss functions for stormflow peak-focused training."""

from __future__ import annotations  # Allows modern annotations with good compatibility

from typing import Dict, Optional  # Defines types for optional normalization parameters

import numpy as np  # Lets us convert real thresholds to the normalized training scale
import torch  # Provides tensor operations for loss computation
from torch import nn  # Includes base classes to build loss modules

from src.pipeline.normalize import normalize_target_values  # Reuses the official conversion of thresholds into normalized space


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
        super().__init__()  # Initializes base class so the module is registered correctly

        if norm_params is not None and not thresholds_are_normalized:  # Converts real MGD thresholds to training space when needed
            normalized_thresholds = normalize_target_values(  # Uses the official pipeline utility to respect target log1p and scaling
                values=np.asarray([p95_threshold, p99_threshold, p999_threshold, baseflow_threshold], dtype=float),  # Packs raw thresholds to transform them together
                norm_params=norm_params,  # Passes normalization parameters that define the training scale
            )
            p95_threshold = float(normalized_thresholds[0])  # Replaces P95 with its equivalent in normalized scale
            p99_threshold = float(normalized_thresholds[1])  # Replaces P99 with its equivalent in normalized scale
            p999_threshold = float(normalized_thresholds[2])  # Replaces P99.9 with its equivalent in normalized scale
            baseflow_threshold = float(normalized_thresholds[3])  # Replaces baseflow threshold with its equivalent in normalized scale

        self.p95_threshold = float(p95_threshold)  # Stores P95 threshold already aligned with y_true scale
        self.p99_threshold = float(p99_threshold)  # Stores P99 threshold already aligned with y_true scale
        self.p999_threshold = float(p999_threshold)  # Stores P99.9 threshold already aligned with y_true scale
        self.baseflow_threshold = float(baseflow_threshold)  # Stores baseflow threshold where overestimation should be suppressed
        self.huber_weight = float(huber_weight)  # Stores global weight of the main robust component
        self.asym_weight = float(asym_weight)  # Stores global weight of the underestimation penalty
        self.peak_weight = float(peak_weight)  # Stores global weight of the Peak MSE component
        self.base_over_weight = float(base_over_weight)  # Stores weight of the explicit overestimation penalty in baseflow
        self.tail_focus_weight = float(tail_focus_weight)  # Stores how much the high tail is continuously amplified inside the peak term
        self.huber_base = nn.SmoothL1Loss(reduction="none", beta=huber_beta)  # Defines elementwise Huber so it can be weighted per sample

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute the mean over a boolean mask, returning zero if empty."""
        if mask.any():  # Checks whether there is at least one active sample before averaging
            return values[mask].mean()  # Computes mean only over active positions in the subset
        return torch.zeros((), device=values.device, dtype=values.dtype)  # Returns scalar zero when the subset is empty

    def _weighted_huber(self, y_pred: torch.Tensor, y_true: torch.Tensor, sample_weights: torch.Tensor) -> torch.Tensor:
        huber_per_sample = self.huber_base(y_pred, y_true)  # Computes Huber error per sample and dimension
        weighted_huber = huber_per_sample * sample_weights  # Applies sample weight coming from the DataLoader
        return weighted_huber.mean()  # Averages across the batch to get the scalar component

    def _asymmetric_underprediction(self, y_pred: torch.Tensor, y_true: torch.Tensor, sample_weights: torch.Tensor) -> torch.Tensor:
        focus_mask = y_true >= self.p95_threshold  # Focuses the penalty only on the high tail defined by the real target
        under_error = torch.relu(y_true - y_pred)  # Keeps only underestimations and zeroes overestimations
        severity_factor = torch.ones_like(y_true) * 2.0  # Uses a base 2x factor for samples between P95 and P99
        severity_factor = torch.where(y_true >= self.p99_threshold, torch.full_like(y_true, 4.0), severity_factor)  # Scales up to 4x for very high-tail samples
        severity_factor = torch.where(y_true >= self.p999_threshold, torch.full_like(y_true, 6.0), severity_factor)  # Scales up to 6x for critical extreme samples
        asym_values = (under_error ** 2) * severity_factor * sample_weights  # Combines error magnitude, severity, and sample weighting
        return self._masked_mean(asym_values, focus_mask)  # Averages only over high-tail samples to avoid baseflow noise

    def _peak_mse(self, y_pred: torch.Tensor, y_true: torch.Tensor, sample_weights: torch.Tensor) -> torch.Tensor:
        peak_mask = y_true >= self.p95_threshold  # Selects peaks directly from the real target and not from the weights
        squared_error = (y_pred - y_true) ** 2  # Computes squared error per sample to emphasize large deviations
        tail_range = max(self.p999_threshold - self.p95_threshold, 1e-6)  # Defines a stable minimum range to measure relative position inside the high tail
        tail_position = torch.clamp((y_true - self.p95_threshold) / tail_range, min=0.0, max=1.0)  # Estimates how close each sample is to the upper extreme of the tail
        tail_factor = 1.0 + (self.tail_focus_weight * tail_position)  # Continuously amplifies loss the more extreme the real sample is
        peak_values = squared_error * sample_weights * tail_factor  # Increases gradient in the high tail without touching baseflow samples
        return self._masked_mean(peak_values, peak_mask)  # Computes mean only on relevant peak samples

    def _base_overprediction_penalty(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        base_mask = y_true <= self.baseflow_threshold  # Selects samples in base regime where false alarms are more operationally costly
        over_error = torch.relu(y_pred - y_true)  # Keeps only overestimations so slight underestimation in base is not punished
        threshold_floor = max(self.baseflow_threshold, 1e-6)  # Avoids division by zero if the configured threshold is extremely small
        near_zero_factor = torch.clamp((self.baseflow_threshold - y_true) / threshold_floor, min=0.0, max=1.0)  # Increases penalty the closer the real target is to zero
        suppression_values = (over_error ** 2) * (1.0 + near_zero_factor)  # Penalizes inflated outputs quadratically and reinforces punishment near zero
        return self._masked_mean(suppression_values, base_mask)  # Averages only over base samples so it does not interfere with the high tail

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, sample_weights: torch.Tensor) -> torch.Tensor:
        if y_pred.shape != y_true.shape:  # Validates that prediction and target have the same shape (batch, 1)
            raise ValueError("y_pred and y_true must have the same shape")  # Clear message to detect model or batching errors
        if sample_weights.shape != y_true.shape:  # Validates that sample weights match the target shape
            raise ValueError("sample_weights must have the same shape as y_true")  # Clear message to detect DataLoader misalignment

        huber_component = self._weighted_huber(y_pred=y_pred, y_true=y_true, sample_weights=sample_weights)  # Computes weighted robust base component
        asym_component = self._asymmetric_underprediction(y_pred=y_pred, y_true=y_true, sample_weights=sample_weights)  # Computes asymmetric underestimation penalty in the high tail
        peak_component = self._peak_mse(y_pred=y_pred, y_true=y_true, sample_weights=sample_weights)  # Computes weighted Peak MSE with continuous emphasis inside the high tail
        base_over_component = self._base_overprediction_penalty(y_pred=y_pred, y_true=y_true)  # Computes false-alarm penalty when the target is in baseflow

        total_loss = (  # Combines requested terms to prioritize peak capture without losing control in baseflow
            self.huber_weight * huber_component  # Keeps a robust base across the full distribution
            + self.asym_weight * asym_component  # Strongly penalizes underestimation in severe events
            + self.peak_weight * peak_component  # Reinforces magnitude fitting in the high tail
            + self.base_over_weight * base_over_component  # Suppresses systematic overestimation in base regime without touching the architecture
        )
        return total_loss  # Returns final scalar loss for backpropagation


class TwoStageLoss(nn.Module):
    """Loss for two-stage training with separate classification and regression."""

    def __init__(
        self,
        event_threshold: float = 0.5,
        norm_params: Dict[str, object] | None = None,
        cls_weight: float = 0.3,
        reg_weight: float = 0.7,
    ) -> None:
        super().__init__()  # Initializes base class so the module is registered correctly
        if norm_params is None:  # Verifies that normalization parameters exist to denormalize y_true
            raise ValueError("norm_params must be provided for TwoStageLoss")  # Fails early if critical information is missing

        self.event_threshold = float(event_threshold)  # Stores real threshold in MGD to define an event
        self.norm_params = norm_params  # Keeps normalization parameters to denormalize inside the loss
        self.cls_weight = float(cls_weight)  # Stores weight of the classification component in total loss
        self.reg_weight = float(reg_weight)  # Stores weight of the regression component in total loss
        self.huber_base = nn.SmoothL1Loss(reduction="none")  # Defines elementwise Huber so it can be weighted per sample
        self.last_cls_loss: torch.Tensor | None = None  # Stores last classification loss for logging in the trainer
        self.last_reg_loss: torch.Tensor | None = None  # Stores last regression loss for logging in the trainer

    def _denormalize_target(self, y_true: torch.Tensor) -> torch.Tensor:
        target_col = str(self.norm_params["target_col"])  # Retrieves target name to access its parameters
        target_mean = float(self.norm_params["mean"][target_col])  # Extracts mean used in target z-score
        target_std = float(self.norm_params["std"][target_col])  # Extracts std used in target z-score
        y_real = (y_true * target_std) + target_mean  # Reverses z-score to the transformed space before scaling
        if target_col in self.norm_params.get("log1p_columns", []):  # Checks whether the target uses log1p
            y_real = torch.expm1(y_real)  # Reverses log1p to return to real MGD
        y_real = torch.clamp(y_real, min=0.0)  # Enforces non-negativity for physical stormflow consistency
        return y_real  # Returns real values in MGD to define events

    @staticmethod
    def _compute_pos_weight(event_label: torch.Tensor) -> torch.Tensor:
        positives = event_label.sum()  # Counts positives in the batch to balance BCE
        negatives = event_label.numel() - positives  # Counts negatives in the batch to balance BCE
        if positives > 0:  # Avoids division by zero when the batch has no events
            return negatives / positives  # Computes pos_weight as neg/pos ratio following standard practice
        return torch.tensor(1.0, device=event_label.device, dtype=event_label.dtype)  # Neutral fallback when there are no positives

    def _weighted_bce(self, cls_prob: torch.Tensor, event_label: torch.Tensor) -> torch.Tensor:
        eps = 1e-7  # Defines epsilon to avoid log(0) in BCE
        cls_prob = torch.clamp(cls_prob, min=eps, max=1.0 - eps)  # Clamps probabilities for numerical stability
        pos_weight = self._compute_pos_weight(event_label)  # Computes batch pos_weight to compensate imbalance
        bce_values = (  # Implements BCE with manual pos_weight because we use probabilities already passed through sigmoid
            -(pos_weight * event_label * torch.log(cls_prob))  # Penalizes false negatives with greater weight
            - ((1.0 - event_label) * torch.log(1.0 - cls_prob))  # Penalizes false positives with normal weight
        )
        return bce_values.mean()  # Returns mean BCE for the batch

    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
        y_true: torch.Tensor,
        sample_weights: torch.Tensor,
    ) -> torch.Tensor:
        if "cls_prob" not in model_output or "reg_value" not in model_output:  # Verifies that the model returns both outputs
            raise KeyError("model_output must contain 'cls_prob' and 'reg_value'")  # Fails clearly if any output is missing

        cls_prob = model_output["cls_prob"]  # Extracts event probabilities from the classification head
        reg_value = model_output["reg_value"]  # Extracts predicted magnitude from the regression head

        if cls_prob.shape != y_true.shape:  # Validates expected classification shape (batch, 1)
            raise ValueError("cls_prob and y_true must have the same shape")  # Clear message to debug malformed outputs
        if reg_value.shape != y_true.shape:  # Validates expected regression shape (batch, 1)
            raise ValueError("reg_value and y_true must have the same shape")  # Clear message to debug malformed outputs
        if sample_weights.shape != y_true.shape:  # Validates sample weights aligned with the target
            raise ValueError("sample_weights must have the same shape as y_true")  # Clear message to detect loader misalignment

        y_true_real = self._denormalize_target(y_true)  # Denormalizes y_true to define events in real MGD
        event_label = (y_true_real > self.event_threshold).float()  # Builds binary event label with real threshold

        cls_loss = self._weighted_bce(cls_prob=cls_prob, event_label=event_label)  # Computes BCE with dynamic pos_weight

        event_mask = (event_label == 1.0).squeeze(1)  # Creates boolean mask to filter only event samples
        if event_mask.any():  # Checks whether there are events in the batch before computing regression
            reg_pred = reg_value[event_mask]  # Selects magnitude predictions only for events
            reg_true = y_true[event_mask]  # Selects normalized targets only for events
            reg_weights = sample_weights[event_mask]  # Selects sample weights only for events
            reg_true_real = y_true_real[event_mask]  # Recovers real magnitude in MGD only for events to decide physical severity
            reg_pred_real = self._denormalize_target(reg_pred)  # Denormalizes regressor prediction to measure relative overshoot in interpretable units
            huber_values = self.huber_base(reg_pred, reg_true)  # Computes Huber error per sample for regression
            under_mask = reg_pred_real < reg_true_real  # Detects underestimations in real space to preserve hydrologic interpretation
            over_mask = reg_pred_real > reg_true_real  # Detects real overestimations so only large excesses can be penalized
            under_factor = torch.ones_like(reg_true_real)  # Starts with symmetric loss in mild events where overshooting should not be encouraged
            under_factor = torch.where(reg_true_real >= 5.0, torch.full_like(reg_true_real, 1.5), under_factor)  # Raises to 1.5x in moderate events to maintain peak-capture priority
            under_factor = torch.where(reg_true_real >= 20.0, torch.full_like(reg_true_real, 2.5), under_factor)  # Raises to 2.5x in high events where underestimation is already operationally serious
            under_factor = torch.where(reg_true_real >= 50.0, torch.full_like(reg_true_real, 3.5), under_factor)  # Raises to 3.5x in rare extremes where losing magnitude is the worst error
            safe_true_real = torch.clamp(reg_true_real, min=1e-6)  # Protects the relative-ratio division even though we are already in positive events
            over_ratio = torch.relu((reg_pred_real - reg_true_real) / safe_true_real)  # Measures how much prediction exceeds the real value in percentage terms
            excess_over_ratio = torch.relu(over_ratio - 0.25)  # Leaves a 25% margin for small overshoot and activates extra penalty only on clear excesses
            over_factor = 1.0 + torch.clamp(1.6 * excess_over_ratio, max=1.5)  # Makes 50-100% overshoot costly without making it worse than underestimating extremes
            significant_over_mask = over_mask & (reg_true_real > 5.0)  # Limits bias correction to moderate or larger events where severe overshoots appeared
            reg_factor = torch.ones_like(huber_values)  # Creates base tensor of multipliers to combine both rules without altering base Huber
            reg_factor = torch.where(under_mask, under_factor, reg_factor)  # Applies gradual penalty only when prediction falls below the real value
            reg_factor = torch.where(significant_over_mask, over_factor, reg_factor)  # Applies extra penalty only to excessive overestimations in significant events
            reg_loss = (huber_values * reg_factor * reg_weights).mean()  # Combines Huber, calibrated asymmetry, and sample weights in a single mean
        else:  # If there are no events in the batch, the regressor cannot be trained
            reg_loss = torch.zeros((), device=y_true.device, dtype=y_true.dtype)  # Uses scalar zero so it does not affect the total

        total_loss = (  # Combines classification and regression losses with configurable weights
            (self.cls_weight * cls_loss)  # Controls impact of the classification head
            + (self.reg_weight * reg_loss)  # Controls impact of the regression head
        )

        self.last_cls_loss = cls_loss.detach()  # Stores classification loss for external logging without gradient
        self.last_reg_loss = reg_loss.detach()  # Stores regression loss for external logging without gradient
        return total_loss  # Returns total loss for backpropagation
