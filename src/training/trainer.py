"""Training utilities for stormflow models with early stopping and diagnostics."""

from __future__ import annotations  # Allows modern type annotations without version issues

from copy import deepcopy  # Lets us save and restore the best model weights
from typing import Any, Dict, List, Tuple  # Defines explicit types for history, configuration, and metadata

import numpy as np  # Provides conversion to arrays for prediction and reporting
import torch  # Provides tensors and training operations on CPU/GPU
from torch import nn  # Includes module types and loss functions
from torch.optim import AdamW  # Optimizer recommended in the proposal
from torch.optim.lr_scheduler import ReduceLROnPlateau  # Scheduler recommended to adjust LR by validation
from torch.utils.data import DataLoader  # Input type for train/val/test loaders
from src.models.tcn import TwoStageTCN  # Imports two-stage class to detect special behavior


def _resolve_device(config: Dict[str, Any]) -> torch.device:
    """Resolve device from config with CUDA fallback when available."""
    if "device" in config:  # Checks whether the user manually fixed a device in the configuration
        return torch.device(str(config["device"]))  # Respects explicit device if it was provided
    if torch.cuda.is_available():  # Checks GPU availability in Colab to speed up training
        return torch.device("cuda")  # Uses GPU by default when CUDA support exists
    return torch.device("cpu")  # Safe fallback to CPU for environments without accelerator


def _unpack_batch(
    batch: Tuple[torch.Tensor, ...],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Move a batch to device and normalize the tuple shape."""
    if len(batch) == 4:  # Supports dataset that includes auxiliary event label for metadata
        x_batch, y_batch, w_batch, event_batch = batch  # Unpacks the full tuple from the DataLoader
    elif len(batch) == 3:  # Preserves compatibility with datasets without explicit auxiliary label
        x_batch, y_batch, w_batch = batch  # Unpacks classic triplet (X, y, w)
        event_batch = (y_batch > 0).to(dtype=y_batch.dtype)  # Builds a minimal mask to preserve consistent metadata
    else:  # Detects unexpected formats before training on a malformed batch
        raise ValueError("Expected batches with 3 or 4 tensors")  # Raises a clear error to debug inconsistent DataLoaders

    x_batch = x_batch.to(device)  # Moves batch features to training device
    y_batch = y_batch.to(device)  # Moves batch target to training device
    w_batch = w_batch.to(device)  # Moves sample weights to training device
    event_batch = event_batch.to(device)  # Moves auxiliary label to device for optional metadata
    return x_batch, y_batch, w_batch, event_batch  # Returns homogenized batch ready for model and metrics


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Train model with AdamW, cosine restarts, grad clipping, and safer early stopping."""
    device = _resolve_device(config)  # Determines training device from config or availability
    model = model.to(device)  # Moves model to target device for consistent computation
    is_two_stage = isinstance(model, TwoStageTCN)  # Detects whether the model uses dict output and two-stage training

    learning_rate = float(config.get("learning_rate", 5e-4))  # Reduces default LR to stabilize startup with new differently scaled features
    weight_decay = float(config.get("weight_decay", 1e-4))  # Uses recommended weight decay unless overridden
    max_epochs = int(config.get("max_epochs", 80))  # Limits epochs according to the proposal to avoid overtraining
    min_epochs = min(int(config.get("min_epochs", 20)), max_epochs)  # Forces a minimum learning phase before allowing early stopping
    grad_clip_max_norm = float(config.get("grad_clip_max_norm", 1.0))  # Defines recommended gradient clipping
    early_stopping_patience = int(config.get("early_stopping_patience", 8))  # Increases patience so noisy validation does not stop too early
    early_stopping_min_delta = float(config.get("early_stopping_min_delta", 5e-5))  # Reduces min_delta to accept small but real improvements in rare tasks

    optimizer = AdamW(  # Initializes AdamW optimizer with proposal hyperparameters
        model.parameters(),  # Optimizes all trainable model parameters
        lr=learning_rate,  # Configures initial learning rate
        weight_decay=weight_decay,  # Configures decoupled L2 regularization
    )
    scheduler = ReduceLROnPlateau(  # Initializes scheduler that reduces LR when val_loss stagnates
        optimizer=optimizer,  # Connects scheduler to the training optimizer
        mode="min",  # Minimizes validation metric (val_loss)
        factor=0.5,  # Cuts LR in half when there is no improvement
        patience=4,  # Waits 4 epochs without improvement before reducing LR
    )

    history: Dict[str, Any] = {  # Prepares history container for monitoring and later analysis
        "train_loss": [],  # Stores mean training loss per epoch
        "val_loss": [],  # Stores mean validation loss per epoch
        "best_epoch": -1,  # Stores epoch of the best validation observed
        "best_val_loss": float("inf"),  # Stores best validation value for early stopping
    }
    if is_two_stage:  # Adds extra history only when the model is two-stage
        history["train_cls_loss"] = []  # Stores mean classification loss per epoch
        history["train_reg_loss"] = []  # Stores mean regression loss per epoch
        history["val_cls_loss"] = []  # Stores validation classification loss per epoch
        history["val_reg_loss"] = []  # Stores validation regression loss per epoch

    best_state_dict = deepcopy(model.state_dict())  # Takes initial snapshot so the best weights can be restored later
    epochs_without_improvement = 0  # Counter of consecutive epochs without validation improvement

    train_diag = getattr(train_loader, "stormflow_diagnostics", {})  # Recovers diagnostics attached by the pipeline if they exist
    if train_diag:  # Prints imbalance diagnostic before training for traceability in Colab
        print(  # Summarizes natural event rate and BCE weight suggested by the train loader
            f"[train] Event rate(train)={train_diag.get('event_rate', float('nan')):.4f} | "
            f"event_pos_weight={train_diag.get('event_pos_weight', float('nan')):.4f} | "
            f"min_epochs={min_epochs}"
        )

    for epoch_idx in range(max_epochs):  # Iterates main training loop over epochs
        model.train()  # Activates training mode for dropout and any train-dependent layers
        train_loss_sum = 0.0  # Accumulates total train loss to average at the end of the epoch
        train_cls_loss_sum = 0.0  # Accumulates classification loss for logging when applicable
        train_reg_loss_sum = 0.0  # Accumulates regression loss for logging when applicable
        train_batches = 0  # Counts processed batches to compute mean correctly

        for batch in train_loader:  # Iterates train DataLoader with tensors from the multitask pipeline
            x_batch, y_batch, w_batch, _event_batch = _unpack_batch(batch=batch, device=device)  # Moves batch to device and normalizes its shape

            optimizer.zero_grad(set_to_none=True)  # Clears previous gradients efficiently in memory
            if is_two_stage:  # Uses dict output for two-stage model
                model_output = model(x_batch)  # Runs model forward to obtain classifier and regressor
                if not isinstance(model_output, dict):  # Verifies expected signature of the two-stage model
                    raise TypeError("Model output must be a dict for TwoStageTCN")  # Raises clear error if output signature is incorrect
                loss = criterion(model_output, y_batch, w_batch)  # Computes two-stage loss with output dict
                cls_loss_value = getattr(criterion, "last_cls_loss", None)  # Recovers classification loss stored by the loss
                reg_loss_value = getattr(criterion, "last_reg_loss", None)  # Recovers regression loss stored by the loss
                if cls_loss_value is None or reg_loss_value is None:  # Verifies that the loss exposed the expected components
                    raise RuntimeError("TwoStageLoss must expose last_cls_loss and last_reg_loss")  # Fails early if component logging is missing
            else:  # Preserves classic flow for direct regression models
                y_pred = model(x_batch)  # Runs model forward to obtain continuous batch prediction
                if not isinstance(y_pred, torch.Tensor):  # Verifies new expected model signature to avoid silent errors
                    raise TypeError("Model output must be a torch.Tensor")  # Raises clear error if any model returns an unsupported structure
                loss = criterion(y_pred, y_batch, w_batch)  # Computes composite loss using only prediction, target, and weights

            loss.backward()  # Runs backpropagation to compute gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)  # Applies clipping to stabilize training
            optimizer.step()  # Updates model parameters with the optimizer

            train_loss_sum += float(loss.detach().item())  # Adds scalar batch loss for epoch average
            if is_two_stage:  # Accumulates classification and regression losses when applicable
                train_cls_loss_sum += float(cls_loss_value.detach().item())  # Adds batch classification loss
                train_reg_loss_sum += float(reg_loss_value.detach().item())  # Adds batch regression loss
            train_batches += 1  # Increments processed batch counter

        train_loss_epoch = train_loss_sum / max(train_batches, 1)  # Computes average train loss while avoiding division by zero
        train_cls_loss_epoch = train_cls_loss_sum / max(train_batches, 1)  # Computes mean classification loss if applicable
        train_reg_loss_epoch = train_reg_loss_sum / max(train_batches, 1)  # Computes mean regression loss if applicable

        model.eval()  # Activates evaluation mode to measure generalization without dropout
        val_loss_sum = 0.0  # Accumulates total validation loss for per-epoch average
        val_cls_loss_sum = 0.0  # Accumulates validation classification loss if applicable
        val_reg_loss_sum = 0.0  # Accumulates validation regression loss if applicable
        val_batches = 0  # Counts validation batches for robust averaging
        with torch.no_grad():  # Disables gradients to reduce memory and speed up validation
            for batch in val_loader:  # Iterates validation DataLoader with the same tensor structure
                x_batch, y_batch, w_batch, _event_batch = _unpack_batch(batch=batch, device=device)  # Moves batch to device and normalizes the signature
                if is_two_stage:  # Uses dict output for two-stage model
                    model_output = model(x_batch)  # Runs forward in validation without updating weights
                    if not isinstance(model_output, dict):  # Verifies output signature in validation too to detect inconsistencies early
                        raise TypeError("Model output must be a dict for TwoStageTCN")  # Raises clear error if output signature is incorrect
                    loss = criterion(model_output, y_batch, w_batch)  # Computes validation loss with the same objective function
                    cls_loss_value = getattr(criterion, "last_cls_loss", None)  # Recovers classification loss from the criterion
                    reg_loss_value = getattr(criterion, "last_reg_loss", None)  # Recovers regression loss from the criterion
                    if cls_loss_value is None or reg_loss_value is None:  # Verifies that the loss exposed the expected components
                        raise RuntimeError("TwoStageLoss must expose last_cls_loss and last_reg_loss")  # Fails early if component logging is missing
                else:  # Preserves classic flow for direct regression models
                    y_pred = model(x_batch)  # Runs forward in validation without updating weights
                    if not isinstance(y_pred, torch.Tensor):  # Verifies output signature in validation too to detect inconsistencies early
                        raise TypeError("Model output must be a torch.Tensor")  # Raises clear error if output signature is incorrect
                    loss = criterion(y_pred, y_batch, w_batch)  # Computes validation loss with the same objective function

                val_loss_sum += float(loss.detach().item())  # Accumulates batch loss for epoch average
                if is_two_stage:  # Accumulates classification and regression losses when applicable
                    val_cls_loss_sum += float(cls_loss_value.detach().item())  # Adds batch classification loss
                    val_reg_loss_sum += float(reg_loss_value.detach().item())  # Adds batch regression loss
                val_batches += 1  # Increments validation batch counter

        val_loss_epoch = val_loss_sum / max(val_batches, 1)  # Computes average validation loss while avoiding division by zero
        val_cls_loss_epoch = val_cls_loss_sum / max(val_batches, 1)  # Computes mean validation classification loss if applicable
        val_reg_loss_epoch = val_reg_loss_sum / max(val_batches, 1)  # Computes mean validation regression loss if applicable

        scheduler.step(val_loss_epoch)  # Updates scheduler with validation metric to adapt LR
        current_lr = float(optimizer.param_groups[0]["lr"])  # Gets current LR to print per-epoch diagnostic

        history["train_loss"].append(train_loss_epoch)  # Stores train_loss in history for later analysis
        history["val_loss"].append(val_loss_epoch)  # Stores val_loss in history for later analysis
        if is_two_stage:  # Stores additional history for two-stage diagnostics
            history["train_cls_loss"].append(train_cls_loss_epoch)  # Records mean training classification loss
            history["train_reg_loss"].append(train_reg_loss_epoch)  # Records mean training regression loss
            history["val_cls_loss"].append(val_cls_loss_epoch)  # Records mean validation classification loss
            history["val_reg_loss"].append(val_reg_loss_epoch)  # Records mean validation regression loss

        improved = val_loss_epoch < (history["best_val_loss"] - early_stopping_min_delta)  # Requires a real minimum improvement to accept a new best epoch
        if improved:  # Updates tracking of the best model when there is meaningful improvement
            history["best_val_loss"] = val_loss_epoch  # Records new best validation value
            history["best_epoch"] = epoch_idx + 1  # Stores epoch (1-based) where improvement happened
            best_state_dict = deepcopy(model.state_dict())  # Saves copy of the current best model weights
            epochs_without_improvement = 0  # Resets patience counter when improving
        else:  # Handles case where there was no meaningful validation improvement
            epochs_without_improvement += 1  # Increments counter for early-stopping criterion

        star_marker = " *" if improved else ""  # ASCII visual marker when validation improves
        if is_two_stage:  # Prints extended progress when the model is two-stage
            print(  # Prints per-epoch progress with total/cls/reg and LR
                f"[train] Epoch {epoch_idx + 1:03d}/{max_epochs} | "
                f"train_total={train_loss_epoch:.6f} | train_cls={train_cls_loss_epoch:.6f} | train_reg={train_reg_loss_epoch:.6f} | "
                f"val_total={val_loss_epoch:.6f} | val_cls={val_cls_loss_epoch:.6f} | val_reg={val_reg_loss_epoch:.6f} | "
                f"lr={current_lr:.6e}{star_marker}"
            )
        else:  # Keeps original print format for direct models
            print(  # Prints per-epoch progress with train, val, and LR metrics
                f"[train] Epoch {epoch_idx + 1:03d}/{max_epochs} | "
                f"train_loss={train_loss_epoch:.6f} | val_loss={val_loss_epoch:.6f} | "
                f"lr={current_lr:.6e}{star_marker}"
            )

        if (epoch_idx + 1) < min_epochs:  # Prevents stopping before a minimum useful learning phase is completed
            continue  # Skips early-stopping evaluation until the minimum epoch count is reached

        if epochs_without_improvement >= early_stopping_patience:  # Checks early-stopping condition after min_epochs
            print(f"[train] Early stopping activated at epoch {epoch_idx + 1}")  # Reports early stopping activation in console
            break  # Stops training to avoid overfitting and save compute time

    model.load_state_dict(best_state_dict)  # Restores best weights found during training
    history["epochs_trained"] = len(history["train_loss"])  # Records how many epochs were actually run for later auditing
    print(  # Summarizes final result after restoring best state
        f"[train] Best epoch: {history['best_epoch']} | "
        f"best_val_loss={history['best_val_loss']:.6f} | "
        f"epochs_trained={history['epochs_trained']}"
    )

    return history  # Returns full history to plot curves and audit training


def predict(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device | str,
    return_metadata: bool = False,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Run inference over a loader and optionally return event metadata."""
    resolved_device = torch.device(device)  # Normalizes received device for consistent use in PyTorch
    model = model.to(resolved_device)  # Moves model to inference device
    model.eval()  # Activates evaluation mode to disable dropout and fix model behavior
    is_two_stage = isinstance(model, TwoStageTCN)  # Detects whether the model uses hard-switch prediction

    predictions: List[np.ndarray] = []  # Accumulates predictions batch by batch to concatenate at the end
    targets: List[np.ndarray] = []  # Accumulates real targets batch by batch for final return
    event_targets: List[np.ndarray] = []  # Accumulates real event labels if the user requests metadata
    event_probabilities: List[np.ndarray] = []  # Accumulates predicted event probabilities for later diagnostics

    with torch.no_grad():  # Disables gradients for efficient inference and lower memory use
        for batch in data_loader:  # Consumes loader with the multitask structure defined by the pipeline
            x_batch, y_batch, _w_batch, event_batch = _unpack_batch(batch=batch, device=resolved_device)  # Reuses common logic to move tensors to device
            if is_two_stage:  # Uses hard-switch prediction if the model is two-stage
                y_pred = model.predict(x_batch, threshold=0.3)  # Runs hard switch with low threshold to avoid false negatives
                if not isinstance(y_pred, torch.Tensor):  # Verifies expected output signature for robust inference
                    raise TypeError("Model predict must return a torch.Tensor")  # Raises clear error if any model returns an unsupported structure

                predictions.append(y_pred.detach().cpu().numpy().reshape(-1))  # Converts prediction to 1D numpy and accumulates it
                targets.append(y_batch.detach().cpu().numpy().reshape(-1))  # Converts target to 1D numpy and accumulates it

                if return_metadata:  # Only accumulates extra metadata when the caller explicitly needs it
                    model_output = model(x_batch)  # Runs an additional forward pass to recover event probability
                    if not isinstance(model_output, dict):  # Verifies expected output signature for two-stage metadata
                        raise TypeError("Model output must be a dict for TwoStageTCN")  # Raises clear error if output signature is incorrect
                    cls_prob = model_output.get("cls_prob")  # Extracts event probability from dict output
                    if cls_prob is None:  # Verifies that the expected key exists
                        raise KeyError("model_output must contain 'cls_prob'")  # Fails clearly if event probability is missing
                    event_targets.append(event_batch.detach().cpu().numpy().reshape(-1))  # Stores real event label aligned with each target
                    event_probabilities.append(cls_prob.detach().cpu().numpy().reshape(-1))  # Stores probability predicted by the classifier
            else:  # Preserves original flow for direct regression models
                y_pred = model(x_batch)  # Runs model forward to obtain batch predictions
                if not isinstance(y_pred, torch.Tensor):  # Verifies expected output signature for robust inference
                    raise TypeError("Model output must be a torch.Tensor")  # Raises clear error if any model returns an unsupported structure

                predictions.append(y_pred.detach().cpu().numpy().reshape(-1))  # Converts prediction to 1D numpy and accumulates it
                targets.append(y_batch.detach().cpu().numpy().reshape(-1))  # Converts target to 1D numpy and accumulates it

                if return_metadata:  # Only accumulates extra metadata when the caller explicitly needs it
                    event_targets.append(event_batch.detach().cpu().numpy().reshape(-1))  # Stores real event label aligned with each target
                    event_probabilities.append(np.zeros_like(y_pred.detach().cpu().numpy().reshape(-1), dtype=np.float32))  # Preserves compatibility by returning zero probability when no event head exists

    if predictions:  # Checks that there is at least one batch before concatenating
        y_pred_array = np.concatenate(predictions, axis=0)  # Concatenates all predictions into one continuous vector
        y_real_array = np.concatenate(targets, axis=0)  # Concatenates all targets into one continuous vector
    else:  # Handles empty loader case with no available samples
        y_pred_array = np.empty((0,), dtype=np.float32)  # Returns empty prediction array if there was no data
        y_real_array = np.empty((0,), dtype=np.float32)  # Returns empty target array if there was no data

    if not return_metadata:  # Preserves simple signature when the user does not need extra information
        return y_pred_array, y_real_array  # Returns numpy pairs for metrics and later analysis

    metadata: Dict[str, np.ndarray] = {  # Prepares structured container for sample-aligned metadata
        "event_targets": np.concatenate(event_targets, axis=0) if event_targets else np.empty((0,), dtype=np.float32),  # Returns real event mask if it existed in the loader
        "event_probabilities": np.concatenate(event_probabilities, axis=0) if event_probabilities else np.empty((0,), dtype=np.float32),  # Returns zero vector to preserve compatibility with existing consumers
    }
    return y_pred_array, y_real_array, metadata  # Returns predictions, targets, and auxiliary metadata for advanced evaluation
