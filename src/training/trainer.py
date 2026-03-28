"""Training utilities for stormflow models with early stopping and diagnostics."""

from __future__ import annotations  # Permite anotaciones modernas de tipos sin problemas de version

from copy import deepcopy  # Permite guardar y restaurar los mejores pesos del modelo
from typing import Any, Dict, List, Tuple  # Define tipos explicitos para historia y configuracion

import numpy as np  # Aporta conversion a arreglos para prediccion y reportes
import torch  # Provee tensores y operaciones de entrenamiento en CPU/GPU
from torch import nn  # Incluye tipos de modulos y funciones de perdida
from torch.optim import AdamW  # Optimizer recomendado en la propuesta
from torch.optim.lr_scheduler import ReduceLROnPlateau  # Scheduler recomendado para ajustar LR por validacion
from torch.utils.data import DataLoader  # Tipo de entrada para loaders de train/val/test


def _resolve_device(config: Dict[str, Any]) -> torch.device:
    """Resolve device from config with CUDA fallback when available."""
    if "device" in config:  # Revisa si el usuario fijo dispositivo manualmente en la configuracion
        return torch.device(str(config["device"]))  # Respeta dispositivo explicito si fue proporcionado
    if torch.cuda.is_available():  # Verifica disponibilidad de GPU en Colab para acelerar entrenamiento
        return torch.device("cuda")  # Usa GPU por defecto cuando existe soporte CUDA
    return torch.device("cpu")  # Fallback seguro a CPU para entornos sin acelerador


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Train model with AdamW, ReduceLROnPlateau, grad clipping, and early stopping."""
    device = _resolve_device(config)  # Determina dispositivo de entrenamiento segun config o disponibilidad
    model = model.to(device)  # Mueve modelo al dispositivo objetivo para computo consistente

    learning_rate = float(config.get("learning_rate", 1e-3))  # Usa LR recomendado si no viene override
    weight_decay = float(config.get("weight_decay", 1e-4))  # Usa weight decay recomendado si no viene override
    max_epochs = int(config.get("max_epochs", 80))  # Limita epocas segun propuesta para evitar sobreentrenamiento
    grad_clip_max_norm = float(config.get("grad_clip_max_norm", 1.0))  # Define clipping de gradiente recomendado
    early_stopping_patience = int(config.get("early_stopping_patience", 6))  # Define paciencia de early stopping

    optimizer = AdamW(  # Inicializa optimizer AdamW con hiperparametros de propuesta
        model.parameters(),  # Optimiza todos los parametros entrenables del modelo
        lr=learning_rate,  # Configura tasa de aprendizaje inicial
        weight_decay=weight_decay,  # Configura regularizacion L2 desacoplada
    )
    scheduler = ReduceLROnPlateau(  # Inicializa scheduler que reduce LR cuando se estanca val_loss
        optimizer=optimizer,  # Conecta scheduler al optimizer de entrenamiento
        mode="min",  # Minimiza metrica de validacion (val_loss)
        factor=0.5,  # Reduce LR a la mitad cuando no mejora
        patience=3,  # Espera 3 epocas sin mejora antes de reducir
    )

    history: Dict[str, Any] = {  # Prepara contenedor de historia para monitoreo y analisis posterior
        "train_loss": [],  # Guarda perdida media de entrenamiento por epoca
        "val_loss": [],  # Guarda perdida media de validacion por epoca
        "best_epoch": -1,  # Guarda epoca de mejor validacion observada
        "best_val_loss": float("inf"),  # Guarda mejor valor de validacion para early stopping
    }

    best_state_dict = deepcopy(model.state_dict())  # Toma snapshot inicial para poder restaurar mejores pesos luego
    epochs_without_improvement = 0  # Contador de epocas consecutivas sin mejora en validacion

    for epoch_idx in range(max_epochs):  # Recorre ciclo principal de entrenamiento por epocas
        model.train()  # Activa modo entrenamiento (BatchNorm/Dropout en modo train)
        train_loss_sum = 0.0  # Acumula perdida total de train para promediar al final de la epoca
        train_batches = 0  # Cuenta batches procesados para calcular media correctamente

        for x_batch, y_batch, w_batch in train_loader:  # Itera DataLoader de train con tres tensores (X, y, w)
            x_batch = x_batch.to(device)  # Mueve features de batch al dispositivo de entrenamiento
            y_batch = y_batch.to(device)  # Mueve target del batch al dispositivo de entrenamiento
            w_batch = w_batch.to(device)  # Mueve pesos por muestra al dispositivo de entrenamiento

            optimizer.zero_grad(set_to_none=True)  # Limpia gradientes previos de forma eficiente en memoria
            y_pred = model(x_batch)  # Ejecuta forward del modelo para obtener predicciones del batch
            loss = criterion(y_pred, y_batch, w_batch)  # Calcula loss compuesta usando prediccion, target y sample weights
            loss.backward()  # Ejecuta backpropagation para calcular gradientes
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)  # Aplica clipping para estabilizar entrenamiento
            optimizer.step()  # Actualiza parametros del modelo con el optimizer

            train_loss_sum += float(loss.detach().item())  # Suma perdida escalar del batch para promedio de epoca
            train_batches += 1  # Incrementa contador de batches procesados

        train_loss_epoch = train_loss_sum / max(train_batches, 1)  # Calcula perdida promedio de train evitando division por cero

        model.eval()  # Activa modo evaluacion (BatchNorm/Dropout en modo eval)
        val_loss_sum = 0.0  # Acumula perdida total de validacion para promedio por epoca
        val_batches = 0  # Cuenta batches de validacion para media robusta
        with torch.no_grad():  # Desactiva gradientes para reducir memoria y acelerar validacion
            for x_batch, y_batch, w_batch in val_loader:  # Itera DataLoader de validacion con tres tensores
                x_batch = x_batch.to(device)  # Mueve features de validacion al dispositivo
                y_batch = y_batch.to(device)  # Mueve target de validacion al dispositivo
                w_batch = w_batch.to(device)  # Mueve sample weights de validacion al dispositivo

                y_pred = model(x_batch)  # Ejecuta forward en validacion sin actualizar pesos
                loss = criterion(y_pred, y_batch, w_batch)  # Calcula perdida de validacion con la misma funcion objetivo
                val_loss_sum += float(loss.detach().item())  # Acumula perdida batch para promedio de epoca
                val_batches += 1  # Incrementa contador de batches de validacion

        val_loss_epoch = val_loss_sum / max(val_batches, 1)  # Calcula perdida promedio de validacion evitando division por cero

        scheduler.step(val_loss_epoch)  # Actualiza scheduler con metrica de validacion para adaptar LR
        current_lr = float(optimizer.param_groups[0]["lr"])  # Obtiene LR actual para imprimir diagnostico por epoca

        history["train_loss"].append(train_loss_epoch)  # Guarda train_loss en historia para analisis posterior
        history["val_loss"].append(val_loss_epoch)  # Guarda val_loss en historia para analisis posterior

        improved = val_loss_epoch < history["best_val_loss"]  # Verifica si esta epoca mejora el mejor valor de validacion
        if improved:  # Actualiza tracking del mejor modelo cuando hay mejora real
            history["best_val_loss"] = val_loss_epoch  # Registra nuevo mejor valor de validacion
            history["best_epoch"] = epoch_idx + 1  # Guarda epoca (1-based) donde ocurrio la mejora
            best_state_dict = deepcopy(model.state_dict())  # Guarda copia de pesos del mejor modelo actual
            epochs_without_improvement = 0  # Reinicia contador de paciencia al mejorar
        else:  # Maneja caso en que no hubo mejora de validacion
            epochs_without_improvement += 1  # Incrementa contador para criterio de early stopping

        star_marker = " \u2605" if improved else ""  # Marca visual solicitada cuando mejora val_loss
        print(  # Imprime progreso de cada epoca con metrica de train, val y LR
            f"[train] Epoch {epoch_idx + 1:03d}/{max_epochs} | "
            f"train_loss={train_loss_epoch:.6f} | val_loss={val_loss_epoch:.6f} | "
            f"lr={current_lr:.6e}{star_marker}"
        )

        if epochs_without_improvement >= early_stopping_patience:  # Verifica condicion de parada temprana por paciencia agotada
            print(f"[train] Early stopping activado en epoch {epoch_idx + 1}")  # Informa activacion de early stopping en consola
            break  # Detiene entrenamiento para evitar sobreajuste y ahorrar tiempo de computo

    model.load_state_dict(best_state_dict)  # Restaura mejores pesos encontrados durante entrenamiento
    print(  # Resume resultado final tras restaurar el mejor estado
        f"[train] Mejor epoch: {history['best_epoch']} | "
        f"best_val_loss={history['best_val_loss']:.6f}"
    )

    return history  # Devuelve historia completa para graficar curvas y auditar entrenamiento


def predict(model: nn.Module, data_loader: DataLoader, device: torch.device | str) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference over a loader and return numpy arrays (y_pred, y_real)."""
    resolved_device = torch.device(device)  # Normaliza dispositivo recibido para uso consistente en PyTorch
    model = model.to(resolved_device)  # Mueve modelo al dispositivo de inferencia
    model.eval()  # Activa modo evaluacion para desactivar dropout y usar BatchNorm en inferencia

    predictions: List[np.ndarray] = []  # Acumula predicciones batch a batch para concatenar al final
    targets: List[np.ndarray] = []  # Acumula targets reales batch a batch para retorno final

    with torch.no_grad():  # Desactiva gradientes para inferencia eficiente y menor uso de memoria
        for x_batch, y_batch, _w_batch in data_loader:  # Consume loader con tres tensores y descarta w en prediccion
            x_batch = x_batch.to(resolved_device)  # Mueve features al dispositivo de inferencia
            y_pred = model(x_batch)  # Ejecuta forward del modelo para obtener predicciones del batch

            predictions.append(y_pred.detach().cpu().numpy().reshape(-1))  # Convierte prediccion a numpy 1D y acumula
            targets.append(y_batch.detach().cpu().numpy().reshape(-1))  # Convierte target a numpy 1D y acumula

    if predictions:  # Verifica que haya al menos un batch antes de concatenar
        y_pred_array = np.concatenate(predictions, axis=0)  # Concatena todas las predicciones en un vector continuo
        y_real_array = np.concatenate(targets, axis=0)  # Concatena todos los targets en un vector continuo
    else:  # Maneja caso de loader vacio sin muestras disponibles
        y_pred_array = np.empty((0,), dtype=np.float32)  # Retorna arreglo vacio de predicciones si no hubo datos
        y_real_array = np.empty((0,), dtype=np.float32)  # Retorna arreglo vacio de targets si no hubo datos

    return y_pred_array, y_real_array  # Devuelve pares numpy para metrica y analisis posterior

