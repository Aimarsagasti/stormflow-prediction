"""Sequence and DataLoader utilities for MSD stormflow modeling."""

from __future__ import annotations  # Permite anotaciones de tipos modernas con buena compatibilidad

from typing import Dict, List, Optional, Tuple  # Define tipos claros para datasets, sampler y retornos

import numpy as np  # Aporta operaciones vectorizadas para construir ventanas eficientemente
import pandas as pd  # Provee acceso tabular para extraer arrays de entrada
import torch  # Base tensorial para construir datasets y dataloaders en PyTorch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler  # Utilidades de carga de datos y muestreo estratificado


class StormflowSequenceDataset(Dataset):
    """PyTorch dataset that returns (X, y, sample_weight)."""

    def __init__(self, x_array: np.ndarray, y_array: np.ndarray, weight_array: np.ndarray) -> None:
        self.x_tensor = torch.tensor(x_array, dtype=torch.float32)  # Convierte ventanas de entrada a tensor float32 para entrenamiento GPU
        self.y_tensor = torch.tensor(y_array, dtype=torch.float32).unsqueeze(-1)  # Convierte target a tensor columna para compatibilidad con modelos de regresion
        self.weight_tensor = torch.tensor(weight_array, dtype=torch.float32).unsqueeze(-1)  # Convierte peso por muestra a tensor columna para perdida ponderada

    def __len__(self) -> int:
        return int(self.x_tensor.shape[0])  # Devuelve numero total de secuencias disponibles en el dataset

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x_tensor[index], self.y_tensor[index], self.weight_tensor[index]  # Retorna tripleta solicitada por el usuario (X, y, sample_weight)


def _build_window_arrays(
    df_split: pd.DataFrame,
    feature_columns: List[str],
    target_col: str,
    aux_col: str,
    seq_length: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build sliding-window arrays and aligned target/event vectors."""
    feature_matrix = df_split[feature_columns].to_numpy(dtype=np.float32)  # Extrae matriz de features en float32 para reducir memoria
    target_array = df_split[target_col].to_numpy(dtype=np.float32)  # Extrae target en arreglo continuo para indexacion rapida
    aux_array = df_split[aux_col].to_numpy(dtype=bool)  # Extrae indicador auxiliar de evento para estratificacion

    total_rows = len(df_split)  # Obtiene tamano del split para calcular cuantas ventanas se pueden formar
    max_end_index = total_rows - horizon - 1  # Define ultimo indice t valido que permite acceder a t+horizon sin salir del arreglo
    if max_end_index < seq_length - 1:  # Detecta caso con muy pocos datos para construir una sola ventana
        empty_x = np.empty((0, seq_length, len(feature_columns)), dtype=np.float32)  # Crea tensor vacio con shape consistente para no romper pipeline
        empty_y = np.empty((0,), dtype=np.float32)  # Crea vector vacio de targets cuando no hay muestras
        empty_aux = np.empty((0,), dtype=bool)  # Crea vector vacio de aux para mantener contrato de retorno
        return empty_x, empty_y, empty_aux  # Retorna estructuras vacias que el llamador sabra manejar

    windows_x: List[np.ndarray] = []  # Acumula secuencias de entrada por cada indice t valido
    windows_y: List[float] = []  # Acumula target escalar asociado al horizonte futuro
    windows_aux: List[bool] = []  # Acumula bandera de evento de la muestra objetivo

    for end_index in range(seq_length - 1, max_end_index + 1):  # Recorre todos los indices t que permiten ventana y horizonte completos
        start_index = end_index - seq_length + 1  # Calcula inicio de la ventana causal [t-seq_length+1, ..., t]
        target_index = end_index + horizon  # Calcula indice objetivo en t+horizon segun requerimiento del usuario
        windows_x.append(feature_matrix[start_index : end_index + 1])  # Guarda bloque de features de longitud seq_length
        windows_y.append(float(target_array[target_index]))  # Guarda valor escalar objetivo en el horizonte
        windows_aux.append(bool(aux_array[target_index]))  # Guarda indicador auxiliar alineado con el target

    x_array = np.stack(windows_x).astype(np.float32)  # Convierte lista de ventanas a arreglo 3D [N, L, F]
    y_array = np.asarray(windows_y, dtype=np.float32)  # Convierte targets a arreglo 1D para dataset
    aux_target_array = np.asarray(windows_aux, dtype=bool)  # Convierte flags de evento a arreglo booleano

    return x_array, y_array, aux_target_array  # Devuelve tensores base para construir datasets y sampler


def _compute_sample_weights(y_array: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """Compute sample weights based on target magnitude buckets."""
    if y_array.size == 0:  # Maneja splits sin muestras para evitar errores de percentiles
        return np.empty((0,), dtype=np.float32), {"p95": float("nan"), "p99": float("nan"), "p999": float("nan")}  # Retorna salida vacia y percentiles no definidos

    p95 = float(np.quantile(y_array, 0.95))  # Calcula P95 sobre targets del split para definir bucket alto
    p99 = float(np.quantile(y_array, 0.99))  # Calcula P99 para bucket muy alto
    p999 = float(np.quantile(y_array, 0.999))  # Calcula P99.9 para bucket extremo

    sample_weights = np.ones_like(y_array, dtype=np.float32)  # Inicializa pesos base para valores de baja magnitud
    sample_weights[(y_array >= p95) & (y_array < p99)] = 4.0  # Aumenta peso para cola alta segun propuesta
    sample_weights[(y_array >= p99) & (y_array < p999)] = 10.0  # Aumenta mas el peso para cola muy alta
    sample_weights[y_array >= p999] = 20.0  # Da maxima prioridad a extremos de magnitud critica

    return sample_weights, {"p95": p95, "p99": p99, "p999": p999}  # Devuelve vector de pesos y umbrales usados


def _build_train_sampler(
    y_array: np.ndarray,
    aux_array: np.ndarray,
) -> Tuple[Optional[WeightedRandomSampler], Dict[str, int], Dict[str, float], Dict[str, float]]:
    """Create weighted sampler with 40/30/20/10 stratified proportions."""
    if y_array.size == 0:  # Evita construir sampler si no hay ventanas de entrenamiento
        empty_counts = {"event": 0, "p95": 0, "p99": 0, "base": 0}  # Define conteos nulos por categoria
        empty_ratios = {"event": 0.0, "p95": 0.0, "p99": 0.0, "base": 0.0}  # Define ratios nulos para diagnostico
        return None, empty_counts, empty_ratios, {"p95": float("nan"), "p99": float("nan")}  # Retorna sin sampler con metadata vacia

    p95 = float(np.quantile(y_array, 0.95))  # Calcula umbral P95 para estrato medio-alto
    p99 = float(np.quantile(y_array, 0.99))  # Calcula umbral P99 para estrato extremo

    p99_mask = y_array >= p99  # Marca muestras en el estrato mas extremo por magnitud
    p95_mask = (y_array >= p95) & (~p99_mask)  # Marca muestras altas no incluidas en P99 por prioridad
    event_mask = aux_array & (~p95_mask) & (~p99_mask)  # Marca muestras de evento no capturadas por buckets de magnitud
    base_mask = ~(p99_mask | p95_mask | event_mask)  # Marca resto de muestras como base/no evento

    category_masks = {  # Agrupa mascaras de categorias para iteracion uniforme en el calculo de pesos
        "event": event_mask,  # Categoria de evento para 40% del muestreo
        "p95": p95_mask,  # Categoria de targets >= P95 para 30% del muestreo
        "p99": p99_mask,  # Categoria de targets >= P99 para 20% del muestreo
        "base": base_mask,  # Categoria base para 10% del muestreo
    }
    desired_ratios = {  # Define distribucion objetivo del sampler segun requerimiento del usuario
        "event": 0.40,  # Reserva 40% de probabilidad para ventanas de evento
        "p95": 0.30,  # Reserva 30% para cola >= P95 (sin doble conteo con P99)
        "p99": 0.20,  # Reserva 20% para cola >= P99
        "base": 0.10,  # Reserva 10% para baseflow/no evento
    }

    category_counts = {name: int(mask.sum()) for name, mask in category_masks.items()}  # Cuenta muestras disponibles por categoria
    active_categories = [name for name, count in category_counts.items() if count > 0]  # Identifica categorias con al menos una muestra disponible

    if not active_categories:  # Maneja caso patologico sin muestras categorizables
        return None, category_counts, {key: 0.0 for key in desired_ratios}, {"p95": p95, "p99": p99}  # Retorna sin sampler y con metadatos minimos

    total_active_ratio = sum(desired_ratios[name] for name in active_categories)  # Suma ratios de categorias activas para renormalizar si alguna falta
    adjusted_ratios = {name: 0.0 for name in desired_ratios}  # Inicializa diccionario de ratios ajustados para muestreo efectivo
    for name in active_categories:  # Reparte la masa de probabilidad solo entre categorias presentes
        adjusted_ratios[name] = desired_ratios[name] / total_active_ratio  # Renormaliza ratio deseado manteniendo proporciones relativas

    sample_probs = np.zeros_like(y_array, dtype=np.float64)  # Inicializa probabilidad por muestra para WeightedRandomSampler
    for name in active_categories:  # Asigna probabilidad a cada muestra segun su categoria
        mask = category_masks[name]  # Recupera mascara booleana de la categoria actual
        category_size = category_counts[name]  # Obtiene cuantas muestras hay en dicha categoria
        per_sample_prob = adjusted_ratios[name] / category_size  # Distribuye ratio de categoria uniformemente dentro del grupo
        sample_probs[mask] = per_sample_prob  # Escribe probabilidad de cada muestra de la categoria

    sample_probs = sample_probs / sample_probs.sum()  # Normaliza para garantizar suma total 1.0 y estabilidad numerica
    sampler = WeightedRandomSampler(  # Crea sampler estratificado para el DataLoader de entrenamiento
        weights=torch.tensor(sample_probs, dtype=torch.double),  # Pasa pesos en double como recomienda PyTorch para sampler
        num_samples=len(sample_probs),  # Mantiene mismo numero de muestras por epoca que el dataset de train
        replacement=True,  # Permite reemplazo para sostener proporciones aun con categorias pequenas
    )

    return sampler, category_counts, adjusted_ratios, {"p95": p95, "p99": p99}  # Devuelve sampler y metadatos para diagnosticos impresos


def _build_single_loader(
    df_split: pd.DataFrame,
    feature_columns: List[str],
    target_col: str,
    aux_col: str,
    seq_length: int,
    horizon: int,
    batch_size: int,
    use_stratified_sampler: bool,
) -> Tuple[DataLoader, Dict[str, object]]:
    """Create one DataLoader and diagnostics for a split."""
    x_array, y_array, aux_array = _build_window_arrays(  # Construye ventanas causales y targets alineados al horizonte
        df_split=df_split,
        feature_columns=feature_columns,
        target_col=target_col,
        aux_col=aux_col,
        seq_length=seq_length,
        horizon=horizon,
    )
    sample_weight_array, magnitude_quantiles = _compute_sample_weights(y_array)  # Calcula pesos por bucket de magnitud para la perdida

    sampler: Optional[WeightedRandomSampler] = None  # Inicializa sampler opcional para splits sin estratificacion
    category_counts = {"event": 0, "p95": 0, "p99": 0, "base": 0}  # Define conteos por defecto para diagnostico uniforme
    sampler_ratios = {"event": 0.0, "p95": 0.0, "p99": 0.0, "base": 0.0}  # Define ratios por defecto para splits no estratificados
    sampler_thresholds = {"p95": float("nan"), "p99": float("nan")}  # Define umbrales por defecto para imprimir siempre misma estructura

    if use_stratified_sampler:  # Activa sampler estratificado solo para train segun requerimiento
        sampler, category_counts, sampler_ratios, sampler_thresholds = _build_train_sampler(y_array, aux_array)  # Construye sampler 40/30/20/10 con prioridad de categorias

    dataset = StormflowSequenceDataset(x_array=x_array, y_array=y_array, weight_array=sample_weight_array)  # Envuelve arreglos en dataset PyTorch tripleta (X, y, w)
    dataloader = DataLoader(  # Crea DataLoader para iteracion eficiente por lotes
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and not use_stratified_sampler),  # Usa shuffle solo cuando no hay sampler y corresponde evaluacion uniforme
        sampler=sampler,  # Inyecta sampler estratificado cuando aplique en train
        drop_last=False,  # Conserva ultimo batch incompleto para no perder muestras
    )

    diagnostics: Dict[str, object] = {  # Empaqueta estadisticas utiles para trazabilidad de pipeline
        "num_windows": int(len(dataset)),  # Guarda numero total de ventanas creadas para el split
        "x_shape": tuple(x_array.shape),  # Guarda shape de entrada [N, seq_length, n_features]
        "y_shape": tuple(y_array.shape),  # Guarda shape de target [N]
        "num_batches": int(len(dataloader)),  # Guarda cantidad de batches resultantes
        "category_counts": category_counts,  # Guarda conteo por categoria del sampler
        "sampler_ratios": sampler_ratios,  # Guarda ratios efectivos usados por el sampler
        "sampler_thresholds": sampler_thresholds,  # Guarda umbrales P95/P99 del sampler
        "magnitude_quantiles": magnitude_quantiles,  # Guarda umbrales P95/P99/P99.9 para sample_weight
    }

    return dataloader, diagnostics  # Devuelve DataLoader listo y metadata para diagnosticos


def create_dataloaders(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_columns: List[str],
    target_col: str,
    aux_col: str,
    seq_length: int = 72,
    horizon: int = 6,
    batch_size: int = 256,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders with stratified sampling on train."""
    train_loader, train_diag = _build_single_loader(  # Crea loader de train con sampler estratificado
        df_split=df_train,
        feature_columns=feature_columns,
        target_col=target_col,
        aux_col=aux_col,
        seq_length=seq_length,
        horizon=horizon,
        batch_size=batch_size,
        use_stratified_sampler=True,
    )
    val_loader, val_diag = _build_single_loader(  # Crea loader de validacion con muestreo uniforme
        df_split=df_val,
        feature_columns=feature_columns,
        target_col=target_col,
        aux_col=aux_col,
        seq_length=seq_length,
        horizon=horizon,
        batch_size=batch_size,
        use_stratified_sampler=False,
    )
    test_loader, test_diag = _build_single_loader(  # Crea loader de test con muestreo uniforme
        df_split=df_test,
        feature_columns=feature_columns,
        target_col=target_col,
        aux_col=aux_col,
        seq_length=seq_length,
        horizon=horizon,
        batch_size=batch_size,
        use_stratified_sampler=False,
    )

    print(f"[sequences] Train X shape: {train_diag['x_shape']} | y shape: {train_diag['y_shape']} | batches: {train_diag['num_batches']}")  # Reporta tamanos de train para validar ventana y batching
    print(f"[sequences] Val   X shape: {val_diag['x_shape']} | y shape: {val_diag['y_shape']} | batches: {val_diag['num_batches']}")  # Reporta tamanos de validacion
    print(f"[sequences] Test  X shape: {test_diag['x_shape']} | y shape: {test_diag['y_shape']} | batches: {test_diag['num_batches']}")  # Reporta tamanos de test

    print(f"[sequences] Sampler counts(train): {train_diag['category_counts']}")  # Imprime conteos por categoria en train para revisar cobertura
    print(f"[sequences] Sampler ratios(train): {train_diag['sampler_ratios']}")  # Imprime ratios efectivos del sampler 40/30/20/10
    print(f"[sequences] Sampler thresholds(train): {train_diag['sampler_thresholds']}")  # Imprime P95/P99 usados por el sampler estratificado
    print(f"[sequences] Magnitude quantiles(train): {train_diag['magnitude_quantiles']}")  # Imprime P95/P99/P99.9 usados para sample_weight

    return train_loader, val_loader, test_loader  # Devuelve loaders listos para entrenamiento y evaluacion
