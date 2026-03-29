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


def _compute_quantile_thresholds(y_array: np.ndarray) -> Dict[str, float]:
    """Compute magnitude thresholds from the training target distribution."""
    if y_array.size == 0:  # Evita calcular cuantiles cuando no hay muestras disponibles
        return {"p95": float("nan"), "p99": float("nan"), "p999": float("nan")}  # Retorna umbrales no definidos si el split esta vacio
    return {  # Devuelve cuantiles clave para pesos y diagnosticos de severidad
        "p95": float(np.quantile(y_array, 0.95)),  # Define umbral alto de la distribucion
        "p99": float(np.quantile(y_array, 0.99)),  # Define umbral muy alto de la distribucion
        "p999": float(np.quantile(y_array, 0.999)),  # Define umbral extremo de la distribucion
    }


def _compute_sample_weights(y_array: np.ndarray, thresholds: Dict[str, float]) -> np.ndarray:
    """Compute sample weights based on fixed target magnitude thresholds."""
    if y_array.size == 0:  # Maneja splits sin muestras para evitar errores posteriores
        return np.empty((0,), dtype=np.float32)  # Retorna vector vacio cuando no hay targets

    p95 = thresholds["p95"]  # Recupera umbral P95 calculado sobre train para consistencia entre splits
    p99 = thresholds["p99"]  # Recupera umbral P99 calculado sobre train para consistencia entre splits
    p999 = thresholds["p999"]  # Recupera umbral P99.9 calculado sobre train para consistencia entre splits

    sample_weights = np.ones_like(y_array, dtype=np.float32)  # Inicializa pesos base para valores de baja magnitud
    sample_weights[(y_array >= p95) & (y_array < p99)] = 4.0  # Aumenta peso para cola alta segun propuesta
    sample_weights[(y_array >= p99) & (y_array < p999)] = 10.0  # Aumenta mas el peso para cola muy alta
    sample_weights[y_array >= p999] = 20.0  # Da maxima prioridad a extremos de magnitud critica
    return sample_weights  # Devuelve vector de pesos por muestra alineado al target


def _build_train_sampler(
    y_array: np.ndarray,
    aux_array: np.ndarray,
    thresholds: Dict[str, float],
    sampler_ratios: Dict[str, float],
) -> Tuple[Optional[WeightedRandomSampler], Dict[str, int], Dict[str, float]]:
    """Create weighted sampler with configurable stratified proportions."""
    if y_array.size == 0:  # Evita construir sampler si no hay ventanas de entrenamiento
        empty_counts = {"event": 0, "p95": 0, "p99": 0, "base": 0}  # Define conteos nulos por categoria
        empty_ratios = {"event": 0.0, "p95": 0.0, "p99": 0.0, "base": 0.0}  # Define ratios nulos para diagnostico
        return None, empty_counts, empty_ratios  # Retorna sin sampler cuando no hay muestras disponibles

    p95_mask = y_array >= thresholds["p95"]  # Marca muestras altas usando umbral fijo de train
    p99_mask = y_array >= thresholds["p99"]  # Marca muestras extremas usando umbral fijo de train
    high_only_mask = p95_mask & (~p99_mask)  # Separa cola alta no extrema para evitar doble conteo
    event_only_mask = aux_array & (~p95_mask) & (~p99_mask)  # Conserva eventos no capturados por buckets de magnitud
    base_mask = ~(p99_mask | high_only_mask | event_only_mask)  # Marca resto de muestras como base/no evento

    category_masks = {  # Agrupa mascaras de categorias para asignar probabilidades por muestra
        "event": event_only_mask,  # Categoria evento para mantener sensibilidad a ventanas de evento
        "p95": high_only_mask,  # Categoria de cola alta moderada
        "p99": p99_mask,  # Categoria de cola extrema
        "base": base_mask,  # Categoria base para mantener calibracion del regimen dominante
    }

    category_counts = {name: int(mask.sum()) for name, mask in category_masks.items()}  # Cuenta muestras disponibles por categoria
    active_categories = [name for name, count in category_counts.items() if count > 0]  # Detecta categorias que realmente tienen muestras
    if not active_categories:  # Maneja caso patologico sin categorias activas
        return None, category_counts, {key: 0.0 for key in sampler_ratios}  # Retorna sin sampler y con ratios nulos

    total_active_ratio = sum(sampler_ratios[name] for name in active_categories)  # Suma masa objetivo de categorias presentes para renormalizar
    adjusted_ratios = {name: 0.0 for name in sampler_ratios}  # Inicializa ratios efectivos con cero por defecto
    for name in active_categories:  # Recorre categorias presentes para asignar probabilidad efectiva
        adjusted_ratios[name] = sampler_ratios[name] / total_active_ratio  # Renormaliza ratios manteniendo proporciones relativas

    sample_probs = np.zeros_like(y_array, dtype=np.float64)  # Inicializa probabilidad por muestra para sampler ponderado
    for name in active_categories:  # Asigna probabilidad uniforme dentro de cada categoria activa
        mask = category_masks[name]  # Recupera mascara booleana de la categoria actual
        per_sample_prob = adjusted_ratios[name] / category_counts[name]  # Distribuye masa de categoria entre sus muestras
        sample_probs[mask] = per_sample_prob  # Escribe probabilidad por muestra de la categoria actual

    sample_probs = sample_probs / sample_probs.sum()  # Normaliza para garantizar suma 1 y estabilidad numerica
    sampler = WeightedRandomSampler(  # Crea sampler estratificado para train con reemplazo
        weights=torch.tensor(sample_probs, dtype=torch.double),  # Convierte probabilidades a tensor double compatible con PyTorch
        num_samples=len(sample_probs),  # Mantiene una epoca con tantas muestras como ventanas disponibles
        replacement=True,  # Permite sostener ratios incluso cuando una categoria es pequena
    )

    return sampler, category_counts, adjusted_ratios  # Devuelve sampler y metadata util para diagnosticos


def _build_single_loader(
    df_split: pd.DataFrame,
    feature_columns: List[str],
    target_col: str,
    aux_col: str,
    seq_length: int,
    horizon: int,
    batch_size: int,
    use_stratified_sampler: bool,
    thresholds: Dict[str, float],
    sampler_ratios: Dict[str, float],
) -> Tuple[DataLoader, Dict[str, object]]:
    """Create one DataLoader and diagnostics for a split."""
    x_array, y_array, aux_array = _build_window_arrays(  # Construye ventanas causales y targets alineados al horizonte
        df_split=df_split,  # Usa el split actual como fuente de secuencias
        feature_columns=feature_columns,  # Usa el orden de features definido por el pipeline
        target_col=target_col,  # Usa columna target ya normalizada del split
        aux_col=aux_col,  # Usa bandera auxiliar para sampler y diagnosticos
        seq_length=seq_length,  # Usa longitud de historia definida para el modelo
        horizon=horizon,  # Usa horizonte objetivo definido para la prediccion
    )
    sample_weight_array = _compute_sample_weights(y_array, thresholds=thresholds)  # Calcula pesos con umbrales fijos de train para coherencia entre splits

    sampler: Optional[WeightedRandomSampler] = None  # Inicializa sampler opcional para splits sin estratificacion
    category_counts = {"event": 0, "p95": 0, "p99": 0, "base": 0}  # Define conteos por defecto para diagnostico uniforme
    effective_ratios = {"event": 0.0, "p95": 0.0, "p99": 0.0, "base": 0.0}  # Define ratios por defecto para splits no estratificados
    shuffle_loader = False  # Mantiene orden estable por defecto, especialmente en validacion y test

    if use_stratified_sampler:  # Activa sampler estratificado solo para train segun diseno del pipeline
        sampler, category_counts, effective_ratios = _build_train_sampler(  # Construye sampler con ratios configurables y umbrales fijos de train
            y_array=y_array,  # Pasa targets del split train para categorizar ventanas
            aux_array=aux_array,  # Pasa bandera auxiliar de evento para categoria evento
            thresholds=thresholds,  # Usa umbrales calculados sobre train para definir categorias de magnitud
            sampler_ratios=sampler_ratios,  # Usa ratios configurables para balancear sensibilidad y calibracion
        )
        shuffle_loader = False  # Desactiva shuffle explicito cuando existe sampler dedicado en train

    dataset = StormflowSequenceDataset(x_array=x_array, y_array=y_array, weight_array=sample_weight_array)  # Envuelve arreglos en dataset PyTorch tripleta (X, y, w)
    dataloader = DataLoader(  # Crea DataLoader para iteracion eficiente por lotes
        dataset,  # Usa dataset de secuencias ya construido para el split actual
        batch_size=batch_size,  # Usa tamano de batch configurado para entrenamiento o evaluacion
        shuffle=shuffle_loader,  # Mantiene orden temporal estable cuando no hay sampler especifico
        sampler=sampler,  # Inyecta sampler estratificado cuando aplique en train
        drop_last=False,  # Conserva ultimo batch incompleto para no perder muestras
    )

    diagnostics: Dict[str, object] = {  # Empaqueta estadisticas utiles para trazabilidad de pipeline
        "num_windows": int(len(dataset)),  # Guarda numero total de ventanas creadas para el split
        "x_shape": tuple(x_array.shape),  # Guarda shape de entrada [N, seq_length, n_features]
        "y_shape": tuple(y_array.shape),  # Guarda shape de target [N]
        "num_batches": int(len(dataloader)),  # Guarda cantidad de batches resultantes
        "category_counts": category_counts,  # Guarda conteo por categoria del sampler
        "sampler_ratios": effective_ratios,  # Guarda ratios efectivos usados por el sampler
        "thresholds": thresholds,  # Guarda umbrales P95/P99/P99.9 usados en pesos y categorias
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
    sampler_ratios: Optional[Dict[str, float]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders with stratified sampling on train."""
    effective_sampler_ratios = sampler_ratios or {  # Define mezcla mas equilibrada para no perder calibracion del regimen base
        "event": 0.25,  # Mantiene sensibilidad a eventos sin monopolizar la epoca completa
        "p95": 0.25,  # Mantiene suficiente presencia de cola alta moderada
        "p99": 0.20,  # Sigue enfatizando cola extrema por su importancia operativa
        "base": 0.30,  # Devuelve mas cobertura al regimen dominante para reducir sesgo positivo en baseflow
    }
    ratio_sum = sum(effective_sampler_ratios.values())  # Calcula suma total para validar distribucion del sampler
    if not np.isclose(ratio_sum, 1.0):  # Verifica que la mezcla del sampler sume 1 para interpretar bien las proporciones
        raise ValueError("sampler_ratios must sum to 1.0")  # Lanza error claro cuando la mezcla no es valida

    train_thresholds = _compute_quantile_thresholds(df_train[target_col].to_numpy(dtype=np.float32))  # Calcula umbrales de magnitud una sola vez sobre train

    train_loader, train_diag = _build_single_loader(  # Crea loader de train con sampler estratificado
        df_split=df_train,  # Usa split train como fuente de secuencias estratificadas
        feature_columns=feature_columns,  # Pasa columnas de entrada en el orden correcto
        target_col=target_col,  # Pasa nombre del target ya normalizado
        aux_col=aux_col,  # Pasa bandera auxiliar de eventos
        seq_length=seq_length,  # Pasa longitud de historia del modelo
        horizon=horizon,  # Pasa horizonte de prediccion deseado
        batch_size=batch_size,  # Pasa tamano de batch configurado
        use_stratified_sampler=True,  # Activa sampler solo en train
        thresholds=train_thresholds,  # Usa cuantiles de train para pesos y categorias
        sampler_ratios=effective_sampler_ratios,  # Usa mezcla mas equilibrada y configurable
    )
    val_loader, val_diag = _build_single_loader(  # Crea loader de validacion con muestreo uniforme y orden estable
        df_split=df_val,  # Usa split de validacion en orden cronologico
        feature_columns=feature_columns,  # Pasa columnas de entrada
        target_col=target_col,  # Pasa nombre del target normalizado
        aux_col=aux_col,  # Pasa bandera auxiliar para diagnosticos
        seq_length=seq_length,  # Pasa longitud de secuencia del modelo
        horizon=horizon,  # Pasa horizonte de prediccion
        batch_size=batch_size,  # Pasa tamano de batch de evaluacion
        use_stratified_sampler=False,  # Desactiva estratificacion en validacion
        thresholds=train_thresholds,  # Usa mismos umbrales de train para coherencia de loss/metricas
        sampler_ratios=effective_sampler_ratios,  # Pasa estructura de ratios aunque no se use en validacion
    )
    test_loader, test_diag = _build_single_loader(  # Crea loader de test con muestreo uniforme y orden estable
        df_split=df_test,  # Usa split de test en orden cronologico
        feature_columns=feature_columns,  # Pasa columnas de entrada
        target_col=target_col,  # Pasa nombre del target normalizado
        aux_col=aux_col,  # Pasa bandera auxiliar para diagnosticos posteriores
        seq_length=seq_length,  # Pasa longitud de secuencia definida
        horizon=horizon,  # Pasa horizonte objetivo
        batch_size=batch_size,  # Pasa tamano de batch para inferencia/evaluacion
        use_stratified_sampler=False,  # Desactiva estratificacion en test
        thresholds=train_thresholds,  # Reusa umbrales de train para evaluar en la misma escala operativa
        sampler_ratios=effective_sampler_ratios,  # Pasa estructura de ratios aunque no se use en test
    )

    print(f"[sequences] Train X shape: {train_diag['x_shape']} | y shape: {train_diag['y_shape']} | batches: {train_diag['num_batches']}")  # Reporta tamanos de train para validar ventana y batching
    print(f"[sequences] Val   X shape: {val_diag['x_shape']} | y shape: {val_diag['y_shape']} | batches: {val_diag['num_batches']}")  # Reporta tamanos de validacion
    print(f"[sequences] Test  X shape: {test_diag['x_shape']} | y shape: {test_diag['y_shape']} | batches: {test_diag['num_batches']}")  # Reporta tamanos de test

    print(f"[sequences] Sampler counts(train): {train_diag['category_counts']}")  # Imprime conteos por categoria en train para revisar cobertura
    print(f"[sequences] Sampler ratios(train): {train_diag['sampler_ratios']}")  # Imprime ratios efectivos del sampler tras renormalizacion
    print(f"[sequences] Weight thresholds(train): {train_diag['thresholds']}")  # Imprime P95/P99/P99.9 usados por pesos y categorias

    return train_loader, val_loader, test_loader  # Devuelve loaders listos para entrenamiento y evaluacion
