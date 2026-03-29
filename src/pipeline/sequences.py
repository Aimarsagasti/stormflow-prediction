"""Sequence and DataLoader utilities for MSD stormflow modeling."""

from __future__ import annotations  # Permite anotaciones de tipos modernas con buena compatibilidad

from typing import Dict, List, Tuple  # Define tipos claros para datasets y retornos del pipeline

import numpy as np  # Aporta operaciones vectorizadas para construir ventanas eficientemente
import pandas as pd  # Provee acceso tabular para extraer arrays de entrada
import torch  # Base tensorial para construir datasets y dataloaders en PyTorch
from torch.utils.data import DataLoader, Dataset  # Utilidades de carga de datos para entrenamiento y evaluacion


class StormflowSequenceDataset(Dataset):
    """PyTorch dataset that returns (X, y, sample_weight, event_target)."""

    def __init__(
        self,
        x_array: np.ndarray,
        y_array: np.ndarray,
        weight_array: np.ndarray,
        event_array: np.ndarray,
    ) -> None:
        self.x_tensor = torch.tensor(x_array, dtype=torch.float32)  # Convierte ventanas de entrada a tensor float32 para entrenamiento GPU
        self.y_tensor = torch.tensor(y_array, dtype=torch.float32).unsqueeze(-1)  # Convierte target a tensor columna para compatibilidad con modelos de regresion
        self.weight_tensor = torch.tensor(weight_array, dtype=torch.float32).unsqueeze(-1)  # Convierte peso por muestra a tensor columna para perdida ponderada
        self.event_tensor = torch.tensor(event_array.astype(np.float32), dtype=torch.float32).unsqueeze(-1)  # Convierte la etiqueta de evento futuro a tensor columna para multitarea

    def __len__(self) -> int:
        return int(self.x_tensor.shape[0])  # Devuelve numero total de secuencias disponibles en el dataset

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (  # Retorna la cuadrupla completa requerida por entrenamiento multitarea
            self.x_tensor[index],  # Devuelve la secuencia de entrada ya tensorizada
            self.y_tensor[index],  # Devuelve el target escalar alineado al horizonte
            self.weight_tensor[index],  # Devuelve el peso de muestra para la loss compuesta
            self.event_tensor[index],  # Devuelve la etiqueta booleana de evento a predecir
        )


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
    event_array = df_split[aux_col].to_numpy(dtype=bool)  # Extrae indicador auxiliar de evento para supervision multitarea

    total_rows = len(df_split)  # Obtiene tamano del split para calcular cuantas ventanas se pueden formar
    max_end_index = total_rows - horizon - 1  # Define ultimo indice t valido que permite acceder a t+horizon sin salir del arreglo
    if max_end_index < seq_length - 1:  # Detecta caso con muy pocos datos para construir una sola ventana
        empty_x = np.empty((0, seq_length, len(feature_columns)), dtype=np.float32)  # Crea tensor vacio con shape consistente para no romper pipeline
        empty_y = np.empty((0,), dtype=np.float32)  # Crea vector vacio de targets cuando no hay muestras
        empty_event = np.empty((0,), dtype=bool)  # Crea vector vacio de eventos para mantener contrato de retorno
        return empty_x, empty_y, empty_event  # Retorna estructuras vacias que el llamador sabra manejar

    windows_x: List[np.ndarray] = []  # Acumula secuencias de entrada por cada indice t valido
    windows_y: List[float] = []  # Acumula target escalar asociado al horizonte futuro
    windows_event: List[bool] = []  # Acumula etiqueta de evento de la muestra objetivo

    for end_index in range(seq_length - 1, max_end_index + 1):  # Recorre todos los indices t que permiten ventana y horizonte completos
        start_index = end_index - seq_length + 1  # Calcula inicio de la ventana causal [t-seq_length+1, ..., t]
        target_index = end_index + horizon  # Calcula indice objetivo en t+horizon segun requerimiento del usuario
        windows_x.append(feature_matrix[start_index : end_index + 1])  # Guarda bloque de features de longitud seq_length
        windows_y.append(float(target_array[target_index]))  # Guarda valor escalar objetivo en el horizonte
        windows_event.append(bool(event_array[target_index]))  # Guarda la etiqueta de evento alineada exactamente al mismo objetivo

    x_array = np.stack(windows_x).astype(np.float32)  # Convierte lista de ventanas a arreglo 3D [N, L, F]
    y_array = np.asarray(windows_y, dtype=np.float32)  # Convierte targets a arreglo 1D para dataset
    event_target_array = np.asarray(windows_event, dtype=bool)  # Convierte flags de evento a arreglo booleano alineado con el target

    return x_array, y_array, event_target_array  # Devuelve tensores base para construir datasets y diagnosticos


def _compute_quantile_thresholds(y_array: np.ndarray) -> Dict[str, float]:
    """Compute magnitude thresholds from the training target distribution."""
    if y_array.size == 0:  # Evita calcular cuantiles cuando no hay muestras disponibles
        return {"p95": float("nan"), "p99": float("nan"), "p999": float("nan")}  # Retorna umbrales no definidos si el split esta vacio
    return {  # Devuelve cuantiles clave para pesos y diagnosticos de severidad
        "p95": float(np.quantile(y_array, 0.95)),  # Define umbral alto de la distribucion
        "p99": float(np.quantile(y_array, 0.99)),  # Define umbral muy alto de la distribucion
        "p999": float(np.quantile(y_array, 0.999)),  # Define umbral extremo de la distribucion
    }


def _compute_sample_weights(y_array: np.ndarray, event_array: np.ndarray, thresholds: Dict[str, float]) -> np.ndarray:
    """Compute sample weights using both event presence and target magnitude."""
    if y_array.size == 0:  # Maneja splits sin muestras para evitar errores posteriores
        return np.empty((0,), dtype=np.float32)  # Retorna vector vacio cuando no hay targets

    p95 = thresholds["p95"]  # Recupera umbral P95 calculado sobre train para consistencia entre splits
    p99 = thresholds["p99"]  # Recupera umbral P99 calculado sobre train para consistencia entre splits
    p999 = thresholds["p999"]  # Recupera umbral P99.9 calculado sobre train para consistencia entre splits

    sample_weights = np.ones_like(y_array, dtype=np.float32)  # Inicializa pesos base para el regimen dominante de baja magnitud
    sample_weights[event_array] = np.maximum(sample_weights[event_array], 2.5)  # Da mas importancia a eventos aun cuando la magnitud futura todavia es modesta
    sample_weights[(y_array >= p95) & (y_array < p99)] = 6.0  # Aumenta peso para cola alta sin llegar al rango extremo
    sample_weights[(y_array >= p99) & (y_array < p999)] = 12.0  # Aumenta mas el peso para cola muy alta donde la infraestimacion es costosa
    sample_weights[y_array >= p999] = 20.0  # Da maxima prioridad a extremos de magnitud critica
    return sample_weights  # Devuelve vector de pesos por muestra alineado al target


def _compute_event_pos_weight(event_array: np.ndarray) -> float:
    """Compute a bounded positive-class weight for event BCE."""
    if event_array.size == 0:  # Maneja datasets vacios sin romper la clasificacion auxiliar
        return 1.0  # Retorna peso neutro cuando no hay muestras
    positive_count = int(event_array.sum())  # Cuenta cuantas ventanas objetivo pertenecen a evento
    negative_count = int(event_array.size - positive_count)  # Cuenta cuantas ventanas objetivo pertenecen a no-evento
    if positive_count == 0 or negative_count == 0:  # Evita divisiones por cero en casos degenerados
        return 1.0  # Retorna peso neutro cuando solo existe una clase en el split
    raw_ratio = negative_count / positive_count  # Calcula razon negativa/positiva para compensar el desbalance natural
    return float(np.clip(raw_ratio, 1.0, 25.0))  # Limita el peso para evitar gradientes excesivos o inestables


def _build_single_loader(
    df_split: pd.DataFrame,
    feature_columns: List[str],
    target_col: str,
    aux_col: str,
    seq_length: int,
    horizon: int,
    batch_size: int,
    shuffle: bool,
    thresholds: Dict[str, float],
) -> Tuple[DataLoader, Dict[str, object]]:
    """Create one DataLoader and diagnostics for a split."""
    x_array, y_array, event_array = _build_window_arrays(  # Construye ventanas causales y targets alineados al horizonte
        df_split=df_split,  # Usa el split actual como fuente de secuencias
        feature_columns=feature_columns,  # Usa el orden de features definido por el pipeline
        target_col=target_col,  # Usa columna target ya normalizada del split
        aux_col=aux_col,  # Usa bandera auxiliar de evento para supervision multitarea
        seq_length=seq_length,  # Usa longitud de historia definida para el modelo
        horizon=horizon,  # Usa horizonte objetivo definido para la prediccion
    )
    sample_weight_array = _compute_sample_weights(y_array=y_array, event_array=event_array, thresholds=thresholds)  # Calcula pesos con umbrales fijos de train y prioridad explicita a eventos
    event_pos_weight = _compute_event_pos_weight(event_array)  # Calcula peso de la clase positiva para BCE en este split

    dataset = StormflowSequenceDataset(  # Envuelve arreglos en dataset PyTorch con supervision multitarea
        x_array=x_array,  # Pasa ventanas de entrada ya construidas
        y_array=y_array,  # Pasa targets escalares por ventana
        weight_array=sample_weight_array,  # Pasa pesos de muestra para la loss compuesta
        event_array=event_array,  # Pasa etiquetas de evento alineadas con el target futuro
    )
    dataloader = DataLoader(  # Crea DataLoader para iteracion eficiente por lotes
        dataset,  # Usa dataset de secuencias ya construido para el split actual
        batch_size=batch_size,  # Usa tamano de batch configurado para entrenamiento o evaluacion
        shuffle=shuffle,  # Mezcla solo train y preserva orden temporal en validacion/test
        drop_last=False,  # Conserva ultimo batch incompleto para no perder muestras
    )

    diagnostics: Dict[str, object] = {  # Empaqueta estadisticas utiles para trazabilidad de pipeline
        "num_windows": int(len(dataset)),  # Guarda numero total de ventanas creadas para el split
        "x_shape": tuple(x_array.shape),  # Guarda shape de entrada [N, seq_length, n_features]
        "y_shape": tuple(y_array.shape),  # Guarda shape de target [N]
        "num_batches": int(len(dataloader)),  # Guarda cantidad de batches resultantes
        "event_windows": int(event_array.sum()),  # Guarda numero de ventanas cuya etiqueta objetivo es evento
        "event_rate": float(event_array.mean()) if event_array.size > 0 else float("nan"),  # Guarda tasa natural de eventos a ese horizonte
        "event_pos_weight": float(event_pos_weight),  # Guarda peso sugerido para BCE de evento
        "thresholds": thresholds,  # Guarda umbrales P95/P99/P99.9 usados en pesos y diagnosticos
        "shuffle": bool(shuffle),  # Registra si el loader mezcla o preserva orden temporal
    }
    setattr(dataloader, "stormflow_diagnostics", diagnostics)  # Adjunta diagnosticos al loader para recuperarlos sin cambiar la firma publica

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
    """Create train/val/test DataLoaders preserving the natural validation/test distribution."""
    train_thresholds = _compute_quantile_thresholds(df_train[target_col].to_numpy(dtype=np.float32))  # Calcula umbrales de magnitud una sola vez sobre train

    train_loader, train_diag = _build_single_loader(  # Crea loader de train con distribucion natural y mezcla ligera
        df_split=df_train,  # Usa split train como fuente de secuencias
        feature_columns=feature_columns,  # Pasa columnas de entrada en el orden correcto
        target_col=target_col,  # Pasa nombre del target ya normalizado
        aux_col=aux_col,  # Pasa bandera auxiliar de eventos
        seq_length=seq_length,  # Pasa longitud de historia del modelo
        horizon=horizon,  # Pasa horizonte de prediccion deseado
        batch_size=batch_size,  # Pasa tamano de batch configurado
        shuffle=True,  # Mezcla train para romper correlacion intra-batch sin alterar val/test
        thresholds=train_thresholds,  # Usa cuantiles de train para pesos y categorias
    )
    val_loader, val_diag = _build_single_loader(  # Crea loader de validacion con orden estable para analisis temporal
        df_split=df_val,  # Usa split de validacion en orden cronologico
        feature_columns=feature_columns,  # Pasa columnas de entrada
        target_col=target_col,  # Pasa nombre del target normalizado
        aux_col=aux_col,  # Pasa bandera auxiliar para diagnosticos
        seq_length=seq_length,  # Pasa longitud de secuencia del modelo
        horizon=horizon,  # Pasa horizonte de prediccion
        batch_size=batch_size,  # Pasa tamano de batch de evaluacion
        shuffle=False,  # Desactiva mezcla para preservar alineacion temporal
        thresholds=train_thresholds,  # Usa mismos umbrales de train para coherencia de loss/metricas
    )
    test_loader, test_diag = _build_single_loader(  # Crea loader de test con orden estable para analisis temporal
        df_split=df_test,  # Usa split de test en orden cronologico
        feature_columns=feature_columns,  # Pasa columnas de entrada
        target_col=target_col,  # Pasa nombre del target normalizado
        aux_col=aux_col,  # Pasa bandera auxiliar para diagnosticos posteriores
        seq_length=seq_length,  # Pasa longitud de secuencia definida
        horizon=horizon,  # Pasa horizonte objetivo
        batch_size=batch_size,  # Pasa tamano de batch para inferencia/evaluacion
        shuffle=False,  # Desactiva mezcla para preservar orden natural
        thresholds=train_thresholds,  # Reusa umbrales de train para evaluar en la misma escala operativa
    )

    print(f"[sequences] Train X shape: {train_diag['x_shape']} | y shape: {train_diag['y_shape']} | batches: {train_diag['num_batches']}")  # Reporta tamanos de train para validar ventana y batching
    print(f"[sequences] Val   X shape: {val_diag['x_shape']} | y shape: {val_diag['y_shape']} | batches: {val_diag['num_batches']}")  # Reporta tamanos de validacion
    print(f"[sequences] Test  X shape: {test_diag['x_shape']} | y shape: {test_diag['y_shape']} | batches: {test_diag['num_batches']}")  # Reporta tamanos de test

    print(f"[sequences] Event rate(train): {train_diag['event_rate']:.4f} | event BCE pos_weight: {train_diag['event_pos_weight']:.4f}")  # Imprime desbalance natural de eventos en train para configurar la loss
    print(f"[sequences] Event rate(val): {val_diag['event_rate']:.4f} | event windows: {val_diag['event_windows']}")  # Imprime cobertura de eventos en validacion
    print(f"[sequences] Event rate(test): {test_diag['event_rate']:.4f} | event windows: {test_diag['event_windows']}")  # Imprime cobertura de eventos en test
    print(f"[sequences] Weight thresholds(train): {train_diag['thresholds']}")  # Imprime P95/P99/P99.9 usados por pesos y diagnosticos

    return train_loader, val_loader, test_loader  # Devuelve loaders listos para entrenamiento y evaluacion
