"""Diagnostic utilities for full stormflow model inspection."""

from __future__ import annotations  # Permite usar anotaciones modernas sin romper compatibilidad

from typing import Any, Dict, List, Tuple  # Define tipos explicitos para estructuras de diagnostico

import numpy as np  # Aporta operaciones vectorizadas para estadisticas y metricas
import pandas as pd  # Permite leer columnas de los splits normalizados de forma consistente
import torch  # Permite ejecutar inferencia y permutaciones sobre tensores del modelo
from torch.utils.data import DataLoader  # Tipa el DataLoader usado en permutation importance

from src.pipeline.normalize import denormalize_target  # Reutiliza la desnormalizacion oficial del pipeline


def _to_1d_array(values: np.ndarray) -> np.ndarray:
    """Convert any array-like input into a float 1D numpy array."""
    return np.asarray(values, dtype=float).reshape(-1)  # Fuerza vector 1D en float para calculos numericos estables


def _safe_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute RMSE safely for possibly empty arrays."""
    if y_true.size == 0:  # Evita media sobre arreglos vacios
        return float("nan")  # Marca metrica no definida cuando no hay datos
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))  # Calcula raiz del error cuadratico medio


def _safe_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute MAE safely for possibly empty arrays."""
    if y_true.size == 0:  # Evita media sobre arreglos vacios
        return float("nan")  # Marca metrica no definida cuando no hay datos
    return float(np.mean(np.abs(y_pred - y_true)))  # Calcula error absoluto medio


def _safe_nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Nash-Sutcliffe Efficiency safely."""
    if y_true.size == 0:  # Evita calcular NSE sin muestras
        return float("nan")  # Devuelve NaN cuando la metrica no aplica
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)  # Calcula la varianza observada usada como denominador
    if denominator <= 0:  # Controla caso degenerado sin variabilidad en y_true
        return float("nan")  # Marca NSE como no interpretable en ese caso
    numerator = np.sum((y_true - y_pred) ** 2)  # Calcula suma de errores cuadrados del modelo
    return float(1.0 - (numerator / denominator))  # Retorna NSE final donde 1.0 es ajuste perfecto


def _safe_correlation(x_values: np.ndarray, y_values: np.ndarray) -> float:
    """Compute Pearson correlation safely for possibly degenerate arrays."""
    if x_values.size == 0 or y_values.size == 0:  # Evita calcular correlacion si falta alguno de los vectores
        return float("nan")  # Retorna NaN para indicar ausencia de datos suficientes
    if x_values.size != y_values.size:  # Valida alineacion longitud a longitud entre ambos vectores
        raise ValueError("x_values and y_values must have the same length")  # Lanza error claro si hay desalineacion
    if np.isclose(np.std(x_values), 0.0) or np.isclose(np.std(y_values), 0.0):  # Evita division por cero implicita en correlacion
        return float("nan")  # Devuelve NaN cuando no hay variabilidad en alguno de los vectores
    return float(np.corrcoef(x_values, y_values)[0, 1])  # Retorna coeficiente de Pearson del par de series


def _distribution_stats(y_norm: np.ndarray) -> Dict[str, float]:
    """Build normalized target distribution summary for one split."""
    if y_norm.size == 0:  # Maneja split vacio sin romper la estructura del reporte
        return {  # Devuelve estructura completa con NaN/0 para mantener contrato estable
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "median": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "p999": float("nan"),
            "pct_eq_0": 0.0,
            "pct_lt_0_01": 0.0,
            "pct_lt_0_05": 0.0,
            "pct_lt_0_10": 0.0,
            "n_samples": 0.0,
        }

    return {  # Construye resumen pedido para diagnosticar compresion en escala normalizada
        "min": float(np.min(y_norm)),  # Reporta minimo del target normalizado en el split
        "max": float(np.max(y_norm)),  # Reporta maximo del target normalizado en el split
        "mean": float(np.mean(y_norm)),  # Reporta media del target normalizado en el split
        "median": float(np.median(y_norm)),  # Reporta mediana para robustez frente a cola larga
        "p95": float(np.quantile(y_norm, 0.95)),  # Reporta percentil 95 del target normalizado
        "p99": float(np.quantile(y_norm, 0.99)),  # Reporta percentil 99 del target normalizado
        "p999": float(np.quantile(y_norm, 0.999)),  # Reporta percentil 99.9 del target normalizado
        "pct_eq_0": float(np.mean(y_norm == 0.0) * 100.0),  # Reporta porcentaje exacto de valores en cero
        "pct_lt_0_01": float(np.mean(y_norm < 0.01) * 100.0),  # Reporta porcentaje de valores menores a 0.01
        "pct_lt_0_05": float(np.mean(y_norm < 0.05) * 100.0),  # Reporta porcentaje de valores menores a 0.05
        "pct_lt_0_10": float(np.mean(y_norm < 0.10) * 100.0),  # Reporta porcentaje de valores menores a 0.10
        "n_samples": float(y_norm.size),  # Reporta cantidad total de muestras del split
    }


def _bias_by_severity(y_real_mgd: np.ndarray, y_pred_mgd: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Compute bias by severity bucket using requested MGD thresholds."""
    buckets: List[Tuple[str, float | None, float | None]] = [  # Define cortes de severidad solicitados por el usuario
        ("base", None, 0.5),  # Bucket base para muestras por debajo de 0.5 MGD
        ("pequeno", 0.5, 2.0),  # Bucket pequeno para rango de 0.5 a 2.0 MGD
        ("moderado", 2.0, 13.0),  # Bucket moderado para rango de 2.0 a 13.0 MGD
        ("grande", 13.0, 51.0),  # Bucket grande para rango de 13.0 a 51.0 MGD
        ("extremo", 51.0, None),  # Bucket extremo para valores mayores o iguales a 51.0 MGD
    ]

    results: Dict[str, Dict[str, float]] = {}  # Prepara contenedor de bias por bucket
    for bucket_name, lower_bound, upper_bound in buckets:  # Recorre cada bucket para aplicar su mascara
        if lower_bound is None:  # Maneja bucket con solo limite superior
            mask = y_real_mgd < upper_bound  # Selecciona muestras reales por debajo del limite superior
        elif upper_bound is None:  # Maneja bucket con solo limite inferior
            mask = y_real_mgd >= lower_bound  # Selecciona muestras reales por encima o igual al limite inferior
        else:  # Maneja bucket con limites inferior y superior
            mask = (y_real_mgd >= lower_bound) & (y_real_mgd < upper_bound)  # Selecciona muestras dentro del intervalo

        if not mask.any():  # Controla bucket sin muestras para evitar medias invalidas
            results[bucket_name] = {"bias": float("nan"), "n_samples": 0.0}  # Guarda salida vacia manteniendo estructura
            continue  # Continua con el siguiente bucket sin intentar calculos adicionales

        bucket_bias = float(np.mean(y_pred_mgd[mask] - y_real_mgd[mask]))  # Calcula sesgo firmado (pred-real) en bucket actual
        results[bucket_name] = {"bias": bucket_bias, "n_samples": float(mask.sum())}  # Guarda bias y conteo para trazabilidad

    return results  # Devuelve estructura completa de bias por severidad


def _top_peak_ratios(y_real_mgd: np.ndarray, y_pred_mgd: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
    """Compute pred/real ratio for the top-k largest real peaks."""
    if y_real_mgd.size == 0:  # Maneja evaluaciones vacias sin romper la serializacion
        return []  # Retorna lista vacia cuando no hay muestras

    ranked_indices = np.argsort(y_real_mgd)[::-1]  # Ordena indices por magnitud real descendente
    selected_indices = ranked_indices[: min(top_k, ranked_indices.size)]  # Toma solo los primeros top_k indices disponibles

    peak_rows: List[Dict[str, Any]] = []  # Inicializa lista de filas con detalle por pico
    for rank_position, sample_index in enumerate(selected_indices, start=1):  # Recorre picos seleccionados con ranking humano 1-based
        real_value = float(y_real_mgd[sample_index])  # Extrae magnitud real del pico actual
        pred_value = float(y_pred_mgd[sample_index])  # Extrae prediccion del mismo indice del pico real
        ratio_value = float(pred_value / real_value) if real_value > 0.0 else float("nan")  # Calcula ratio pred/real evitando division por cero
        peak_rows.append(  # Agrega fila serializable para auditoria puntual de los picos mas altos
            {
                "rank": int(rank_position),  # Guarda posicion en ranking de picos reales
                "sample_index": int(sample_index),  # Guarda indice original para trazabilidad en la serie
                "real_mgd": real_value,  # Guarda valor real del pico en MGD
                "pred_mgd": pred_value,  # Guarda valor predicho del pico en MGD
                "pred_real_ratio": ratio_value,  # Guarda ratio prediccion/real para ver compresion o sobreestimacion
            }
        )

    return peak_rows  # Devuelve lista con detalle de top-10 picos reales


def _extract_xy_from_batch(
    batch: Tuple[torch.Tensor, ...],
    resolved_device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract X and y from a training batch with 3 or 4 tensors."""
    if len(batch) == 4:  # Soporta loaders con metadata adicional de evento
        x_batch, y_batch, _w_batch, _event_batch = batch  # Ignora pesos y metadata porque solo se requiere inferencia
    elif len(batch) == 3:  # Soporta loaders clasicos que solo entregan X, y y pesos
        x_batch, y_batch, _w_batch = batch  # Ignora pesos en este diagnostico porque RMSE usa pred y target
    else:  # Detecta formatos inesperados para evitar errores silenciosos
        raise ValueError("Expected dataloader batches with 3 or 4 tensors")  # Lanza error claro para depuracion del pipeline
    x_batch = x_batch.to(resolved_device)  # Mueve features al dispositivo de inferencia configurado
    y_batch = y_batch.to(resolved_device)  # Mueve target al mismo dispositivo para alinear comparacion
    return x_batch, y_batch  # Devuelve solo tensores necesarios para prediccion y RMSE

def _predict_over_loader(
    model: torch.nn.Module,
    dataloader: DataLoader,
    resolved_device: torch.device,
    permuted_feature_index: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference over a loader, optionally permuting one feature column."""
    predictions: List[np.ndarray] = []  # Acumula predicciones batch a batch para concatenar al final
    targets: List[np.ndarray] = []  # Acumula targets reales batch a batch para medir RMSE comparable
    with torch.no_grad():  # Desactiva gradientes porque este flujo es solo diagnostico
        for batch in dataloader:  # Recorre todos los batches del dataloader recibido
            x_batch, y_batch = _extract_xy_from_batch(  # Reutiliza rutina comun para aceptar firmas 3 o 4 tensores
                batch=batch,  # Pasa el batch crudo emitido por el DataLoader
                resolved_device=resolved_device,  # Usa el dispositivo resuelto por la funcion llamadora
            )
            if permuted_feature_index is not None:  # Solo aplica permutacion cuando se evalua una feature especifica
                if x_batch.ndim != 3:  # Valida forma esperada (batch, seq_length, n_features) antes de permutar
                    raise ValueError("Expected x_batch with shape (batch, seq_length, n_features)")  # Lanza error claro si la forma del tensor no coincide
                if not (0 <= permuted_feature_index < x_batch.shape[2]):  # Protege contra indices fuera de rango en columnas de features
                    raise ValueError("permuted_feature_index is out of bounds for input features")  # Lanza error explicito para depuracion rapida
                x_batch = x_batch.clone()  # Clona batch para no mutar el tensor original compartido por el DataLoader
                feature_values = x_batch[:, :, permuted_feature_index].reshape(-1)  # Toma todos los valores de la feature en batch y tiempo
                permutation_indices = torch.randperm(feature_values.numel(), device=resolved_device)  # Construye permutacion aleatoria en el mismo dispositivo
                feature_values = feature_values[permutation_indices]  # Reordena valores para romper asociacion feature-target
                x_batch[:, :, permuted_feature_index] = feature_values.view_as(x_batch[:, :, permuted_feature_index])  # Reescribe la feature permutada conservando forma original
            y_pred = model(x_batch)  # Ejecuta forward del modelo con batch original o permutado
            if not isinstance(y_pred, torch.Tensor):  # Valida firma de salida esperada para evitar incompatibilidades silenciosas
                raise TypeError("Model output must be a torch.Tensor")  # Lanza error claro si la salida del modelo no es tensor
            predictions.append(y_pred.detach().cpu().numpy().reshape(-1))  # Convierte prediccion a numpy 1D y la acumula
            targets.append(y_batch.detach().cpu().numpy().reshape(-1))  # Convierte target a numpy 1D y lo acumula
    if not predictions:  # Maneja loaders vacios para no romper concatenacion
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)  # Devuelve arreglos vacios si no hubo batches
    y_pred_array = np.concatenate(predictions, axis=0)  # Concatena predicciones de todos los batches en un unico vector
    y_true_array = np.concatenate(targets, axis=0)  # Concatena targets reales de todos los batches en un unico vector
    return y_pred_array, y_true_array  # Retorna pares comparables para calculo de RMSE

def run_permutation_importance(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device | str,
    feature_columns: List[str],
    norm_params: Dict[str, object],
) -> List[Dict[str, float]]:
    """Compute permutation importance ranking using RMSE delta per feature."""
    resolved_device = torch.device(device)  # Normaliza dispositivo recibido para uso consistente en PyTorch
    model = model.to(resolved_device)  # Mueve modelo al dispositivo de inferencia para evitar copias implicitas
    model_was_training = model.training  # Guarda estado previo para restaurarlo al finalizar el diagnostico
    model.eval()  # Fuerza modo evaluacion para desactivar dropout y estabilizar comparaciones
    baseline_pred_norm, baseline_true_norm = _predict_over_loader(  # Obtiene baseline sin permutar ninguna feature
        model=model,  # Reutiliza el mismo modelo entrenado para toda la comparacion
        dataloader=dataloader,  # Recorre exactamente el mismo conjunto de muestras para baseline
        resolved_device=resolved_device,  # Usa mismo dispositivo para mantener costos y precision comparables
        permuted_feature_index=None,  # No permuta columnas en la corrida base
    )
    baseline_pred_mgd = denormalize_target(baseline_pred_norm, norm_params)  # Convierte predicciones baseline a MGD reales para interpretacion hidrologica
    baseline_true_mgd = denormalize_target(baseline_true_norm, norm_params)  # Convierte target baseline a MGD reales con los mismos parametros
    baseline_pred_mgd = np.clip(baseline_pred_mgd, a_min=0.0, a_max=None)  # Impone restriccion fisica de no negatividad en prediccion
    baseline_true_mgd = np.clip(baseline_true_mgd, a_min=0.0, a_max=None)  # Impone restriccion fisica de no negatividad en target real
    baseline_rmse_mgd = _safe_rmse(y_true=baseline_true_mgd, y_pred=baseline_pred_mgd)  # Calcula RMSE base en unidades reales
    ranking_rows: List[Dict[str, float]] = []  # Inicializa lista de resultados por feature para ranking final
    for feature_index, feature_name in enumerate(feature_columns):  # Recorre todas las columnas para estimar impacto individual
        permuted_pred_norm, permuted_true_norm = _predict_over_loader(  # Ejecuta inferencia permutando solo la feature actual
            model=model,  # Reutiliza mismo modelo para mantener comparabilidad
            dataloader=dataloader,  # Recorre mismo conjunto para aislar efecto de la permutacion
            resolved_device=resolved_device,  # Mantiene mismo dispositivo de ejecucion
            permuted_feature_index=feature_index,  # Selecciona feature puntual a barajar en todos los batches
        )
        permuted_pred_mgd = denormalize_target(permuted_pred_norm, norm_params)  # Convierte predicciones permutadas a MGD reales
        permuted_true_mgd = denormalize_target(permuted_true_norm, norm_params)  # Convierte target permutado a MGD reales para consistencia
        permuted_pred_mgd = np.clip(permuted_pred_mgd, a_min=0.0, a_max=None)  # Mantiene restriccion fisica en predicciones permutadas
        permuted_true_mgd = np.clip(permuted_true_mgd, a_min=0.0, a_max=None)  # Mantiene restriccion fisica en targets permutados
        permuted_rmse_mgd = _safe_rmse(y_true=permuted_true_mgd, y_pred=permuted_pred_mgd)  # Calcula RMSE con feature perturbada
        delta_rmse_mgd = float(permuted_rmse_mgd - baseline_rmse_mgd)  # Mide impacto absoluto de romper la feature en MGD
        if np.isfinite(baseline_rmse_mgd) and baseline_rmse_mgd > 0.0:  # Evita division por cero al calcular cambio relativo
            relative_increase_pct = float((delta_rmse_mgd / baseline_rmse_mgd) * 100.0)  # Convierte impacto a porcentaje relativo al baseline
        else:  # Maneja baseline degenerado para no introducir infinidades en el ranking
            relative_increase_pct = float("nan")  # Marca porcentaje como no definido cuando baseline no es util
        ranking_rows.append(  # Agrega fila con metrica completa de importancia para la feature actual
            {
                "feature": str(feature_name),  # Guarda nombre de feature para lectura humana del ranking
                "baseline_rmse_mgd": float(baseline_rmse_mgd),  # Repite RMSE base para trazabilidad en cada fila
                "permuted_rmse_mgd": float(permuted_rmse_mgd),  # Guarda RMSE observado tras permutar la feature
                "delta_rmse_mgd": delta_rmse_mgd,  # Guarda incremento absoluto de RMSE usado como importancia principal
                "relative_increase_pct": relative_increase_pct,  # Guarda incremento relativo para comparar features de forma estandarizada
            }
        )
    ranking_rows.sort(key=lambda row: row["delta_rmse_mgd"], reverse=True)  # Ordena de mayor a menor impacto para ranking final
    print("[diag] === Permutation Importance (RMSE delta en MGD) ===")  # Imprime encabezado para separar esta seccion diagnostica
    for rank_index, row in enumerate(ranking_rows, start=1):  # Recorre ranking ya ordenado para mostrar top de importancia
        print(  # Reporta posicion, feature y cambios absolutos/relativos de RMSE
            f"[diag] #{rank_index:02d} {row['feature']}: "
            f"delta_rmse={row['delta_rmse_mgd']:.6f} MGD | "
            f"permuted_rmse={row['permuted_rmse_mgd']:.6f} | "
            f"rel_increase={row['relative_increase_pct']:.2f}%"
        )
    if model_was_training:  # Restaura estado original por si el llamador continua entrenando despues del diagnostico
        model.train()  # Reactiva modo entrenamiento solo si el modelo estaba previamente en train
    return ranking_rows  # Devuelve ranking serializable para guardar en JSON o markdown

def run_full_diagnostics(
    y_pred_norm: np.ndarray,
    y_real_norm: np.ndarray,
    norm_params: Dict[str, object],
    df_train_norm: pd.DataFrame,
    df_val_norm: pd.DataFrame,
    df_test_norm: pd.DataFrame,
    target_col: str,
) -> Dict[str, Any]:
    """Run full diagnostics for normalized target distribution, predictions, residuals, and normalization."""
    y_pred_norm_array = _to_1d_array(y_pred_norm)  # Normaliza entrada de predicciones a vector 1D float
    y_real_norm_array = _to_1d_array(y_real_norm)  # Normaliza entrada de target real a vector 1D float

    if y_pred_norm_array.shape[0] != y_real_norm_array.shape[0]:  # Valida alineacion exacta entre prediccion y target
        raise ValueError("y_pred_norm and y_real_norm must have the same length")  # Lanza error claro si longitudes no coinciden

    train_target_norm = _to_1d_array(df_train_norm[target_col].to_numpy())  # Extrae target normalizado de train para su resumen
    val_target_norm = _to_1d_array(df_val_norm[target_col].to_numpy())  # Extrae target normalizado de val para su resumen
    test_target_norm = _to_1d_array(df_test_norm[target_col].to_numpy())  # Extrae target normalizado de test para su resumen

    distributions = {  # Agrupa distribucion normalizada por split tal como fue solicitado
        "train": _distribution_stats(train_target_norm),  # Calcula resumen estadistico de train en escala normalizada
        "val": _distribution_stats(val_target_norm),  # Calcula resumen estadistico de val en escala normalizada
        "test": _distribution_stats(test_target_norm),  # Calcula resumen estadistico de test en escala normalizada
    }

    y_pred_mgd = denormalize_target(y_pred_norm_array, norm_params)  # Convierte predicciones normalizadas a MGD reales
    y_real_mgd = denormalize_target(y_real_norm_array, norm_params)  # Convierte target real normalizado a MGD real
    y_pred_mgd = np.clip(y_pred_mgd, a_min=0.0, a_max=None)  # Aplica restriccion fisica de no negatividad en prediccion
    y_real_mgd = np.clip(y_real_mgd, a_min=0.0, a_max=None)  # Aplica restriccion fisica de no negatividad en target real

    nse_value = _safe_nse(y_real_mgd, y_pred_mgd)  # Calcula NSE global en unidades fisicas
    rmse_value = _safe_rmse(y_real_mgd, y_pred_mgd)  # Calcula RMSE global en MGD
    mae_value = _safe_mae(y_real_mgd, y_pred_mgd)  # Calcula MAE global en MGD

    if y_real_mgd.size > 0:  # Verifica que existan muestras para calcular diagnostico de pico
        peak_real_mgd = float(np.max(y_real_mgd))  # Obtiene magnitud maxima real observada
        peak_pred_mgd = float(np.max(y_pred_mgd))  # Obtiene magnitud maxima predicha por el modelo
        peak_error_mgd = float(peak_pred_mgd - peak_real_mgd)  # Calcula error firmado del pico global
        peak_error_pct = float((peak_error_mgd / max(peak_real_mgd, 1e-12)) * 100.0)  # Calcula error relativo de pico en porcentaje
    else:  # Maneja vector vacio para mantener estructura del reporte estable
        peak_real_mgd = float("nan")  # Marca pico real como no definido
        peak_pred_mgd = float("nan")  # Marca pico predicho como no definido
        peak_error_mgd = float("nan")  # Marca error absoluto de pico como no definido
        peak_error_pct = float("nan")  # Marca error porcentual de pico como no definido

    severity_bias = _bias_by_severity(y_real_mgd=y_real_mgd, y_pred_mgd=y_pred_mgd)  # Calcula sesgo por buckets de severidad solicitados
    top_10_ratios = _top_peak_ratios(y_real_mgd=y_real_mgd, y_pred_mgd=y_pred_mgd, top_k=10)  # Resume calibracion en los 10 picos mas altos

    residuals_mgd = y_pred_mgd - y_real_mgd  # Define residuo firmado como pred-real para separar sobre e infraestimacion
    residual_mean = float(np.mean(residuals_mgd)) if residuals_mgd.size > 0 else float("nan")  # Calcula media de residuos para sesgo global
    residual_median = float(np.median(residuals_mgd)) if residuals_mgd.size > 0 else float("nan")  # Calcula mediana robusta de residuos
    residual_std = float(np.std(residuals_mgd)) if residuals_mgd.size > 0 else float("nan")  # Calcula dispersion de residuos en MGD
    pct_over_gt_2 = float(np.mean(residuals_mgd > 2.0) * 100.0) if residuals_mgd.size > 0 else 0.0  # Calcula porcentaje con sobreestimacion mayor a 2 MGD
    pct_under_gt_2 = float(np.mean(residuals_mgd < -2.0) * 100.0) if residuals_mgd.size > 0 else 0.0  # Calcula porcentaje con infraestimacion mayor a 2 MGD
    residual_corr = _safe_correlation(x_values=y_real_mgd, y_values=residuals_mgd)  # Estima correlacion entre magnitud real y residuo

    q95_norm = float(np.quantile(y_real_norm_array, 0.95)) if y_real_norm_array.size > 0 else float("nan")  # Calcula P95 en espacio normalizado del vector evaluado
    q99_norm = float(np.quantile(y_real_norm_array, 0.99)) if y_real_norm_array.size > 0 else float("nan")  # Calcula P99 en espacio normalizado del vector evaluado
    q95_real = float(np.quantile(y_real_mgd, 0.95)) if y_real_mgd.size > 0 else float("nan")  # Calcula P95 en MGD reales del mismo vector
    q99_real = float(np.quantile(y_real_mgd, 0.99)) if y_real_mgd.size > 0 else float("nan")  # Calcula P99 en MGD reales del mismo vector

    norm_span = float(q99_norm - q95_norm) if np.isfinite(q99_norm) and np.isfinite(q95_norm) else float("nan")  # Calcula ancho de banda P95-P99 normalizado
    real_span = float(q99_real - q95_real) if np.isfinite(q99_real) and np.isfinite(q95_real) else float("nan")  # Calcula ancho de banda P95-P99 en MGD reales
    if np.isfinite(norm_span) and np.isfinite(real_span) and real_span > 0.0:  # Verifica que ambos spans sean validos antes de dividir
        compression_ratio = float(norm_span / real_span)  # Calcula ratio solicitado para medir compresion relativa
    else:  # Maneja caso degenerado sin rango real util
        compression_ratio = float("nan")  # Marca ratio no definido cuando no hay base numerica valida

    flag_train_lt_005 = bool(distributions["train"]["pct_lt_0_05"] > 80.0)  # Marca si train esta muy concentrado por debajo de 0.05
    flag_val_lt_005 = bool(distributions["val"]["pct_lt_0_05"] > 80.0)  # Marca si val esta muy concentrado por debajo de 0.05
    flag_test_lt_005 = bool(distributions["test"]["pct_lt_0_05"] > 80.0)  # Marca si test esta muy concentrado por debajo de 0.05

    normalization_diagnostics = {  # Agrupa seccion de normalizacion solicitada
        "effective_target_range_norm": {  # Reporta rango efectivo del target normalizado por split
            "train": {  # Guarda min, max y rango de train normalizado
                "min": distributions["train"]["min"],
                "max": distributions["train"]["max"],
                "range": float(distributions["train"]["max"] - distributions["train"]["min"]),
            },
            "val": {  # Guarda min, max y rango de val normalizado
                "min": distributions["val"]["min"],
                "max": distributions["val"]["max"],
                "range": float(distributions["val"]["max"] - distributions["val"]["min"]),
            },
            "test": {  # Guarda min, max y rango de test normalizado
                "min": distributions["test"]["min"],
                "max": distributions["test"]["max"],
                "range": float(distributions["test"]["max"] - distributions["test"]["min"]),
            },
        },
        "compression_p95_p99": {  # Reporta diagnostico de compresion entre escalas normalizada y real
            "p95_norm": q95_norm,
            "p99_norm": q99_norm,
            "p95_real_mgd": q95_real,
            "p99_real_mgd": q99_real,
            "norm_span_p95_p99": norm_span,
            "real_span_p95_p99": real_span,
            "norm_real_span_ratio": compression_ratio,
        },
        "high_mass_below_0_05_flags": {  # Reporta flags de concentracion baja por split
            "train_gt_80pct": flag_train_lt_005,
            "val_gt_80pct": flag_val_lt_005,
            "test_gt_80pct": flag_test_lt_005,
            "any_split_gt_80pct": bool(flag_train_lt_005 or flag_val_lt_005 or flag_test_lt_005),
        },
    }

    diagnostics: Dict[str, Any] = {  # Construye diccionario final completo para persistir en JSON
        "normalized_target_distribution": distributions,  # Incluye estadistica del target normalizado para train/val/test
        "prediction_diagnostics": {  # Incluye diagnostico de predicciones en MGD reales
            "global_metrics": {  # Resume metricas globales principales de desempeno
                "nse": nse_value,
                "rmse": rmse_value,
                "mae": mae_value,
            },
            "severity_bias": severity_bias,  # Incluye sesgo por buckets base-pequeno-moderado-grande-extremo
            "peak_diagnostics": {  # Incluye diagnosticos de pico global solicitados
                "peak_real_mgd": peak_real_mgd,
                "peak_pred_mgd": peak_pred_mgd,
                "peak_error_mgd": peak_error_mgd,
                "peak_error_pct": peak_error_pct,
                "max_pred_mgd": peak_pred_mgd,
                "max_real_mgd": peak_real_mgd,
            },
            "top_10_peak_pred_real_ratios": top_10_ratios,  # Incluye ratio pred/real de los 10 picos reales mas altos
        },
        "residual_diagnostics": {  # Incluye resumen estadistico y comportamiento de residuos
            "mean": residual_mean,
            "median": residual_median,
            "std": residual_std,
            "pct_overestimation_gt_2_mgd": pct_over_gt_2,
            "pct_underestimation_gt_2_mgd": pct_under_gt_2,
            "corr_real_magnitude_vs_residual": residual_corr,
        },
        "normalization_diagnostics": normalization_diagnostics,  # Incluye chequeos de compresion y concentracion en baja escala
    }

    print("[diag] === Distribucion Target Normalizado ===")  # Imprime encabezado de la seccion 1 para lectura rapida en consola
    print(  # Imprime resumen corto por split con cuantiles y concentracion baja
        f"[diag] train pct<0.05={distributions['train']['pct_lt_0_05']:.2f}% | "
        f"val pct<0.05={distributions['val']['pct_lt_0_05']:.2f}% | "
        f"test pct<0.05={distributions['test']['pct_lt_0_05']:.2f}%"
    )

    print("[diag] === Predicciones (MGD) ===")  # Imprime encabezado de la seccion 2 para metricas globales
    print(  # Resume NSE, RMSE y MAE para comparar rapido entre iteraciones
        f"[diag] NSE={nse_value:.4f} | RMSE={rmse_value:.4f} | MAE={mae_value:.4f}"
    )
    print(  # Resume diagnostico del pico global para ver compresion/expansion de amplitud
        f"[diag] Pico real={peak_real_mgd:.4f} | Pico pred={peak_pred_mgd:.4f} | Error%={peak_error_pct:.2f}%"
    )

    print("[diag] === Residuos ===")  # Imprime encabezado de la seccion 3 para sesgo y dispersion
    print(  # Resume tendencia central y dispersion de residuos
        f"[diag] mean={residual_mean:.4f} | median={residual_median:.4f} | std={residual_std:.4f}"
    )
    print(  # Resume tasas de errores grandes de sobre/infraestimacion
        f"[diag] over>2MGD={pct_over_gt_2:.2f}% | under>2MGD={pct_under_gt_2:.2f}% | corr(real,res)={residual_corr:.4f}"
    )

    print("[diag] === Normalizacion ===")  # Imprime encabezado de la seccion 4 para diagnostico de compresion
    print(  # Reporta ratio de compresion entre span normalizado y span real en P95-P99
        f"[diag] Compression ratio P95-P99 (norm/real)={compression_ratio:.6f}"
    )
    print(  # Reporta si hay concentracion excesiva bajo 0.05 en alguno de los splits
        f"[diag] Flag >80% target_norm <0.05 (any split): {normalization_diagnostics['high_mass_below_0_05_flags']['any_split_gt_80pct']}"
    )

    return diagnostics  # Devuelve el diccionario completo para guardado posterior en JSON
