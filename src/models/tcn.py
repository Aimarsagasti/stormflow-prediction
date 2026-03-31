"""TCN model definitions for stormflow prediction."""

from __future__ import annotations  # Permite anotaciones modernas de tipos sin problemas de version

from typing import List, Sequence  # Define tipos para listas de canales y dilataciones

import torch  # Provee tensores y operaciones base para el modelo
from torch import nn  # Incluye modulos de red neuronal en PyTorch


class CausalConv1d(nn.Module):
    """1D causal convolution implemented with left padding and right trimming."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()  # Inicializa correctamente la clase base nn.Module
        self.causal_padding = (kernel_size - 1) * dilation  # Calcula padding total necesario para mantener causalidad
        self.conv = nn.Conv1d(  # Define convolucion 1D con dilatacion para ampliar campo receptivo
            in_channels=in_channels,  # Define numero de canales de entrada del bloque
            out_channels=out_channels,  # Define numero de canales de salida del bloque
            kernel_size=kernel_size,  # Define tamano de kernel temporal
            dilation=dilation,  # Define separacion entre elementos del kernel para cubrir mas historia
            padding=self.causal_padding,  # Aplica padding simetrico y luego se recorta para dejar solo informacion pasada
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)  # Ejecuta convolucion temporal sobre la secuencia
        if self.causal_padding > 0:  # Verifica si hace falta recorte para mantener longitud y causalidad
            x = x[:, :, :-self.causal_padding]  # Elimina posiciones futuras introducidas por el padding derecho
        return x  # Devuelve salida causal con la misma longitud temporal que la entrada


def _build_group_norm(num_channels: int) -> nn.GroupNorm:
    """Build GroupNorm with a valid number of groups for the channel count."""
    candidate_groups = [8, 4, 2, 1]  # Prueba grupos tipicos para mantener normalizacion estable sin depender del batch
    for num_groups in candidate_groups:  # Recorre posibles grupos hasta encontrar uno compatible con los canales actuales
        if num_channels % num_groups == 0:  # Verifica divisibilidad exacta requerida por GroupNorm
            return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)  # Devuelve normalizacion estable para el ancho actual
    return nn.GroupNorm(num_groups=1, num_channels=num_channels)  # Fallback seguro equivalente a LayerNorm por canal


class TCNResidualBlock(nn.Module):
    """Residual TCN block with two dilated causal convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()  # Inicializa estructura base del bloque residual
        self.conv1 = CausalConv1d(  # Primera convolucion causal dilatada del bloque
            in_channels=in_channels,  # Recibe canales de entrada del bloque
            out_channels=out_channels,  # Proyecta al numero de canales objetivo del bloque
            kernel_size=kernel_size,  # Usa kernel definido para toda la arquitectura
            dilation=dilation,  # Usa dilatacion del bloque para cubrir diferente escala temporal
        )
        self.norm1 = _build_group_norm(out_channels)  # Normaliza por grupos para evitar drift con batches no representativos
        self.relu1 = nn.ReLU()  # Introduce no linealidad tras la primera convolucion
        self.drop1 = nn.Dropout(dropout)  # Regulariza activaciones para reducir sobreajuste

        self.conv2 = CausalConv1d(  # Segunda convolucion causal dilatada del bloque
            in_channels=out_channels,  # Usa salida intermedia como nueva entrada del bloque
            out_channels=out_channels,  # Mantiene dimensionalidad para sumar con el skip connection
            kernel_size=kernel_size,  # Repite tamano de kernel del bloque
            dilation=dilation,  # Repite dilatacion para consistencia multiescala dentro del bloque
        )
        self.norm2 = _build_group_norm(out_channels)  # Repite GroupNorm para estabilizar la segunda transformacion del bloque
        self.relu2 = nn.ReLU()  # Aplica no linealidad en segunda transformacion
        self.drop2 = nn.Dropout(dropout)  # Aplica regularizacion adicional en la segunda capa

        if in_channels != out_channels:  # Revisa si el residual requiere ajustar cantidad de canales
            self.skip_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)  # Proyecta residual con Conv1x1 cuando cambian canales
        else:  # Usa atajo identidad cuando las dimensiones ya coinciden
            self.skip_proj = nn.Identity()  # Evita costo extra si no hace falta proyeccion

        self.out_relu = nn.ReLU()  # Activa salida combinada residual para mantener estabilidad y no linealidad final

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip_proj(x)  # Construye rama residual alineada en canales con la salida principal

        out = self.conv1(x)  # Ejecuta primera convolucion causal
        out = self.norm1(out)  # Normaliza salida de la primera convolucion sin depender del batch
        out = self.relu1(out)  # Activa salida intermedia para modelar relaciones no lineales
        out = self.drop1(out)  # Aplica dropout en activaciones intermedias

        out = self.conv2(out)  # Ejecuta segunda convolucion causal del bloque
        out = self.norm2(out)  # Normaliza segunda salida convolucional con GroupNorm
        out = self.relu2(out)  # Activa segunda salida intermedia
        out = self.drop2(out)  # Aplica dropout final antes de la suma residual

        out = out + residual  # Suma rama principal y residual para facilitar flujo de gradiente
        out = self.out_relu(out)  # Aplica activacion final tras la fusion residual
        return out  # Devuelve salida del bloque con misma longitud temporal


class StormflowTCN(nn.Module):
    """Residual TCN with direct scalar regression head for stormflow prediction."""

    def __init__(
        self,
        n_features: int,
        num_channels: Sequence[int] | None = None,
        dilations: Sequence[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()  # Inicializa la clase base para registrar submodulos correctamente
        self.n_features = n_features  # Guarda numero de features de entrada para referencia y depuracion
        self.num_channels = list(num_channels) if num_channels is not None else [32, 64, 64, 64, 32]  # Define canales por bloque segun propuesta
        self.dilations = list(dilations) if dilations is not None else [1, 2, 4, 8, 16]  # Define dilataciones por bloque segun propuesta
        self.kernel_size = kernel_size  # Guarda kernel temporal para metodos de campo receptivo
        self.dropout = dropout  # Guarda dropout global para bloques y cabeza de regresion

        if len(self.num_channels) != len(self.dilations):  # Verifica consistencia entre numero de bloques y dilataciones
            raise ValueError("num_channels and dilations must have the same length")  # Lanza error claro si hay configuracion inconsistente

        self.input_projection = nn.Conv1d(  # Proyecta features de entrada a canales iniciales de la TCN
            in_channels=n_features,  # Recibe numero de features por timestamp
            out_channels=self.num_channels[0],  # Mapea al ancho del primer bloque residual
            kernel_size=1,  # Usa Conv1x1 para mezclar features sin alterar longitud temporal
        )

        blocks: List[nn.Module] = []  # Acumula bloques residuales para construir la red temporal
        in_channels = self.num_channels[0]  # Inicializa canales de entrada del primer bloque
        for out_channels, dilation in zip(self.num_channels, self.dilations):  # Recorre canales y dilataciones definidos por bloque
            block = TCNResidualBlock(  # Crea bloque residual causal para la escala temporal actual
                in_channels=in_channels,  # Usa ancho actual de la representacion temporal
                out_channels=out_channels,  # Configura ancho de salida del bloque
                kernel_size=self.kernel_size,  # Usa kernel comun a toda la arquitectura
                dilation=dilation,  # Usa dilatacion especifica del bloque
                dropout=self.dropout,  # Usa dropout definido a nivel de modelo
            )
            blocks.append(block)  # Agrega bloque creado a la lista secuencial
            in_channels = out_channels  # Actualiza canales de entrada para el siguiente bloque
        self.tcn_blocks = nn.Sequential(*blocks)  # Empaqueta bloques en una secuencia ejecutable

        final_channels = self.num_channels[-1]  # Obtiene canales finales tras el ultimo bloque TCN
        self.regression_head = nn.Sequential(  # Define MLP final para mapear estado causal a prediccion escalar
            nn.Linear(final_channels, 128),  # Proyecta estado final a un espacio intermedio con mayor capacidad
            nn.ReLU(),  # Introduce no linealidad para aprender relaciones hidrologicas complejas
            nn.Dropout(self.dropout),  # Regulariza activaciones intermedias para reducir sobreajuste
            nn.Linear(128, 64),  # Reduce dimensionalidad para estabilizar la etapa final de regresion
            nn.ReLU(),  # Introduce una segunda no linealidad antes de la salida
            nn.Dropout(self.dropout),  # Aplica regularizacion adicional antes de la capa final
            nn.Linear(64, 1),  # Produce una sola prediccion continua de stormflow en el horizonte
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:  # Valida forma esperada (batch, seq_length, n_features)
            raise ValueError("Input tensor must have shape (batch, seq_length, n_features)")  # Mensaje claro para depurar entradas mal formadas

        x = x.transpose(1, 2)  # Reordena a (batch, n_features, seq_length) para Conv1d de PyTorch
        x = self.input_projection(x)  # Proyecta features de entrada al espacio de canales de la TCN
        x = self.tcn_blocks(x)  # Procesa secuencia con bloques residuales causales multiescala

        last_state = x[:, :, -1]  # Conserva el ultimo timestep causal como resumen del estado mas reciente
        stormflow_prediction = self.regression_head(last_state)  # Genera prediccion escalar directa sin cabeza de evento ni gating
        return stormflow_prediction  # Devuelve tensor (batch, 1) para entrenamiento de regresion directa

    def compute_receptive_field(self, print_result: bool = True) -> int:
        receptive_field = 1  # Inicializa campo receptivo en 1 para el timestep actual
        for dilation in self.dilations:  # Recorre cada bloque para acumular cobertura temporal total
            receptive_field += 2 * (self.kernel_size - 1) * dilation  # Suma aporte de dos convoluciones por bloque
        if print_result:  # Permite imprimir o solo devolver el valor segun necesidad del usuario
            print(f"[tcn] Campo receptivo: {receptive_field} timesteps")  # Reporta campo receptivo total en pasos de tiempo
        return receptive_field  # Devuelve cobertura temporal efectiva del modelo

    def count_parameters(self, print_result: bool = True) -> int:
        trainable_params = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)  # Cuenta parametros entrenables para estimar complejidad
        if print_result:  # Controla si se imprime diagnostico o solo se retorna el valor
            print(f"[tcn] Parametros entrenables: {trainable_params:,}")  # Muestra total de parametros con separador para legibilidad
        return trainable_params  # Retorna cantidad total de parametros entrenables