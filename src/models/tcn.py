"""TCN model definitions for stormflow prediction."""

from __future__ import annotations  # Allows modern type annotations without version issues

from typing import List, Sequence  # Defines types for channel and dilation lists

import torch  # Provides tensors and base operations for the model
from torch import nn  # Includes neural network modules in PyTorch


class CausalConv1d(nn.Module):
    """1D causal convolution implemented with left padding and right trimming."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()  # Properly initializes the base nn.Module class
        self.causal_padding = (kernel_size - 1) * dilation  # Computes total padding needed to preserve causality
        self.conv = nn.Conv1d(  # Defines 1D convolution with dilation to expand receptive field
            in_channels=in_channels,  # Defines number of input channels for the block
            out_channels=out_channels,  # Defines number of output channels for the block
            kernel_size=kernel_size,  # Defines temporal kernel size
            dilation=dilation,  # Defines spacing between kernel elements to cover more history
            padding=self.causal_padding,  # Applies symmetric padding and then trims to keep only past information
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)  # Runs temporal convolution over the sequence
        if self.causal_padding > 0:  # Checks whether trimming is needed to preserve length and causality
            x = x[:, :, :-self.causal_padding]  # Removes future positions introduced by right padding
        return x  # Returns causal output with the same temporal length as the input


def _build_group_norm(num_channels: int) -> nn.GroupNorm:
    """Build GroupNorm with a valid number of groups for the channel count."""
    candidate_groups = [8, 4, 2, 1]  # Tries common groups to keep normalization stable without depending on batch size
    for num_groups in candidate_groups:  # Iterates through possible groups until it finds one compatible with the current channels
        if num_channels % num_groups == 0:  # Checks exact divisibility required by GroupNorm
            return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)  # Returns stable normalization for the current width
    return nn.GroupNorm(num_groups=1, num_channels=num_channels)  # Safe fallback equivalent to channel-wise LayerNorm


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
        super().__init__()  # Initializes base structure of the residual block
        self.conv1 = CausalConv1d(  # First dilated causal convolution in the block
            in_channels=in_channels,  # Receives input channels for the block
            out_channels=out_channels,  # Projects to the block target channel count
            kernel_size=kernel_size,  # Uses kernel defined for the whole architecture
            dilation=dilation,  # Uses this block's dilation to cover a different temporal scale
        )
        self.norm1 = _build_group_norm(out_channels)  # Normalizes by groups to avoid drift with unrepresentative batches
        self.relu1 = nn.ReLU()  # Introduces nonlinearity after the first convolution
        self.drop1 = nn.Dropout(dropout)  # Regularizes activations to reduce overfitting

        self.conv2 = CausalConv1d(  # Second dilated causal convolution in the block
            in_channels=out_channels,  # Uses intermediate output as the new block input
            out_channels=out_channels,  # Keeps dimensionality so it can be added to the skip connection
            kernel_size=kernel_size,  # Repeats block kernel size
            dilation=dilation,  # Repeats dilation for multiscale consistency inside the block
        )
        self.norm2 = _build_group_norm(out_channels)  # Repeats GroupNorm to stabilize the second block transformation
        self.relu2 = nn.ReLU()  # Applies nonlinearity in the second transformation
        self.drop2 = nn.Dropout(dropout)  # Applies additional regularization in the second layer

        if in_channels != out_channels:  # Checks whether the residual needs channel adjustment
            self.skip_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)  # Projects residual with Conv1x1 when channels change
        else:  # Uses identity shortcut when dimensions already match
            self.skip_proj = nn.Identity()  # Avoids extra cost if no projection is needed

        self.out_relu = nn.ReLU()  # Activates combined residual output to maintain stability and final nonlinearity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip_proj(x)  # Builds residual branch aligned in channels with the main output

        out = self.conv1(x)  # Runs first causal convolution
        out = self.norm1(out)  # Normalizes first convolution output without depending on batch statistics
        out = self.relu1(out)  # Activates intermediate output to model nonlinear relationships
        out = self.drop1(out)  # Applies dropout to intermediate activations

        out = self.conv2(out)  # Runs second causal convolution in the block
        out = self.norm2(out)  # Normalizes second convolution output with GroupNorm
        out = self.relu2(out)  # Activates second intermediate output
        out = self.drop2(out)  # Applies final dropout before residual addition

        out = out + residual  # Adds main and residual branches to ease gradient flow
        out = self.out_relu(out)  # Applies final activation after residual fusion
        return out  # Returns block output with the same temporal length


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
        super().__init__()  # Initializes the base class to register submodules correctly
        self.n_features = n_features  # Stores number of input features for reference and debugging
        self.num_channels = list(num_channels) if num_channels is not None else [32, 64, 64, 64, 32]  # Defines channels per block according to the proposal
        self.dilations = list(dilations) if dilations is not None else [1, 2, 4, 8, 16]  # Defines dilations per block according to the proposal
        self.kernel_size = kernel_size  # Stores temporal kernel size for receptive-field methods
        self.dropout = dropout  # Stores global dropout for blocks and regression head

        if len(self.num_channels) != len(self.dilations):  # Verifies consistency between number of blocks and dilations
            raise ValueError("num_channels and dilations must have the same length")  # Raises a clear error if configuration is inconsistent

        self.input_projection = nn.Conv1d(  # Projects input features into initial TCN channels
            in_channels=n_features,  # Receives number of features per timestamp
            out_channels=self.num_channels[0],  # Maps to the width of the first residual block
            kernel_size=1,  # Uses Conv1x1 to mix features without altering temporal length
        )

        blocks: List[nn.Module] = []  # Accumulates residual blocks to build the temporal network
        in_channels = self.num_channels[0]  # Initializes input channels of the first block
        for out_channels, dilation in zip(self.num_channels, self.dilations):  # Iterates through channels and dilations defined per block
            block = TCNResidualBlock(  # Creates residual causal block for the current temporal scale
                in_channels=in_channels,  # Uses current temporal representation width
                out_channels=out_channels,  # Configures block output width
                kernel_size=self.kernel_size,  # Uses kernel shared across the whole architecture
                dilation=dilation,  # Uses block-specific dilation
                dropout=self.dropout,  # Uses model-level dropout
            )
            blocks.append(block)  # Adds the created block to the sequential list
            in_channels = out_channels  # Updates input channels for the next block
        self.tcn_blocks = nn.Sequential(*blocks)  # Packs blocks into an executable sequence

        final_channels = self.num_channels[-1]  # Gets final channels after the last TCN block
        self.regression_head = nn.Sequential(  # Defines final MLP to map causal state to scalar prediction
            nn.Linear(final_channels, 128),  # Projects final state into a larger intermediate space
            nn.ReLU(),  # Introduces nonlinearity to learn complex hydrologic relationships
            nn.Dropout(self.dropout),  # Regularizes intermediate activations to reduce overfitting
            nn.Linear(128, 64),  # Reduces dimensionality to stabilize the final regression stage
            nn.ReLU(),  # Introduces a second nonlinearity before output
            nn.Dropout(self.dropout),  # Applies additional regularization before the final layer
            nn.Linear(64, 1),  # Produces a single continuous stormflow prediction at the horizon
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:  # Validates expected shape (batch, seq_length, n_features)
            raise ValueError("Input tensor must have shape (batch, seq_length, n_features)")  # Clear message to debug malformed inputs

        x = x.transpose(1, 2)  # Reorders to (batch, n_features, seq_length) for PyTorch Conv1d
        x = self.input_projection(x)  # Projects input features into the TCN channel space
        x = self.tcn_blocks(x)  # Processes sequence with multiscale causal residual blocks

        last_state = x[:, :, -1]  # Keeps the last causal timestep as a summary of the most recent state
        stormflow_prediction = self.regression_head(last_state)  # Generates direct scalar prediction without event head or gating
        return stormflow_prediction  # Returns tensor (batch, 1) for direct regression training

    def compute_receptive_field(self, print_result: bool = True) -> int:
        receptive_field = 1  # Initializes receptive field at 1 for the current timestep
        for dilation in self.dilations:  # Iterates through each block to accumulate total temporal coverage
            receptive_field += 2 * (self.kernel_size - 1) * dilation  # Adds contribution of two convolutions per block
        if print_result:  # Lets the caller print or only return the value as needed
            print(f"[tcn] Receptive field: {receptive_field} timesteps")  # Reports total receptive field in timesteps
        return receptive_field  # Returns effective temporal coverage of the model

    def count_parameters(self, print_result: bool = True) -> int:
        trainable_params = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)  # Counts trainable parameters to estimate complexity
        if print_result:  # Controls whether to print a diagnostic or only return the value
            print(f"[tcn] Trainable parameters: {trainable_params:,}")  # Shows total parameter count with separator for readability
        return trainable_params  # Returns total number of trainable parameters


class TwoStageTCN(nn.Module):
    """Two-stage model with shared backbone and two heads."""

    def __init__(
        self,
        n_features: int,
        num_channels: Sequence[int] | None = None,
        dilations: Sequence[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()  # Initializes the base class to register submodules correctly
        self.n_features = n_features  # Stores number of input features for reference and debugging
        self.num_channels = list(num_channels) if num_channels is not None else [32, 64, 64, 64, 32]  # Defines channels per block according to the proposal
        self.dilations = list(dilations) if dilations is not None else [1, 2, 4, 8, 16]  # Defines dilations per block according to the proposal
        self.kernel_size = kernel_size  # Stores temporal kernel size for receptive-field methods
        self.dropout = dropout  # Stores global dropout for blocks and model heads

        if len(self.num_channels) != len(self.dilations):  # Verifies consistency between number of blocks and dilations
            raise ValueError("num_channels and dilations must have the same length")  # Raises a clear error if configuration is inconsistent

        self.input_projection = nn.Conv1d(  # Projects input features into initial TCN channels
            in_channels=n_features,  # Receives number of features per timestamp
            out_channels=self.num_channels[0],  # Maps to the width of the first residual block
            kernel_size=1,  # Uses Conv1x1 to mix features without altering temporal length
        )

        blocks: List[nn.Module] = []  # Accumulates residual blocks to build the shared temporal network
        in_channels = self.num_channels[0]  # Initializes input channels of the first block
        for out_channels, dilation in zip(self.num_channels, self.dilations):  # Iterates through channels and dilations defined per block
            block = TCNResidualBlock(  # Creates residual causal block for the current temporal scale
                in_channels=in_channels,  # Uses current temporal representation width
                out_channels=out_channels,  # Configures block output width
                kernel_size=self.kernel_size,  # Uses kernel shared across the whole architecture
                dilation=dilation,  # Uses block-specific dilation
                dropout=self.dropout,  # Uses model-level dropout
            )
            blocks.append(block)  # Adds the created block to the sequential list
            in_channels = out_channels  # Updates input channels for the next block
        self.tcn_blocks = nn.Sequential(*blocks)  # Packs blocks into an executable sequence

        final_channels = self.num_channels[-1]  # Gets final channels after the last TCN block
        self.classifier_head = nn.Sequential(  # Defines binary head to detect event presence
            nn.Linear(final_channels, 64),  # Expands causal state for a more stable binary decision
            nn.ReLU(),  # Introduces nonlinearity to separate events from non-events
            nn.Dropout(self.dropout),  # Regularizes the head to reduce overfitting on rare classes
            nn.Linear(64, 1),  # Projects to one scalar logit/score per sample
            nn.Sigmoid(),  # Converts score into event probability in [0, 1]
        )
        self.regressor_head = nn.Sequential(  # Defines regression head for stormflow magnitude
            nn.Linear(final_channels, 128),  # Increases capacity to model extreme magnitudes
            nn.ReLU(),  # Introduces nonlinearity to capture the rainfall->magnitude relationship
            nn.Dropout(self.dropout),  # Regularizes activations to avoid overfitting
            nn.Linear(128, 64),  # Reduces dimensionality while keeping enough capacity
            nn.ReLU(),  # Applies additional nonlinearity before output
            nn.Linear(64, 1),  # Produces a single continuous stormflow prediction
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 3:  # Validates expected shape (batch, seq_length, n_features)
            raise ValueError("Input tensor must have shape (batch, seq_length, n_features)")  # Clear message to debug malformed inputs

        x = x.transpose(1, 2)  # Reorders to (batch, n_features, seq_length) for PyTorch Conv1d
        x = self.input_projection(x)  # Projects input features into the TCN channel space
        x = self.tcn_blocks(x)  # Processes sequence with multiscale causal residual blocks

        last_state = x[:, :, -1]  # Keeps the last causal timestep as a summary of the most recent state
        cls_prob = self.classifier_head(last_state)  # Produces event probability for the batch
        reg_value = self.regressor_head(last_state)  # Produces continuous stormflow magnitude for the batch
        return {"cls_prob": cls_prob, "reg_value": reg_value}  # Returns dictionary with both outputs for the loss

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        outputs = self.forward(x)  # Runs forward to obtain probability and magnitude
        cls_prob = outputs["cls_prob"]  # Extracts event probability to apply hard switch
        reg_value = outputs["reg_value"]  # Extracts magnitude predicted by the regressor
        zeros = torch.zeros_like(reg_value)  # Creates zero tensor for non-event cases
        return torch.where(cls_prob >= threshold, reg_value, zeros)  # Applies hard switch without multiplicative gating

    def compute_receptive_field(self, print_result: bool = True) -> int:
        receptive_field = 1  # Initializes receptive field at 1 for the current timestep
        for dilation in self.dilations:  # Iterates through each block to accumulate total temporal coverage
            receptive_field += 2 * (self.kernel_size - 1) * dilation  # Adds contribution of two convolutions per block
        if print_result:  # Lets the caller print or only return the value as needed
            print(f"[tcn] Receptive field: {receptive_field} timesteps")  # Reports total receptive field in timesteps
        return receptive_field  # Returns effective temporal coverage of the model

    def count_parameters(self, print_result: bool = True) -> int:
        trainable_params = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)  # Counts trainable parameters to estimate complexity
        if print_result:  # Controls whether to print a diagnostic or only return the value
            print(f"[tcn] Trainable parameters: {trainable_params:,}")  # Shows total parameter count with separator for readability
        return trainable_params  # Returns total number of trainable parameters
