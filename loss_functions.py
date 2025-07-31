# loss_functions.py

import logging
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import Dataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants and configuration
CONFIG = {
    "LOSS_FUNCTIONS": {
        "MARGIN_LINE_LOSS": "margin_line_loss",
        "VELOCITY_THRESHOLD_LOSS": "velocity_threshold_loss",
        "FLOW_THEORY_LOSS": "flow_theory_loss",
    },
}

class MarginLineLoss(Module):
    """
    Custom loss function for margin line generation.
    """

    def __init__(self):
        super().__init__()

    def forward(self, predicted: Tensor, target: Tensor) -> Tensor:
        """
        Compute the margin line loss.

        Args:
            predicted (Tensor): Predicted margin line.
            target (Tensor): Ground truth margin line.

        Returns:
            Tensor: Margin line loss.
        """
        # Compute the difference between predicted and target margin lines
        diff = predicted - target

        # Compute the L1 loss
        loss = torch.mean(torch.abs(diff))

        return loss

class VelocityThresholdLoss(Module):
    """
    Custom loss function based on velocity threshold.
    """

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold

    def forward(self, predicted: Tensor, target: Tensor) -> Tensor:
        """
        Compute the velocity threshold loss.

        Args:
            predicted (Tensor): Predicted velocity.
            target (Tensor): Ground truth velocity.

        Returns:
            Tensor: Velocity threshold loss.
        """
        # Compute the difference between predicted and target velocity
        diff = predicted - target

        # Compute the L1 loss
        loss = torch.mean(torch.abs(diff))

        # Apply the velocity threshold
        loss = loss * (diff > self.threshold).float()

        return loss

class FlowTheoryLoss(Module):
    """
    Custom loss function based on flow theory.
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, predicted: Tensor, target: Tensor) -> Tensor:
        """
        Compute the flow theory loss.

        Args:
            predicted (Tensor): Predicted flow.
            target (Tensor): Ground truth flow.

        Returns:
            Tensor: Flow theory loss.
        """
        # Compute the difference between predicted and target flow
        diff = predicted - target

        # Compute the L1 loss
        loss = torch.mean(torch.abs(diff))

        # Apply the flow theory weights
        loss = self.alpha * loss + self.beta * (diff ** 2)

        return loss

class CustomLoss(Module):
    """
    Custom loss function that combines multiple loss functions.
    """

    def __init__(self, margin_line_loss_weight: float = 0.5, velocity_threshold_loss_weight: float = 0.3, flow_theory_loss_weight: float = 0.2):
        super().__init__()
        self.margin_line_loss_weight = margin_line_loss_weight
        self.velocity_threshold_loss_weight = velocity_threshold_loss_weight
        self.flow_theory_loss_weight = flow_theory_loss_weight
        self.margin_line_loss = MarginLineLoss()
        self.velocity_threshold_loss = VelocityThresholdLoss()
        self.flow_theory_loss = FlowTheoryLoss()

    def forward(self, predicted: Tensor, target: Tensor) -> Tensor:
        """
        Compute the custom loss.

        Args:
            predicted (Tensor): Predicted output.
            target (Tensor): Ground truth output.

        Returns:
            Tensor: Custom loss.
        """
        # Compute the margin line loss
        margin_line_loss = self.margin_line_loss(predicted, target)

        # Compute the velocity threshold loss
        velocity_threshold_loss = self.velocity_threshold_loss(predicted, target)

        # Compute the flow theory loss
        flow_theory_loss = self.flow_theory_loss(predicted, target)

        # Combine the losses
        loss = self.margin_line_loss_weight * margin_line_loss + self.velocity_threshold_loss_weight * velocity_threshold_loss + self.flow_theory_loss_weight * flow_theory_loss

        return loss

def get_loss_function(loss_name: str) -> Module:
    """
    Get a custom loss function by name.

    Args:
        loss_name (str): Name of the loss function.

    Returns:
        Module: Custom loss function.
    """
    if loss_name == CONFIG["LOSS_FUNCTIONS"]["MARGIN_LINE_LOSS"]:
        return MarginLineLoss()
    elif loss_name == CONFIG["LOSS_FUNCTIONS"]["VELOCITY_THRESHOLD_LOSS"]:
        return VelocityThresholdLoss()
    elif loss_name == CONFIG["LOSS_FUNCTIONS"]["FLOW_THEORY_LOSS"]:
        return FlowTheoryLoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")

def get_custom_loss(loss_weights: Dict[str, float]) -> Module:
    """
    Get a custom loss function with weighted losses.

    Args:
        loss_weights (Dict[str, float]): Weights for each loss function.

    Returns:
        Module: Custom loss function.
    """
    custom_loss = CustomLoss()
    for loss_name, weight in loss_weights.items():
        if loss_name in CONFIG["LOSS_FUNCTIONS"]:
            setattr(custom_loss, f"{loss_name}_loss", get_loss_function(loss_name))
            setattr(custom_loss, f"{loss_name}_loss_weight", weight)
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")
    return custom_loss