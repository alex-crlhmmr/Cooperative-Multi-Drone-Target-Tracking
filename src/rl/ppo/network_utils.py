import torch
from typing import List, Optional

def build_mlp(
    input_dim: int,
    hidden_sizes: List[int],
    output_dim: Optional[int],
    activation: str = "relu",
    output_activation: Optional[str] = None,
    layer_norm: bool = False,
    dropout: Optional[float] = None,
) -> torch.nn.Sequential:
    """Build a feedforward neural network (MLP) with the specified architecture.
    Args:
        input_dim (int): The dimension of the input features.
        hidden_sizes (List[int]): A list of integers specifying the number of units in each hidden layer.
        output_dim (Optional[int]): The dimension of the output layer. If None, no output layer is added.
        activation (str): The activation function to use for the hidden layers. Default is "relu".
        output_activation (Optional[str]): The activation function to use for the output layer. If None, no activation is applied to the output layer.
        layer_norm (bool): Whether to apply layer normalization after each hidden layer. Default is False.
        dropout (Optional[float]): The dropout rate to apply after each hidden layer. If None, no dropout is applied.
    Returns:
        torch.nn.Sequential: A PyTorch Sequential model representing the MLP.
    """
    layers = []
    prev_dim = input_dim

    for hidden_size in hidden_sizes:
        layers.append(torch.nn.Linear(prev_dim, hidden_size))
        if layer_norm:
            layers.append(torch.nn.LayerNorm(hidden_size))
        if activation.lower() == "relu":
            layers.append(torch.nn.ReLU())
        elif activation.lower() == "tanh":
            layers.append(torch.nn.Tanh())
        elif activation.lower() == "sigmoid":
            layers.append(torch.nn.Sigmoid())
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        if dropout is not None:
            layers.append(torch.nn.Dropout(dropout))
        prev_dim = hidden_size

    if output_dim is not None:
        layers.append(torch.nn.Linear(prev_dim, output_dim))
        if output_activation is not None:
            if output_activation.lower() == "relu":
                layers.append(torch.nn.ReLU())
            elif output_activation.lower() == "tanh":
                layers.append(torch.nn.Tanh())
            elif output_activation.lower() == "sigmoid":
                layers.append(torch.nn.Sigmoid())
            else:
                raise ValueError(f"Unsupported output activation function: {output_activation}")

    return torch.nn.Sequential(*layers)



