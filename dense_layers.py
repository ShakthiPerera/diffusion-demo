"""Conditional dense layers and utilities for time‑conditioned MLPs.

This module reimplements the conditional fully connected architecture from
the original diffusion codebase.  It includes sinusoidal and learnable
sinusoidal positional encodings, utility functions for creating dense and
convolutional layers, and classes for conditional dense layers and
conditional dense models.  The conditional dense model injects a
learnable time embedding into every linear layer, matching the
architecture used in the baseline implementation.
"""

import torch
import torch.nn as nn

def make_activation(mode):
    """Create an activation layer by name.

    Parameters
    ----------
    mode : str or None
        Name of the activation function.  If ``None`` or ``'none'`` then
        ``nn.Identity`` is returned.

    Returns
    -------
    nn.Module
        Instantiated activation module.
    """
    if mode is None or mode == 'none':
        activation = nn.Identity()
    elif mode == 'sigmoid':
        activation = nn.Sigmoid()
    elif mode == 'tanh':
        activation = nn.Tanh()
    elif mode == 'relu':
        activation = nn.ReLU()
    elif mode == 'leaky_relu':
        activation = nn.LeakyReLU()
    elif mode == 'elu':
        activation = nn.ELU()
    elif mode == 'softplus':
        activation = nn.Softplus()
    elif mode == 'swish':
        activation = nn.SiLU()
    else:
        raise ValueError(f'Unknown activation function: {mode}')
    return activation

def make_norm(mode, num_features):
    """Create a normalisation layer by name.

    Parameters
    ----------
    mode : str or None
        Normalisation type: ``'batch'`` for batch normalisation,
        ``'instance'`` for instance normalisation, or ``None`` for identity.
    num_features : int
        Number of features/channels for the normalisation layer.

    Returns
    -------
    nn.Module
        Instantiated normalisation module.
    """
    if mode is None or mode == 'none':
        norm = nn.Identity()
    elif mode == 'batch':
        norm = nn.BatchNorm2d(num_features)
    elif mode == 'instance':
        norm = nn.InstanceNorm2d(num_features)
    else:
        raise ValueError(f'Unknown normalization type: {mode}')
    return norm

def make_dense(in_features, out_features, bias=True, activation=None):
    """Create a fully connected layer optionally followed by an activation.

    This utility mirrors the ``make_dense`` function from the original
    codebase.  It returns a ``nn.Sequential`` containing a linear
    transformation and the requested activation.

    Parameters
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.
    bias : bool, optional
        Whether to include a bias term.  Default is ``True``.
    activation : str or None, optional
        Activation name.  If ``None`` or ``'none'`` then no activation is
        applied.

    Returns
    -------
    nn.Sequential
        A sequential block with linear and activation layers.
    """
    linear = nn.Linear(in_features, out_features, bias=bias)
    act = make_activation(activation)
    return nn.Sequential(linear, act)

def make_conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1,
              bias=True, norm=None, activation=None):
    """Create a convolutional layer optionally followed by activation and normalisation.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int, optional
        Size of the convolutional kernel.  Default is ``3``.
    stride : int, optional
        Stride for the convolution.  Default is ``1``.
    padding : int, optional
        Zero-padding added to both sides of the input.  Default is ``1``.
    bias : bool, optional
        Whether to include a bias term in the convolution.  Default is ``True``.
    norm : str or None, optional
        Normalisation type.  See ``make_norm``.
    activation : str or None, optional
        Activation function name.  See ``make_activation``.

    Returns
    -------
    nn.Sequential
        A sequential block with convolution, activation and normalisation.
    """
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                     stride=stride, padding=padding, bias=bias)
    act = make_activation(activation)
    norm_layer = make_norm(norm, num_features=out_channels)
    return nn.Sequential(conv, act, norm_layer)

class SinusoidalEncoding(nn.Module):
    """Sinusoidal position encoding as introduced in the Transformer paper.

    This produces deterministic embeddings based on sine and cosine
    functions of varying frequencies.  It is commonly used to encode
    positions or timesteps into a fixed-dimensional vector.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        embed_dim = abs(int(embed_dim))
        if embed_dim < 2:
            raise ValueError('At least two embedding dimensions required')
        if embed_dim % 2 != 0:
            raise ValueError('Dimensionality has to be an even number')
        self.embed_dim = embed_dim
        # precompute angular frequencies
        omega = self._make_frequencies()
        self.register_buffer('omega', omega)

    def _make_frequencies(self) -> torch.Tensor:
        """Create angular frequencies for the sinusoidal encoding."""
        i = torch.arange(self.embed_dim // 2).view(1, -1)
        omega = 1 / (10000 ** (2 * i / self.embed_dim))
        return omega

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Compute sinusoidal embedding for integer timesteps.

        Accepts timesteps as a tensor of shape ``(batch_size,)`` or
        ``(batch_size, 1)`` and returns an array of shape
        ``(batch_size, embed_dim)``.  The timesteps are treated as
        integer indices.
        """
        # reshape to (batch_size, 1) if necessary
        if t.ndim == 1:
            t = t.view(-1, 1)
        elif t.ndim == 2 and t.shape[1] == 1:
            t = t
        else:
            raise ValueError(f'Invalid shape encountered: {t.shape}')
        device = t.device
        batch_size = t.shape[0]
        # allocate embedding matrix
        emb = torch.zeros(batch_size, self.embed_dim, device=device)
        emb[:, 0::2] = torch.sin(self.omega * t)
        emb[:, 1::2] = torch.cos(self.omega * t)
        return emb

class LearnableSinusoidalEncoding(nn.Sequential):
    """Learnable sinusoidal position encoding.

    A sinusoidal encoding is followed by a stack of fully connected layers
    that can learn to adapt the raw sinusoidal representation to better
    suit the downstream task.  This matches the implementation in the
    original diffusion codebase.

    Parameters
    ----------
    num_features : list of int
        List specifying the dimensions of each layer in the learnable
        encoding.  The first entry is the input dimensionality (for the
        sinusoidal encoding), and the last entry is the output
        dimensionality.
    activation : str or None, optional
        Activation function to use between layers (except after the last
        layer).
    """
    def __init__(self, num_features, activation: str = 'relu') -> None:
        if len(num_features) < 2:
            raise ValueError('Number of features needs at least two entries')
        embed_dim = int(num_features[0])
        sinusoidal = SinusoidalEncoding(embed_dim=embed_dim)
        num_dense_layers = len(num_features) - 1
        dense_layers = []
        for idx, (in_features, out_features) in enumerate(zip(num_features[:-1], num_features[1:])):
            is_not_last = (idx < num_dense_layers - 1)
            dense = make_dense(in_features, out_features,
                               activation=activation if is_not_last else None)
            dense_layers.append(dense)
        super().__init__(sinusoidal, *dense_layers)

class ConditionalDense(nn.Module):
    """Conditional fully connected layer with time embedding.

    Each ``ConditionalDense`` layer applies a linear transformation to its
    input and adds a learnable positional embedding derived from the
    timestep ``t``.  An optional activation can be applied after the
    addition.
    """
    def __init__(self, in_features: int, out_features: int,
                 activation: str = 'relu', embed_dim: int = 64) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = make_activation(activation)
        if embed_dim is not None:
            # build learnable sinusoidal encoding followed by two dense layers
            # to match the original architecture
            self.emb = LearnableSinusoidalEncoding([
                embed_dim, out_features, out_features
            ], activation=activation)
        else:
            self.emb = None

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        if self.emb is not None:
            emb = self.emb(t)
            out = out + emb
        if self.activation is not None:
            out = self.activation(out)
        return out

class ConditionalDenseModel(nn.Module):
    """Stack of conditional dense layers for noise prediction.

    This model is defined by a list of feature dimensions.  Given
    ``num_features = [2, 128, 128, 128, 2]`` it will create three
    hidden layers of width 128 followed by an output layer of width 2.
    Each layer receives a learnable embedding of the timestep.
    """
    def __init__(self, num_features, activation: str = 'swish', embed_dim: int  = 64) -> None:
        super().__init__()
        if len(num_features) < 2:
            raise ValueError('Number of features needs at least two entries')
        num_layers = len(num_features) - 1
        dense_list = []
        for idx, (in_features, out_features) in enumerate(zip(num_features[:-1], num_features[1:])):
            is_not_last = (idx < num_layers - 1)
            layer = ConditionalDense(
                in_features,
                out_features,
                activation=activation if is_not_last else None,
                embed_dim=embed_dim
            )
            dense_list.append(layer)
        self.dense_layers = nn.ModuleList(dense_list)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        for layer in self.dense_layers:
            x = layer(x, t)
        return x