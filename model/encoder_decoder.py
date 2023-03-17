"""
This module contains simple encoder and decoder for tabular data.
For your own data you need to create your own encoder and decoder.
However the input and output of your encoder and decoder must be the same of the ones in this module.
"""

from torch import nn

def tabular_encoder(input_size: int, latent_size: int):
    """
    Simple encoder for tabular data.
    If you want to feed image to a VAE make another encoder function with Conv2d instead of Linear layers.
    
    Parameters
    ----------
    input_size : int
        number of input variables. In case of tabular data it's the number of columns.
    latent_size : int
        number of output variables i.e. the size of the latent space since it's the encoder of a VAE

    Returns
    -------
    The untrained encoder model
    
    """
    return nn.Sequential(
        nn.Linear(input_size, 500),
        nn.ReLU(),
        nn.Linear(500, 200),
        nn.ReLU(),
        nn.Linear(200, latent_size * 2)  # times 2 because this is the concatenated vector of latent mean and variance
    )


def tabular_decoder(latent_size: int, output_size: int):
    """
    Simple decoder for tabular data.

    Parameters
    ----------
    latent_size : int
        size of input latent space
    output_size : int
        number of output parameters. Must have the same value of input_size of the encoder

    Returns
    -------
    The untrained decoder
    """
    return nn.Sequential(
        nn.Linear(latent_size, 200),
        nn.ReLU(),
        nn.Linear(200, 500),
        nn.ReLU(),
        nn.Linear(500, output_size * 2)
        # times 2 because this is the concatenated vector of reconstructed mean and variance
    )

