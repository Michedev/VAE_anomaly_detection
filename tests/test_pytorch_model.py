import torch
from model import VAEAnomalyTabular


def test_pytorch_anomaly_detection():
    batch_size = 32
    input_size = 100
    latent_size = 32
    model = VAEAnomalyTabular(input_size, latent_size, L=2)
    batch = torch.rand(batch_size, input_size)
    batch_anomaly = model.is_anomaly(batch, alpha=0.05)
    assert batch_anomaly.shape == (batch_size,)
    assert batch_anomaly.dtype == torch.bool


def test_pytorch_prediction():
    batch_size = 32
    input_size = 100
    latent_size = 32
    model = VAEAnomalyTabular(input_size, latent_size, L=2)
    batch = torch.rand(batch_size, input_size)
    reconstructed_probability = model.reconstructed_probability(batch)
    assert reconstructed_probability.shape == (batch_size,)
    assert reconstructed_probability.dtype == torch.float
    assert 1.0 >= reconstructed_probability.max().item() and \
           reconstructed_probability.min().item() >= 0.0
    

def test_training_step():
    batch_size = 32
    input_size = 100
    latent_size = 32
    model = VAEAnomalyTabular(input_size, latent_size, L=2)
    batch = torch.rand(batch_size, input_size)
    reconstructed_probability = model.training_step(batch)
    assert reconstructed_probability.shape == (batch_size,)
    assert reconstructed_probability.dtype == torch.float
    assert 1.0 >= reconstructed_probability.max().item() and \
           reconstructed_probability.min().item() >= 0.0
    
