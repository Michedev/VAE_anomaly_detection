import torch
from vae_anomaly_detection.VAE import VAEAnomaly as PytorchVAEAnomaly


def test_pytorch_anomaly_detection():
    batch_size = 32
    model = PytorchVAEAnomaly(100, 32, L=2)
    batch = torch.rand(batch_size, 100)
    batch_anomaly = model.is_anomaly(batch, alpha=0.05)
    assert batch_anomaly.shape == (32,)
    assert batch_anomaly.dtype == torch.bool


def test_pytorch_prediction():
    batch_size = 32
    model = PytorchVAEAnomaly(100, 32, L=2)
    batch = torch.rand(batch_size, 100)
    reconstructed_probability = model.reconstructed_probability(batch)
    assert reconstructed_probability.shape == (32,)
    assert reconstructed_probability.dtype == torch.float
    assert 1.0 >= reconstructed_probability.max().item() and \
           reconstructed_probability.min().item() >= 0.0