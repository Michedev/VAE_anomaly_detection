from path import Path
from torch.optim import Adam
from torch.utils.data import DataLoader

from vae_anomaly_detection.train import train as pytorch_train
from vae_anomaly_detection.VAE import VAEAnomaly as PytorchVAEAnomaly
from vae_anomaly_detection.dataset import rand_dataset as random_torch_dataset


def pytorch_single_step_training():
    batch_size = 32
    epochs = 1
    dataset = random_torch_dataset(batch_size, 100)
    dloader = DataLoader(dataset, batch_size=batch_size)
    model = PytorchVAEAnomaly(100, 32, L=2)
    opt = Adam(model.parameters())
    experiments_folder: Path = Path(__file__).parent / 'tmp_experiment'

    pytorch_train(model, opt, dloader, epochs, experiments_folder, 'cpu')
    experiment_folder = experiments_folder / '0'
    assert experiment_folder.exists()
    assert (experiment_folder / 'model.pth').exists()

    experiments_folder.rmtree()

