import argparse

import torch
import yaml
from ignite.engine import Engine, Events
from ignite.metrics import Average, RunningAverage
from path import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from ignite.contrib.handlers.tqdm_logger import ProgressBar

from vae_anomaly_detection.VAE import VAEAnomalyTabular
from vae_anomaly_detection.dataset import rand_dataset

EXPERIMENT_FOLDER = Path(__file__).parent.parent


def get_folder_run() -> Path:
    run_path: Path = EXPERIMENT_FOLDER / 'run'
    if not run_path.exists(): run_path.mkdir()
    i = 0
    while (run_path / str(i)).exists():
        i += 1
    folder_run = run_path / str(i)
    folder_run.mkdir()
    return folder_run


class TrainStep:

    def __init__(self, model, opt, device=None):
        self.model = model
        self.opt = opt
        self.device = device

    def __call__(self, engine, batch):
        x = batch[0]
        if self.device: 
            x = x.to(self.device)
        pred_output = self.model(x)
        pred_output['loss'].backward()  # loss function is computed inside the VAE class since it is unsupervised
        self.opt.step()
        return pred_output


def train(model, opt, dloader, epochs: int, experiment_folder, device, progress_bar=True, steps_log_loss=1_000, steps_log_norm_params=10_000):
    step = TrainStep(model, opt, device)
    trainer = Engine(step)

    Average(lambda o: o['loss']).attach(trainer, 'avg_loss')
    RunningAverage(output_transform=lambda o: o['loss']).attach(trainer, 'running_avg_loss')

    if progress_bar:
        ProgressBar().attach(trainer, ['running_avg_loss'])

    setup_logger(experiment_folder, trainer, model, 
                 steps_log_loss, steps_log_norm_params)

    trainer.add_event_handler(Events.EPOCH_COMPLETED,
                              lambda e: torch.save(model.state_dict(), experiment_folder / 'model.pth'))

    trainer.run(dloader, epochs)


def setup_logger(experiment_folder, trainer, model, freq_loss: int = 1_000, freq_norm_params: int = 1_000):
    logger = SummaryWriter(log_dir=experiment_folder)
    for l in ['loss', 'kl', 'recon_loss']:
        event_handler = lambda e, l=l: logger.add_scalar(f'train/{l}', e.state.output[l], e.state.iteration)
        trainer.add_event_handler(Events.ITERATION_COMPLETED(every=freq_loss), event_handler)

    def log_norm(engine, logger, model=model):
        norm1 = sum(p.norm(1) for p in model.parameters())
        norm1_grad = sum(p.grad.norm(1) for p in model.parameters() if p.grad is not None)
        it = engine.state.iteration
        logger.add_scalar('train/norm1_params', norm1, it)
        logger.add_scalar('train/norm1_grad', norm1_grad, it)

    trainer.add_event_handler(Events.ITERATION_COMPLETED(every=freq_norm_params), log_norm, logger=logger)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', '-i', type=int, required=True, dest='input_size')
    parser.add_argument('--latent-size', '-l', type=int, required=True, dest='latent_size')
    parser.add_argument('--num-resamples', '-L', type=int, dest='num_resamples', default=10,
                        help='Number of resamples in the latent distribution during training')
    parser.add_argument('--epochs', '-e', type=int, dest='epochs', default=100)
    parser.add_argument('--batch-size', '-b', type=int, dest='batch_size', default=32)
    parser.add_argument('--device', '-d', type=str, dest='device', default='cuda:0')
    parser.add_argument('--lr', type=float, dest='lr', default=1e-4)
    parser.add_argument('--no-progress-bar', action='store_true', dest='no_progress_bar')
    parser.add_argument('--steps-log-loss', type=int, dest='steps_log_loss', default=1_000)
    parser.add_argument('--steps-log-norm-params', type=int, 
                        dest='steps_log_norm_params', default=1_000)

    return parser.parse_args()


def store_codebase_into_experiment(experiment_folder):
    with open(Path(__file__).parent / 'VAE.py') as f:
        code = f.read()
    with open(experiment_folder / 'vae.py', 'w') as f:
        f.write(code)


if __name__ == '__main__':
    args = get_args()
    print(args)
    experiment_folder = get_folder_run()
    model = VAEAnomalyTabular(args.input_size, args.latent_size, args.num_resamples).to(args.device)
    opt = torch.optim.Adam(model.parameters(), args.lr)
    dloader = DataLoader(rand_dataset(), args.batch_size)

    store_codebase_into_experiment(experiment_folder)
    with open(experiment_folder / 'config.yaml', 'w') as f:
        yaml.dump(args, f)

    train(model, opt, dloader, args.epochs, experiment_folder, args.device, not args.no_progress_bar,
          args.steps_log_loss, args.steps_log_norm_params)
