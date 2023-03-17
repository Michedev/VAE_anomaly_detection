import argparse
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import yaml
from path import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from model.VAE import VAEAnomalyTabular
from dataset import rand_dataset

ROOT = Path(__file__).parent
SAVED_MODELS = ROOT / 'saved_models'



def get_folder_run() -> Path:
    """
    Get the folder where to store the experiment
    
    Returns:
        Path: the path to the folder
    """
    checkpoint_folder = SAVED_MODELS / datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    checkpoint_folder.makedirs_p()
    return checkpoint_folder




def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', '-i', type=int, required=True, dest='input_size', help='Number of input features. In 1D case it is the vector length, in 2D case it is the number of channels')
    parser.add_argument('--latent-size', '-l', type=int, required=True, dest='latent_size', help='Size of the latent space')
    parser.add_argument('--num-resamples', '-L', type=int, dest='num_resamples', default=10,
                        help='Number of resamples in the latent distribution during training')
    parser.add_argument('--epochs', '-e', type=int, dest='epochs', default=100, help='Number of epochs to train for')
    parser.add_argument('--batch-size', '-b', type=int, dest='batch_size', default=32)
    parser.add_argument('--device', '-d', '--accelerator', type=str, dest='device', default='gpu', help='Device to use for training. Can be cpu, gpu or tpu', choices=['cpu', 'gpu', 'tpu'])
    parser.add_argument('--lr', type=float, dest='lr', default=1e-3, help='Learning rate')
    parser.add_argument('--no-progress-bar', action='store_true', dest='no_progress_bar')
    parser.add_argument('--steps-log-loss', type=int, dest='steps_log_loss', default=1_000, help='Number of steps between each loss logging')
    parser.add_argument('--steps-log-norm-params', type=int, 
                        dest='steps_log_norm_params', default=1_000, help='Number of steps between each model parameters logging')

    return parser.parse_args()

def main():
    args = get_args()
    print(args)
    experiment_folder = get_folder_run()

    # copy model folder into experiment folder
    ROOT.joinpath('model').copytree(experiment_folder / 'model')

    with open(experiment_folder / 'config.yaml', 'w') as f:
        yaml.dump(args, f)

    model = VAEAnomalyTabular(args.input_size, args.latent_size, args.num_resamples, lr=args.lr)

    train_set = rand_dataset()  # set here your dataset
    train_dloader = DataLoader(train_set, args.batch_size)

    val_dataset = rand_dataset()  # set here your dataset
    val_dloader = DataLoader(val_dataset, args.batch_size)

    checkpoint = ModelCheckpoint(
        filepath=experiment_folder / '{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix=''
        save_last=True,
    )

    trainer = Trainer(callbacks=[checkpoint],)
    trainer.fit(model, train_dloader, val_dloader)


if __name__ == '__main__':
    main()