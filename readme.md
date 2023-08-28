# Variational autoencoder for anomaly detection

![PyPI](https://img.shields.io/pypi/v/vae-anomaly-detection?style=flat-square)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/vae-anomaly-detection?style=flat-square)
![PyPI - License](https://img.shields.io/pypi/l/vae-anomaly-detection?style=flat-square)
![PyPI - Downloads](https://img.shields.io/pypi/dm/vae-anomaly-detection?style=flat-square)

Pytorch/TF1 implementation of Variational AutoEncoder for anomaly detection following the paper
 [Variational Autoencoder based Anomaly Detection using Reconstruction Probability by Jinwon An, Sungzoon Cho](https://www.semanticscholar.org/paper/Variational-Autoencoder-based-Anomaly-Detection-An-Cho/061146b1d7938d7a8dae70e3531a00fceb3c78e8)
 <br>

## How to install

#### Python package way
 _pip_ package containing the model and training_step only 
   
    pip install vae-anomaly-detection


#### Hack this repository


   a. Clone the repo

    git clone git@github.com:Michedev/VAE_anomaly_detection.git

   b. Install hatch

    pip install hatch

   c. Make the environment with torch gpu support

    hatch env create
      
or with cpu support

    hatch env create cpu

   d. Run the train

    hatch run train

or in cpu
          
    hatch run cpu:train

   To know all the train parameters run `hatch run train --help`




This version contains the model and the training procedure

## How To Train your Model

- Define your dataset into dataset.py and overwrite the line `train_set = rand_dataset()  # set here your dataset` in `train.py`
- Subclass VAEAnomalyDetection and define the methods `make_encoder` and `make_decoder`. The output of `make_encoder` should be a flat vector while the output of `make_decoder should have the same shape of the input.
## Make your model

Subclass ```VAEAnomalyDetection``` and define your encoder and decoder like in ```VaeAnomalyTabular```

```python
class VAEAnomalyTabular(VAEAnomalyDetection):

    def make_encoder(self, input_size, latent_size):
        """
        Simple encoder for tabular data.
        If you want to feed image to a VAE make another encoder function with Conv2d instead of Linear layers.
        :param input_size: number of input variables
        :param latent_size: number of output variables i.e. the size of the latent space since it's the encoder of a VAE
        :return: The untrained encoder model
        """
        return nn.Sequential(
            nn.Linear(input_size, 500),
            nn.ReLU(),
            nn.Linear(500, 200),
            nn.ReLU(),
            nn.Linear(200, latent_size * 2)
            # times 2 because this is the concatenated vector of latent mean and variance
        )

    def make_decoder(self, latent_size, output_size):
        """
        Simple decoder for tabular data.
        :param latent_size: size of input latent space
        :param output_size: number of output parameters. Must have the same value of input_size
        :return: the untrained decoder
        """
        return nn.Sequential(
            nn.Linear(latent_size, 200),
            nn.ReLU(),
            nn.Linear(200, 500),
            nn.ReLU(),
            nn.Linear(500, output_size * 2)  # times 2 because this is the concatenated vector of reconstructed mean and variance
        )
```

## How to make predictions:
Once the model is trained (suppose for simplicity that it is under _saved_models/{train-datetime}/_ ) just load and predict with this code snippet:
```python
import torch

#load X_test
model = VaeAnomalyTabular.load_checkpoint('saved_models/2022-01-06_15-12-23/last.ckpt')
# load saved parameters from a run
outliers = model.is_anomaly(X_test)
```


## train.py help

        usage: train.py [-h] --input-size INPUT_SIZE --latent-size LATENT_SIZE
                        [--num-resamples NUM_RESAMPLES] [--epochs EPOCHS] [--batch-size BATCH_SIZE]
                        [--device {cpu,gpu,tpu}] [--lr LR] [--no-progress-bar]
                        [--steps-log-loss STEPS_LOG_LOSS]
                        [--steps-log-norm-params STEPS_LOG_NORM_PARAMS]

        options:
        -h, --help            show this help message and exit
        --input-size INPUT_SIZE, -i INPUT_SIZE
                                Number of input features. In 1D case it is the vector length, in 2D
                                case it is the number of channels
        --latent-size LATENT_SIZE, -l LATENT_SIZE
                                Size of the latent space
        --num-resamples NUM_RESAMPLES, -L NUM_RESAMPLES
                                Number of resamples in the latent distribution during training
        --epochs EPOCHS, -e EPOCHS
                                Number of epochs to train for
        --batch-size BATCH_SIZE, -b BATCH_SIZE
        --device {cpu,gpu,tpu}, -d {cpu,gpu,tpu}, --accelerator {cpu,gpu,tpu}
                                Device to use for training. Can be cpu, gpu or tpu
        --lr LR               Learning rate
        --no-progress-bar
        --steps-log-loss STEPS_LOG_LOSS
                                Number of steps between each loss logging
        --steps-log-norm-params STEPS_LOG_NORM_PARAMS
                                Number of steps between each model parameters logging
