# Variational autoencoder for anomaly detection

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/vae-anomaly-detection?style=flat-square)
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/Michedev/VAE_anomaly_detection/Python%20test?style=flat-square)

Pytorch/TF1 implementation of Variational AutoEncoder for anomaly detection following the paper
 [Variational Autoencoder based Anomaly Detection using Reconstruction Probability by Jinwon An, Sungzoon Cho](https://www.semanticscholar.org/paper/Variational-Autoencoder-based-Anomaly-Detection-An-Cho/061146b1d7938d7a8dae70e3531a00fceb3c78e8)
 <br>

## How to install
1. _pip_ package containing the model and training_step only
`pip install vae-anomaly-detection`

1. Use repository
   1. Clone the repo

`git clone git@github.com:Michedev/VAE_anomaly_detection.git`

   2. Install anaconda and install anaconda-project package if you use miniconda

   `conda install anaconda-project`

   3. Install the environment

   `anaconda-project prepare`

   4. Run the train

   `anaconda-project run train`

   To know all the train parameters run `anaconda-project run train --help`




This version contains the model and the training procedure

## How To Train your Model

- Define your dataset into dataset.py and overwrite it in `train.py`
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
Once the model is trained (suppose for simplicity that it is under _run/0/_ ) just load and predict with this code snippet:
```python
import torch

#load X_test
model = VaeAnomalyTabular(input_size=50, latent_size=32)
# could load input_size and latent_size also 
# from run/0/train_config.yaml
model.load_state_dict(torch.load('run/0/model.pth'))
# load saved parameters from a run
outliers = model.is_anomaly(X_test)
```
