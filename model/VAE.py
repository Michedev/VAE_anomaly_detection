from abc import abstractmethod, ABC

import torch
from torch import nn
from torch.distributions import Normal, kl_divergence
from torch.nn.functional import softplus
import pytorch_lightning as pl


class VAEAnomalyDetection(pl.LightningModule, ABC):
    """
    Variational Autoencoder (VAE) for anomaly detection. The model learns a low-dimensional representation of the input
    data using an encoder-decoder architecture, and uses the learned representation to detect anomalies.

    The model is trained to minimize the Kullback-Leibler (KL) divergence between the learned distribution of the latent
    variables and the prior distribution (a standard normal distribution). It is also trained to maximize the likelihood
    of the input data under the learned distribution.

    This implementation uses PyTorch Lightning to simplify training and improve reproducibility.
    """

    def __init__(self, input_size: int, latent_size: int, L: int = 10, lr: float = 1e-3, log_steps: int = 1_000):
        """
        Initializes the VAEAnomalyDetection model.

        Args:
            input_size (int): Number of input features.
            latent_size (int): Size of the latent space.
            L (int, optional): Number of samples in the latent space to detect the anomaly. Defaults to 10.
            lr (float, optional): Learning rate. Defaults to 1e-3.
            log_steps (int, optional): Number of steps between each logging. Defaults to 1_000.
        """
        super().__init__()
        self.L = L
        self.lr = lr
        self.input_size = input_size
        self.latent_size = latent_size
        self.encoder = self.make_encoder(input_size, latent_size)
        self.decoder = self.make_decoder(latent_size, input_size)
        self.prior = Normal(0, 1)
        self.log_steps = log_steps

    @abstractmethod
    def make_encoder(self, input_size: int, latent_size: int) -> nn.Module:
        """
        Abstract method to create the encoder network.

        Args:
            input_size (int): Number of input features.
            latent_size (int): Size of the latent space.

        Returns:
            nn.Module: Encoder network.
        """
        pass

    @abstractmethod
    def make_decoder(self, latent_size: int, output_size: int) -> nn.Module:
        """
        Abstract method to create the decoder network.

        Args:
            latent_size (int): Size of the latent space.
            output_size (int): Number of output features.

        Returns:
            nn.Module: Decoder network.
        """
        pass

    def forward(self, x: torch.Tensor) -> dict:
        """
        Computes the forward pass of the model and returns the loss and other relevant information.

        Args:
            x (torch.Tensor): Input data. Shape [batch_size, num_features].

        Returns:
            Dictionary containing:
            - loss: Total loss.
            - kl: KL-divergence loss.
            - recon_loss: Reconstruction loss.
            - recon_mu: Mean of the reconstructed input.
            - recon_sigma: Standard deviation of the reconstructed input.
            - latent_dist: Distribution of the latent space.
            - latent_mu: Mean of the latent space.
            - latent_sigma: Standard deviation of the latent space.
            - z: Sampled latent space.

        """
        pred_result = self.predict(x)
        x = x.unsqueeze(0)  # unsqueeze to broadcast input across sample dimension (L)
        log_lik = Normal(pred_result['recon_mu'], pred_result['recon_sigma']).log_prob(x).mean(
            dim=0)  # average over sample dimension
        log_lik = log_lik.mean(dim=0).sum()
        kl = kl_divergence(pred_result['latent_dist'], self.prior).mean(dim=0).sum()
        loss = kl - log_lik
        return dict(loss=loss, kl=kl, recon_loss=log_lik, **pred_result)

    def predict(self, x) -> dict:
        """
        Compute the output of the VAE. Does not compute the loss compared to the forward method.

        Args:
            x: Input tensor of shape [batch_size, input_size].

        Returns:
            Dictionary containing:
            - latent_dist: Distribution of the latent space.
            - latent_mu: Mean of the latent space.
            - latent_sigma: Standard deviation of the latent space.
            - recon_mu: Mean of the reconstructed input.
            - recon_sigma: Standard deviation of the reconstructed input.
            - z: Sampled latent space.

        """
        batch_size = len(x)
        latent_mu, latent_sigma = self.encoder(x).chunk(2, dim=1) #both with size [batch_size, latent_size]
        latent_sigma = softplus(latent_sigma)
        dist = Normal(latent_mu, latent_sigma)
        z = dist.rsample([self.L])  # shape: [L, batch_size, latent_size]
        z = z.view(self.L * batch_size, self.latent_size)
        recon_mu, recon_sigma = self.decoder(z).chunk(2, dim=1)
        recon_sigma = softplus(recon_sigma)
        recon_mu = recon_mu.view(self.L, *x.shape)
        recon_sigma = recon_sigma.view(self.L, *x.shape)
        return dict(latent_dist=dist, latent_mu=latent_mu,
                    latent_sigma=latent_sigma, recon_mu=recon_mu,
                    recon_sigma=recon_sigma, z=z)

    def is_anomaly(self, x: torch.Tensor, alpha: float = 0.05) -> torch.Tensor:
        """
        Determines if input samples are anomalous based on a given threshold.
        
        Args:
            x: Input tensor of shape (batch_size, num_features).
            alpha: Anomaly threshold. Values with probability lower than alpha are considered anomalous.
        
        Returns:
            A binary tensor of shape (batch_size,) where `True` represents an anomalous sample and `False` represents a 
            normal sample.
        """
        p = self.reconstructed_probability(x)
        return p < alpha

    def reconstructed_probability(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the probability density of the input samples under the learned
        distribution of reconstructed data.

        Args:
            x: Input data tensor of shape (batch_size, num_features).

        Returns:
            A tensor of shape (batch_size,) containing the probability densities of
            the input samples under the learned distribution of reconstructed data.
        """
        with torch.no_grad():
            pred = self.predict(x)
        recon_dist = Normal(pred['recon_mu'], pred['recon_sigma'])
        x = x.unsqueeze(0)
        p = recon_dist.log_prob(x).exp().mean(dim=0).mean(dim=-1)  # vector of shape [batch_size]
        return p

    def generate(self, batch_size: int = 1) -> torch.Tensor:
        """
        Generates a batch of samples from the learned prior distribution.

        Args:
            batch_size: Number of samples to generate.

        Returns:
            A tensor of shape (batch_size, num_features) containing the generated
            samples.
        """
        z = self.prior.sample((batch_size, self.latent_size))
        recon_mu, recon_sigma = self.decoder(z).chunk(2, dim=1)
        recon_sigma = softplus(recon_sigma)
        return recon_mu + recon_sigma * torch.rand_like(recon_sigma)
    
    
    def training_step(self, batch, batch_idx):
        x = batch
        loss = self.forward(x)
        if self.global_step % self.log_steps == 0:
            self.log('train/loss', loss['loss'])
            self.log('train/loss_kl', loss['kl'], prog_bar=False)
            self.log('train/loss_recon', loss['recon_loss'], prog_bar=False)
            self._log_norm()

        return loss
    

    def validation_step(self, batch, batch_idx):
        x = batch
        loss = self.forward(x)
        self.log('val/loss_epoch', loss['loss'], on_epoch=True)
        self.log('val_kl', loss['kl'], self.global_step)
        self.log('val_recon_loss', loss['recon_loss'], self.global_step)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


    def _log_norm(self):
        norm1 = sum(p.norm(1) for p in self.parameters())
        norm1_grad = sum(p.grad.norm(1) for p in self.parameters() if p.grad is not None)
        self.log('norm1_params', norm1)
        self.log('norm1_grad', norm1_grad)

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


