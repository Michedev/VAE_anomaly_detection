
# Variational autoencoder for anomaly detection

This repo contains my personal implementation of Variational autoencoder
 in tensorflow for anomaly detection, that follows
 [Variational Autoencoder based Anomaly Detection using Reconstruction Probability by Jinwon An, Sungzoon Cho](https://www.semanticscholar.org/paper/Variational-Autoencoder-based-Anomaly-Detection-An-Cho/061146b1d7938d7a8dae70e3531a00fceb3c78e8)
 <br>
 In order to make work the variational autoencoder for anomaly detection i've to change the last layer of the decoder
 from a simple fully connected layer to two layers that estimate mean and variance
 of x~ ( _p(x|z)_ )
<br>
NOTE: this works with tensorflow 1.x <br>
## Main methods of VAE class
- My implementation try to follow scikit-learn api, so it has a _fit_ method
to train the model 
- The constructor of VAE define the architecture of the VAE but some things are fixed
like the type of layers, fully connected layers with _relu_ activation function.
You can specify the size of each layer and the size of the latent space, optionally the decoder layer sizes
but by default are the specular of the encoder layer sizes and mu and sigma of
the prior _p(z)_ that it is a normal distribution by default and
cannot be changed
- It has also different methods like _generate_
that does a sampling from the latent distribution, and run the
decoder in order to generate new data like the X fitted
- _reconstruct_ receive in input X then fit the encoder, calculate mean
and variance of the latent space, then decode the sampled values from
the latent distribution
- For anomaly detection there are two methods _reconstructed_probability_
and _is_outlier_: the first estimate the reconstructed_probability like name
guesses while the second use them to tell than an observation is an outlier if
the rec. prob. is less than alpha, an input parameter such that by default if 0.05

## Code examples
```python
latent_size = 32
encoders_sizes = np.linspace(X_train.shape[1], latent_size, 7).astype('int')[1:-1]
vae = VAE((X_train.shape[1],), encode_sizes=encoders_sizes, latent_size=latent_size, lr=0.00001)
vae.fit(X_train, epochs=200, batch_size=256)
p_x = vae.reconstructed_probability(X_train)
np.save('P_x', p_x)

```

In my implementation i used InteractiveSession instead Session to ease 
the experiments in jupyter notebook but if you want to auto-close the
session after getting the reconstructed probability you can use _with_
keyword

```python
latent_size = 32
encoders_sizes = np.linspace(X_train.shape[1], latent_size, 7).astype('int')[1:-1]
with VAE((X_train.shape[1],), encode_sizes=encoders_sizes, latent_size=latent_size, lr=0.00001) as vae:
    vae.fit(X_train, epochs=200, batch_size=256)
    p_x = vae.reconstructed_probability(X_train)
np.save('P_x', p_x)

```


