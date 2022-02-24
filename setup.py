from pathlib import Path

from setuptools import setup

with open(Path(__file__).parent / 'requirements.txt') as f:
    requirements = f.read().split('\n')

setup(
    name='vae_anomaly_detection',
    version='1.1.0',
    packages=['vae_anomaly_detection'],
    url='https://github.com/Michedev/VAE_anomaly_detection',
    license='MIT',
    author='Michele De Vita',
    author_email='mik3dev@gmail.com',
    description="Pytorch/TF1 implementation of Variational AutoEncoder for anomaly detection",
    long_description='Pytorch/TF1 implementation of Variational AutoEncoder for anomaly detection following the paper "Variational Autoencoder based Anomaly Detection using Reconstruction Probability by Jinwon An, Sungzoon Cho"',
    classifiers = [
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    install_requires=requirements,
    extras_require={
        'test': ['pytest>=7.0.1'],
        'tensorflow_1': ['tensorflow<2']
    },
    python_requires='>=3.8,<4',

)
