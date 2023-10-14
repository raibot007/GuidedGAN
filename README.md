# GuidedGAN

This is work in progress project.
The aim is to guide GAN to generate an image using grdual de-diffusion timesteps and the graidents from a noisy classifier trained on the base image dataset which are adulteratd from the same noise distribution, via multiple mini-GANs.

The motivation is to get a regulated control over Fidelity Vs Diversity tradeoff on the state of the art GANs and improve the FID and other such metrics, as achived by guided diffusion performed by https://github.com/openai/guided-diffusion/
