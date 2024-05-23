## Introduction
The project is dedicated to exploring how Variational Autoencoders (VAEs) differ from traditional Autoencoders (AEs) and how they can be used as generative models. The core of the work focuses on generating artificial human faces that visually resemble the real faces from the training data (CelebA dataset) as closely as possible.


Contrary to common (and incorrect) assumptions, vanilla AEs are generally not suitable as generative models. With the classical AE architecture, it is typically not possible to generate new artificial data points, because the latent space is often not densely populated with training data, making it difficult to achieve high-quality reconstructions for arbitrary latent samples (the Curse of Dimensionality exacerbates this issue). Furthermore, with the classical AE, it remains unclear which distribution p(z) the latent representation follows. This motivates the architecture of VAEs.

Variational Autoencoders (VAEs. Note how the term variational indicates the connection to Variational Inference) are a probabilistic framework and a straightforward application of the Evidence Lower Bound (ELBO). Like the vanilla AE, the VAE has an encoder and a decoder part. The outputs of the encoder and decoder are the probability distributions p(z|x) and p(x|z). Additionally, the VAE is equipped with a user-defined prior distribution p(z) (typically N(0,1)). This allows the VAE to overcome the weakness of traditional AEs, by ensuring, through an appropriate choice of p(z), that the latent space is densely populated. 

## VAE Architecture
![image](https://github.com/TomGoesGitHub/Human-Faces-with-VAEs/assets/81027049/e8dd6885-88c2-4a61-a5ab-e6425c68ca4e)

## Neural Network Architecture
Below, the architecture for the encoder is shown. The decoder is set up symmetrically s.t. it inverts the operations of the encoder.

![image](https://github.com/TomGoesGitHub/Human-Faces-with-VAEs/assets/81027049/24128a72-b0f8-435f-99bf-1c48b1b33934)


## Training on CelebA
(buffering the GIF may take a few seconds)

The GIF shows the performance on validation datapoints (unseen during training).

![vae_training](https://github.com/TomGoesGitHub/Human-Faces-with-VAEs/assets/81027049/01e330c3-7e46-4d6c-9ba4-0b8b3c053798)

## VAE as generative model
The model was trained for 25 Epochs. After Training, the VAE was able to recreate images from the validation dataset (unseen during training) with high quality. However, when using the VAE as a generative model, i.e. sampling z~p(z) from the prior and then only using the decoder part, the image quality was drastically reduced.

![VAE_celebA_36fromNoise](https://github.com/TomGoesGitHub/Human-Faces-with-VAEs/assets/81027049/fda64b1b-20a7-4cac-bb7b-d109867aa985)


What went wrong? Despite the constraint on the Kullback-Leibler divergence, the latent variable z was not following the prior distribution. One could argue, that the constraint was not strong enough and should have been reinforced by an additional weighting factor.
Therefore, the sampling process was manually corrected: Now a corrected latent variable z' is fed to the decoder, where z'=z+0.5 with z~p(z) being sampled from the original prior (this new distribution seemed to fit better to the actual distribution in the latent space). After this correction, a much better image quality was obtained.

![VAE_celebA_36fromNoise_corrected](https://github.com/TomGoesGitHub/Human-Faces-with-VAEs/assets/81027049/1dfdcdca-b03c-4984-afc4-34cbdcdd97b5)


## Having fun in latent space
Some interpolations between random faces in latent space.

![VAE_celebA_latentSpaceInterpolation](https://github.com/TomGoesGitHub/Human-Faces-with-VAEs/assets/81027049/4f98dc91-21c8-4dbf-b978-e7107636da0b)
