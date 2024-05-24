import math
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

import sys
import os
sys.path.append(os.path.join(os.pardir, os.pardir)) # src-directory
from auto_encoder import VariationalAutoEncoder


LATENT_DIM = 250
IMAGE_SHAPE = [256,256,3]


# note: I excluded Distribution Layers into own classes, otherwise I cannot save the
#       custom model (see: https://github.com/tensorflow/probability/issues/1350)
@tf.keras.utils.register_keras_serializable()
class PosteriorDistributionLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
    
    def call(self, input):
        dist = tfp.layers.DistributionLambda(self._make_distribution_fn)(input)
        return dist
    
    def _make_distribution_fn(self, input):
        posterior_loc, posterior_scale = tf.unstack(input, axis=-1)
        dist = tfd.Independent(tfd.Normal(posterior_loc,posterior_scale),
                               reinterpreted_batch_ndims=1)
        return dist
    

@tf.keras.utils.register_keras_serializable()
class LikelihoodDistributionLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, input):
        dist = tfp.layers.DistributionLambda(self._make_distribution_fn)(input)
        return dist
    
    def _make_distribution_fn(self, input):
        likelihood_loc, likelihood_scale = tf.unstack(input, axis=-1)
        single_distributions = tfd.Normal(likelihood_loc,likelihood_scale)
        dist = tfd.Independent(single_distributions, reinterpreted_batch_ndims=3)
        # note: reinterpreted_batch_ndims=3 is for RGB
        return dist


@tf.keras.utils.register_keras_serializable()
class EncoderNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__(name='Encoder')
        self.latent_dim = LATENT_DIM
        self.image_shape = IMAGE_SHAPE
        self.network_arc = [32, 32, 64, 128, 256, 256, 256]

        self.build(input_shape=[None, *self.image_shape])
        self.summary(expand_nested=True)

    def build_cnn_block(self, filters, name='cnn_block'):
        layers = [
            tf.keras.layers.Conv2D(filters, kernel_size=4, strides=1,
                                   activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(pool_size=(2,2))
        ]
        return tf.keras.Sequential(layers, name)
    
    def build(self, input_shape):
        self.hidden_layers = tf.keras.Sequential(
            [self.build_cnn_block(f, name=f'Cnn_Block_{i}')
             for i, f in enumerate(self.network_arc, start=1)],
            name='Hidden_Layers')

        self.posterior_loc_layer = tf.keras.layers.Dense(units=self.latent_dim,
                                    activation='sigmoid', name='Mean_of_Posterior')

        self.posterior_scale_layer = tf.keras.layers.Dense(units=self.latent_dim,
                                    activation='sigmoid', name='ScaleDiag_of_Posterior')
        
        self.distribution_layer = PosteriorDistributionLayer()
        
        super().build(input_shape)
          
    def call(self, input):
        # CNN
        x = input
        x = self.hidden_layers(x)
        # distribution
        x = tf.keras.layers.Flatten()(x)
        posterior_loc = self.posterior_loc_layer(x)
        posterior_scale = self.posterior_scale_layer(x) + 0.01
        stacked = tf.stack([posterior_loc, posterior_scale],axis=-1)
        posterior_distribution = self.distribution_layer(stacked)
        return posterior_distribution


@tf.keras.utils.register_keras_serializable()
class DecoderNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__(name='Decoder')
        self.latent_dim = LATENT_DIM
        self.image_shape = IMAGE_SHAPE
        self.network_arc = [256, 256, 256, 128, 64, 32]

        self.build(input_shape=[None, self.latent_dim])
        self.summary(expand_nested=True)

    def build_cnn_block(self, filters, name='cnn_block'):
        layers = [
            tf.keras.layers.Conv2D(filters, kernel_size=4, strides=1,
                                   activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.UpSampling2D(size=2),
        ]
        return tf.keras.Sequential(layers, name)

    def build(self, input_shape):
        self.flat_embd_layer = tf.keras.layers.Dense(units=math.prod([4,4,128]),
                                                     activation='relu')
        self.cnn_embd_layer = tf.keras.layers.Reshape(target_shape=[4,4,128])

        self.hidden_layers = tf.keras.Sequential(
              [self.build_cnn_block(f, name=f'Cnn_Block_{i}')
              for i, f in enumerate(self.network_arc, start=1)],
              name='Hidden_Layers')

        channels = 3 # RGB
        self.likelihood_loc_layer = tf.keras.layers.Conv2D(filters=channels, kernel_size=3,
                                      strides=1, activation='sigmoid', padding='same',
                                      name='Mean_of_Likelihood')
        self.likelihood_scale_layer = tf.keras.layers.Conv2D(filters=channels, kernel_size=3,
                                       strides=1, activation='sigmoid', padding='same',
                                       name='ScaleDiag_of_Likelihood')
        self.distribution_layer = LikelihoodDistributionLayer()
        super().build(input_shape)

    def call(self, input):
        # embedding
        x = input
        x = self.flat_embd_layer(x)
        x = self.cnn_embd_layer(x)
        # CNN
        x = self.hidden_layers(x)
        # distribution
        likelihood_loc = self.likelihood_loc_layer(x)
        likelihood_scale = self.likelihood_scale_layer(x) + 0.01
        stacked = tf.stack([likelihood_loc, likelihood_scale], axis=-1)
        likelihood_distribution = self.distribution_layer(stacked)
        return likelihood_distribution

def build_model():
    encoder = EncoderNetwork()
    decoder = DecoderNetwork()
    prior = tfp.distributions.MultivariateNormalDiag(
        loc=tf.zeros(shape=LATENT_DIM),
        scale_diag=tf.ones(shape=LATENT_DIM)
    )
    vae = VariationalAutoEncoder(encoder, decoder, prior)
    return vae


# if __name__ == '__main__':
#     model = build_model()
#     dummy_in = tf.zeros([16,256,256,3])
#     model.encode(dummy_in)
#     tf.keras.models.save_model(model, 'tmp_model.tf')
#     reloaded = tf.keras.models.load_model('tmp_model.tf')
#     print('Done!')