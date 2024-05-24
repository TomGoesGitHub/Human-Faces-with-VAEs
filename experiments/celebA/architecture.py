import math
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
tfd = tfp.distributions

import sys
import os
sys.path.append(os.path.join(os.pardir, os.pardir)) # src-directory
from auto_encoder import VariationalAutoEncoder


LATENT_DIM = 2048
IMAGE_SHAPE = [128,128,3]


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
        self.resnet_ccn_arc = [16, 32, 64, 128, 256, 512]
        self.dense_units = [64, 128, 256, 512, 1024, 2048]

        self.build(input_shape=[None, *self.image_shape])
        self.summary(expand_nested=True)

    def build_cnn_block(self, filters, name='cnn_block'):
        layers = [
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters, kernel_size=10, strides=1, activation='relu',
                                   padding='same', kernel_regularizer='l2'),
            tf.keras.layers.Dropout(rate=0.2),
        ]
        return tf.keras.Sequential(layers, name)

    def build_resnet_block(self, input_shape, filters, repeats=3, name='resnet_block'):
        filters_hidden = input_shape[-1]
        
        cnn_blocks = [self.build_cnn_block(filters=filters_hidden, name=f'cnn_block_{i}_in_{name}')
                      for i in range(repeats)]
        
        inputs = x = tf.keras.Input(shape=np.array(input_shape, dtype=int))
        for cnn_block in cnn_blocks:
            x = cnn_block(x)
        outputs = tf.keras.layers.Conv2D(filters=filters, kernel_size=10, strides=1,
                                         activation='relu', padding='same')(x + inputs)
        
        return tf.keras.Model(inputs, outputs, name=name)

    def build(self, input_shape):
        # ResNet-CNN
        self.resnet_embedding = self.build_cnn_block(filters=8, name='resnet_embedding')
        self.resnet_blocks = [self.build_resnet_block(input_shape=[128/2**i, 128/2**i, 8*2**(i-1)],
                                                      filters=f, name=f'resnet_block_{i}')
                              for i, f in enumerate(self.resnet_ccn_arc, start=1)]
        
        # Dense
        self.dense_layers = [tf.keras.layers.Dense(units=64*2**i, name=f'dense_{i+1}')
                             for i in range(len(self.resnet_ccn_arc))]

        # Distribution
        self.posterior_loc_layer = tf.keras.layers.Dense(units=self.latent_dim,
                                    activation='sigmoid', name='Mean_of_Posterior')

        self.posterior_scale_layer = tf.keras.layers.Dense(units=self.latent_dim,
                                    activation='sigmoid', name='ScaleDiag_of_Posterior')
        
        self.distribution_layer = PosteriorDistributionLayer()
        
        super().build(input_shape)
          
    def call(self, input):
        x = input # note: x represents the main flow
        
        # ResNet-CNN
        x = self.resnet_embedding(x)
        x = tf.keras.layers.MaxPool2D()(x)
        
        cnn_outs = []
        for resnet_block in self.resnet_blocks:
            x = resnet_block(x)
            cnn_outs.append(x)
            x = tf.keras.layers.MaxPool2D()(x)

        # Skip Connetions
        dense_ins = [tf.keras.layers.Flatten()(cnn_out) for cnn_out in cnn_outs]

        # Dense
        dense_outs = [dense_layer(dense_in)
                      for dense_layer, dense_in
                      in zip(self.dense_layers, dense_ins)]

        # Distribution
        x = tf.concat(dense_outs, axis=-1)
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
        self.network_arc = [512, 256, 128, 64, 32, 16]
        self.dense_units = [2048, 1024, 512, 256, 128, 64]
        self.build(input_shape=[None, self.latent_dim])
        self.summary(expand_nested=True)

    def build_cnn_block(self, filters, name='cnn_block'):
        layers = [
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters, kernel_size=10, strides=1, activation='relu',
                                   padding='same', kernel_regularizer='l2'),
            tf.keras.layers.Dropout(rate=0.2),
        ]
        return tf.keras.Sequential(layers, name)

    
    def build_resnet_block(self, input_shape, filters, repeats=3, name='resnet_block'):
        inputs = x = tf.keras.Input(shape=np.array(input_shape, dtype=int))
        x = x_ = tf.keras.layers.Conv2D(filters, kernel_size=10, strides=1,
                                        activation='relu', padding='same')(x)
        
        cnn_blocks = [self.build_cnn_block(filters, name=f'cnn_block_{i}_in_{name}')
                      for i in range(repeats)]
        
        
        for cnn_block in cnn_blocks:
            x = cnn_block(x)
        outputs = tf.keras.layers.ReLU()(x + x_)
        
        return tf.keras.Model(inputs, outputs, name=name)

    def build(self, input_shape):
        # embedding
        self.dense_embd_layer = tf.keras.layers.Dense(units=sum(self.dense_units), activation='relu')

        # Dense
        self.dense_layers = [tf.keras.layers.Dense(units=math.prod([2**i, 2**i, u]), activation='relu')
                             for i, u in enumerate(self.network_arc, start=1)]

        # ResNet-CNN
        self.resnet_blocks = [self.build_resnet_block(input_shape=[2**i, 2**i, f], filters=f/2, name=f'resnet_block_{i}')
                              for i, f in enumerate(self.network_arc, start=1)]
        
        # distribution
        channels = 3 # RGB
        self.likelihood_loc_layer = tf.keras.layers.Conv2D(filters=channels, kernel_size=10,
                                      strides=1, activation='sigmoid', padding='same',
                                      name='Mean_of_Likelihood')
        self.likelihood_scale_layer = tf.keras.layers.Conv2D(filters=channels, kernel_size=10,
                                       strides=1, activation='sigmoid', padding='same',
                                       name='ScaleDiag_of_Likelihood')
        self.distribution_layer = LikelihoodDistributionLayer()
        super().build(input_shape)

    def call(self, input):
        x = input # note: x represents main flow
        
        # Dense
        x = self.dense_embd_layer(x)
        dense_ins = tf.split(x, num_or_size_splits=self.dense_units, axis=-1)
        dense_outs = [layer(input) for layer, input in zip(self.dense_layers, dense_ins)]

        # Skip Connections
        cnn_ins = [tf.keras.layers.Reshape([2**i, 2**i, channels])(dense_out) 
                   for i, (channels, dense_out) in enumerate(zip(self.network_arc, dense_outs), start=1)]

        # CNN
        x = tf.zeros_like(cnn_ins[0])
        for resnet_block, cnn_in in zip(self.resnet_blocks, cnn_ins):
            x = resnet_block(x + cnn_in)
            x = tf.keras.layers.UpSampling2D()(x)

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


if __name__ == '__main__':
    # enc = EncoderNetwork()
    # dec = DecoderNetwork()

    model = build_model()
    dummy_in = tf.zeros([16, *IMAGE_SHAPE])
    model.encode(dummy_in)
    # tf.keras.models.save_model(model, 'tmp_model.tf')
    # reloaded = tf.keras.models.load_model('tmp_model.tf')
    print('Done!')