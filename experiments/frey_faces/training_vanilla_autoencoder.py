import os
import sys

from utils import load_frey_faces
sys.path.append(os.path.join(__file__, '..', '..', '..'))
from auto_encoder import AutoEncoder

import numpy as np
import tensorflow as tf


def build_model(n_bottleneck=2):
    img_shape = [28,20,1]
    all_filters = [8, 16, 32, 64, 128, 256] # for cnn-architecture

    # encoder
    inputs = x = tf.keras.layers.Input(shape=img_shape)
    for filters in all_filters:
        x = tf.keras.layers.Conv2D(filters, kernel_size=4, activation='tanh')(x)
        x = tf.keras.layers.BatchNormalization()(x)
    unflattened_shape = x.shape[1:]
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(units=n_bottleneck, activation='linear')(x)
    encoder_network = tf.keras.Model(inputs, outputs)
    encoder_network.summary()

    # decoder
    inputs = x = tf.keras.layers.Input(shape=n_bottleneck)
    x = tf.keras.layers.Dense(units=np.prod(unflattened_shape), activation='tanh')(x)
    x = tf.keras.layers.Reshape(target_shape=unflattened_shape)(x)
    for filters in all_filters[-2::-1]:
        x = tf.keras.layers.Conv2DTranspose(filters, kernel_size=4, activation='tanh')(x)
        x = tf.keras.layers.BatchNormalization()(x)
    outputs = tf.keras.layers.Conv2DTranspose(1, kernel_size=4, activation='sigmoid')(x)
    decoder_network = tf.keras.Model(inputs, outputs)
    decoder_network.summary()

    # vanilla auto-encoder
    ae = AutoEncoder(encoder_network, decoder_network)
    return ae


if __name__ == '__main__':
    imgs = load_frey_faces()
    dir = os.path.join('results/models/vanilla')
    for bottleneck in [102]:
        encoder_path = os.path.join(dir, f'encoder_{bottleneck=}.keras')
        decoder_path = os.path.join(dir, f'decoder_{bottleneck=}.keras')
        
        if not (os.path.exists(encoder_path) and os.path.exists(decoder_path)):
            # train
            ae = build_model(bottleneck)
            ae.compile(loss=tf.keras.losses.MeanSquaredError(),
                       optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
            ae.fit(x=imgs, y=imgs, validation_split=0.2, epochs=2, callbacks=callbacks)
            ae.encoder.save(encoder_path)
            ae.decoder.save(decoder_path)
            ae.save(os.path.join(dir, 'tmp.tf'))
        else:
            print(f'Trained Model for {bottleneck=} already exists. No Training...')
