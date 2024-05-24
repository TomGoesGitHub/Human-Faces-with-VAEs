import os
import sys
import math

sys.path.append(os.path.join(os.pardir, os.pardir)) # src-directory
from auto_encoder import VariationalAutoEncoder
from experiments.utils import Dataloader
from experiments.visualization import VisualizeReconstrcutionCallback

import tensorflow as tf
import matplotlib.pyplot as plt


def build_model(n_bottleneck):
    img_shape = (218, 178, 3)
    all_filters = [8, 16, 32, 64, 64, 64, 64] # for cnn-architecture

    # encoder
    inputs = x = tf.keras.layers.Input(shape=img_shape)
    for filters in all_filters:
        x = tf.keras.layers.Conv2D(filters, kernel_size=12, activation='tanh')(x)
        #x = tf.keras.layers.MaxPool2D(pool_size=2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
    unflattened_shape = x.shape[1:]
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(units=n_bottleneck, activation='linear')(x)
    encoder_network = tf.keras.Model(inputs, outputs)
    encoder_network.summary()

    # # decoder
    # inputs = x = tf.keras.layers.Input(shape=n_bottleneck)
    # x = tf.keras.layers.Dense(units=math.prod(unflattened_shape), activation='tanh')(x)
    # x = tf.keras.layers.Reshape(target_shape=unflattened_shape)(x)
    # for filters in all_filters[-2::-1]:
    #     x = tf.keras.layers.Conv2DTranspose(filters, kernel_size=6, activation='tanh')(x)
    #     x = tf.keras.layers.BatchNormalization()(x)
    # outputs = tf.keras.layers.Conv2DTranspose(1, kernel_size=6, activation='sigmoid')(x)
    # decoder_network = tf.keras.Model(inputs, outputs)
    # decoder_network.summary()

    # # vae
    # vae = VariationalAutoEncoder(encoder_network, decoder_network)
    # return vae


if __name__ == '__main__':
    # data
    DATA_DIR = 'D:\DATASETS\CelebA\img_align_celeba'
    TRAIN_SPLIT = 0.8
    BATCH_SIZE = 256
    
    file_paths = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR)]
    file_paths_train = file_paths[: math.ceil(TRAIN_SPLIT*len(file_paths))]
    file_paths_val = file_paths[math.ceil(TRAIN_SPLIT*len(file_paths)) :]
    data_loader_train= Dataloader(file_paths_train, batch_size=BATCH_SIZE)
    data_loader_val = Dataloader(file_paths_val, batch_size=BATCH_SIZE)

    vae = build_model(n_bottleneck=250)
    raise
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    visualization = VisualizeReconstrcutionCallback(images=[plt.imread(p) for p in file_paths[:9]])
    vae.fit(x=data_loader_train, validation_data=data_loader_val, epochs=1000, callbacks=[early_stopping, visualization])

    print('Done!')