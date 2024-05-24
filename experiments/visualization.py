import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import tensorflow as tf
import numpy as np

import os
import sys

sys.path.append(os.pardir) # src-directory
from auto_encoder import AutoEncoder, VariationalAutoEncoder

def visualize_reconstruction(auto_encoder, images):
    reconstructed = auto_encoder(tf.convert_to_tensor(images))
    reconstructed = np.array(reconstructed)

    cmap = None if images.shape[-1]==3 else 'gray'

    fig = plt.figure()
    gs = GridSpec(nrows=1, ncols=3, figure=fig, width_ratios=[1, 0.2, 1])
    left_subfigure = fig.add_subfigure(subplotspec=gs[0])
    right_subfigure = fig.add_subfigure(subplotspec=gs[2])

    left_axes = left_subfigure.subplots(nrows=3, ncols=3)
    right_axes = right_subfigure.subplots(nrows=3, ncols=3)

    for ax, img in zip(left_axes.ravel(), images):
        img = np.clip(img, 0, 1)
        ax.imshow(img, cmap)
        ax.axis('off')

    for ax, r in zip(right_axes.ravel(), reconstructed):
        r = np.clip(r, 0, 1)
        ax.imshow(r, cmap)
        ax.axis('off')
    
    left_subfigure.supxlabel('Original')
    right_subfigure.supxlabel('Reconstruction')
    plt.subplots_adjust(top=0.88)
    return fig

def visualize_2D_latent_space(latents, classes=None):
    fig, ax = plt.subplots()
    if classes:
        ax.scatter(x=latents[:,0], y=latents[:,1], c=classes, s=1, cmap='tab10', vmin=min(classes)-0.5, vmax=max(classes)+0.5)
        #cbar = plt.colorbar(ticks=np.arange(10))
    else:
        ax.scatter(x=latents[:,0], y=latents[:,1], alpha=0.3, s=1, cmap='tab10')
    plt.xlabel('Latent 1')
    plt.ylabel('Latent 2')
    return fig

def visualize_interpolation_in_latent_space(auto_encoder, z_start, z_end):
    lin_interpol = np.stack(arrays=[np.linspace(z_i_start, z_i_end, 1000)
                                    for z_i_start, z_i_end in zip(z_start, z_end)], 
                            axis=1)
    reconstructed = auto_encoder.get_reconstruction(lin_interpol)
    imgs=reconstructed[::100]
    fig, axes = plt.subplots(nrows=1, ncols=len(imgs))
    for ax, img in zip(axes.ravel(), imgs):
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        fig.set_size_inches(10, 1.75)
    return fig


def visualize2DManifold(auto_encoder, imgs, nrows=10, ncols=10, ):
    # # linear spaced coordinates on the unit-square
    # x1 = np.linspace(0.01, 0.99, ncols)
    # x2 = np.linspace(0.01, 0.99, nrows)

    # # transformed coordinates on the manifold
    # # note: ppf is the inverse cdf
    # z1 = scipy.stats.norm.ppf(x1)
    # z2 = scipy.stats.norm.ppf(x2)
    # zgrid = np.dstack(np.meshgrid(z1, z2))
    latents = auto_encoder.get_latent_representation(imgs)
    z1_min, z1_max = np.min(latents[:,0]), np.max(latents[:,0])
    z2_min, z2_max = np.min(latents[:,1]), np.max(latents[:,1])

    z1 = np.linspace(z1_min, z1_max, ncols)
    z2 = np.linspace(z2_min, z2_max, nrows)
    zgrid = np.dstack(np.meshgrid(z1, z2))

    z = np.reshape(zgrid, newshape=[-1, 2])
    imgs = auto_encoder.get_reconstruction(z)    


    h, w = imgs.shape[1:3]
    fig, axes = plt.subplots(nrows, ncols, figsize=[10, h/w*10])
    for ax, img in zip(axes.ravel(), imgs):
        ax.imshow(img, cmap='grey')
        #ax.set_aspect('equal')
        ax.axis('off')
    plt.tight_layout(h_pad=0, w_pad=0)
    return fig


class VisualizeReconstrcutionCallback(tf.keras.callbacks.Callback):
    def __init__(self, images, freq=10, save_dir=None):
        self.images = np.array(images)
        self.freq = freq
        self.save_dir = save_dir
    
    def on_epoch_begin(self, epoch, logs=None):
        if epoch % self.freq == 0:
            fig = visualize_reconstruction(auto_encoder=self.model,
                                           images=self.images)
            if self.save_dir:
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
                plt.savefig(os.path.join(self.save_dir, f'reconstruction_{epoch=}'))
            plt.show()
            plt.close()

class GatherGifFramesCallback(tf.keras.callbacks.Callback):
    def __init__(self, image_dict, directory):
        self.image_dict = image_dict
        self.directory = directory
    
    def on_epoch_begin(self, epoch, logs=None):
        imgs = [plt.imread(path) for path in self.image_dict.values()]
        latent = self.model.get_latent_representation(tf.constant(imgs))
        reconstructed = self.model.get_reconstruction(latent)
        reconstructed = np.array(reconstructed)

        for subdir_name, r in zip(self.image_dict.keys(), reconstructed):
            target_dir = os.path.join(self.directory, subdir_name)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            r = np.clip(r, 0, 1)
            plt.imshow(r)
            plt.gca().axis('off')
            plt.savefig(os.path.join(target_dir, f'reconstuction_{epoch=}'))
            plt.close()






