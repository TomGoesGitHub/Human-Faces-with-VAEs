import os
import sys

from utils import load_frey_faces, load_img_pairs_for_linear_interpolation
sys.path.append(os.path.join(os.pardir, os.pardir)) # src-directory
from auto_encoder import AutoEncoder
from experiments.visualization import visualize_reconstruction, visualize_interpolation_in_latent_space,\
                                      visualize2DManifold, visualize_2D_latent_space

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # data
    imgs = load_frey_faces()
    img_pairs_for_interpolation = load_img_pairs_for_linear_interpolation()
    random_idx = np.random.choice(len(imgs), size=9)
    examples = imgs[random_idx]

    BOTTLENECK = [100]
    for bottleneck in BOTTLENECK:
        # reload trained model
        encoder_path = os.path.join('results','models', 'vanilla_auto_encoder', f'encoder_{bottleneck=}.keras')
        decoder_path = os.path.join('results','models', 'vanilla_auto_encoder', f'decoder_{bottleneck=}.keras')
        encoder = tf.keras.models.load_model(encoder_path)
        decoder = tf.keras.models.load_model(decoder_path)
        auto_encoder = AutoEncoder(encoder, decoder) # todo make load method in AE-class

        # visualize reconstruction
        fig = visualize_reconstruction(auto_encoder, images=examples)
        fname = os.path.join('results','plots', f'reconstruction_vanilla_{bottleneck=}.png')
        plt.savefig(fname)

        # traverse latent space
        latent_pairs = [auto_encoder.get_latent_representation(tf.constant(pair))
                        for pair in img_pairs_for_interpolation]
        for i, (z_start, z_end) in enumerate(latent_pairs):
            fig = visualize_interpolation_in_latent_space(auto_encoder, z_start, z_end)
            fname = os.path.join('results','plots', f'interpolation_z{2*i+1}z{2*i+2}_vanilla_{bottleneck=}.png')
            fig.savefig(fname)

        if bottleneck == 2:
            # visualize 2D-manifold
            fig = visualize2DManifold(auto_encoder, imgs)
            fname = os.path.join('results','plots', f'manifold_vanilla_{bottleneck=}.png')
            fig.savefig(fname)

            # scatter-plot latent space
            latents = auto_encoder.encode(imgs)
            fig = visualize_2D_latent_space(latents)
            ax = plt.gca()
            for i, (z_start, z_end) in enumerate(latent_pairs):
                ax.plot([z_start[0], z_end[0]], [z_start[1], z_end[1]], c='black', ls='--', marker='.', markersize=10)
                ax.annotate(f'z{2*i+1}', z_start+1) # note: +1 in order to avoid overlap
                ax.annotate(f'z{2*i+2}', z_end+1)
            fname = os.path.join('results','plots', f'2d_zspace_vanilla.png')
            fig.savefig(fname)
    
    print('Done!')