import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class Dataloader(tf.keras.utils.Sequence):
    def __init__(self, files, batch_size):
        self.files = files
        self.batch_size = batch_size
    
    def __len__(self):
        return int(np.ceil(len(self.files) / self.batch_size))

    def __getitem__(self, idx):
        low = idx * self.batch_size
        high = min(low + self.batch_size, len(self.files))
        files = self.files[low:high]
        imgs = [plt.imread(f) for f in files]
        imgs = np.array(imgs)
        imgs = imgs / 255 # rescale to range [0,1]
        imgs = tf.convert_to_tensor(imgs, dtype=tf.float32)
        return (imgs, imgs)
