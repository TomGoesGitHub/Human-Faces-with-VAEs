import numpy as np

def load_frey_faces(shuffle=True):
    FILE_PATH = 'D:\\DATASETS\\FreyFaces\\frey_rawface.mat' # todo
    from scipy.io import loadmat
    data = loadmat(FILE_PATH)
    imgs = data['ff'].T.reshape([-1, 28, 20, 1])
    imgs = imgs / 255.
    np.random.seed(42)
    if shuffle: np.random.shuffle(imgs)
    return imgs

def load_img_pairs_for_linear_interpolation():
    imgs = load_frey_faces(shuffle=False)
    idx_pairs = [(1600, 1097), (300, 1000),] # index, manually chosen
    img_pairs = [(imgs[i1], imgs[i2]) for (i1,i2) in idx_pairs]
    return img_pairs

# if __name__ == '__main__': # todo: main is tmp
#     import matplotlib.pyplot as plt
#     imgs = load_frey_faces(shuffle=False)
#     img_pairs = load_img_pairs_for_linear_interpolation()
#     print('Done!')