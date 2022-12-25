import h5py
import numpy as np


class DataLoader:
    def __init__(self, hdf5_path):
        self.hdf5_path = hdf5_path

    def read_data(self):
        print('Loading dataset ...')
        with h5py.File(self.hdf5_path, 'r') as hf:
            dset_x = np.array(hf['X'])
            dset_y = np.array(hf['Y'])

            print('Dataset loaded successfully!')
            print("Input shape: {}".format(dset_x.shape))
            print("Target shape: {}".format(dset_y.shape))
            print("=============================")

            return dset_x, dset_y
            