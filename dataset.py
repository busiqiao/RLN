import glob
import platform
import numpy as np
import scipy.io
import torch
from torch.utils.data.dataset import Dataset


class EEGDataset(Dataset):

    def __init__(self, file_path, num_class=6):
        mat = scipy.io.loadmat(file_path)
        self.data = np.asarray(mat['X_3D'])
        if num_class == 6:
            self.label = np.asarray(mat['categoryLabels']).squeeze() - 1
        elif num_class == 72:
            self.label = np.asarray(mat['exemplarLabels']).squeeze() - 1

    def __len__(self):
        return self.data.shape[2]

    def __getitem__(self, index):
        feature = torch.tensor(self.data[:, :, index], dtype=torch.float)
        label = torch.tensor(self.label[index], dtype=torch.float)
        return feature, label
