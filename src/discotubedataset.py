import os
import pickle
import random

import torch
# from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset
# , DataLoader
# from torchvision import transforms, utils


class DiscotubeDataset(Dataset):
    """DiscoTube dataset."""

    def __init__(self, pickle_file, root, sampling_strategy='whole', seed=None, transform=None):
        """
        Args:
            pickle_file (string): Path to the pickle file with annotations.
            root (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.gt = pickle.load(open(pickle_file, 'rb'))
        self.keys = list(self.gt.keys())
        
        self.root = root
        self.sampling_strategy=sampling_strategy
        self.seed = seed
        
        if seed:
            numpy.random.seed(seed=seed)
            random.seed(a=seed, version=2)

        self.transform = transform
        
        self.n_bands = 96
        self.patch_size = 64

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        key = self.keys[idx]
        melspectrogram_file = os.path.join(self.root, key)
        
        floats_num = os.path.getsize(melspectrogram_file) // 2  # each float16 has 2 bytes
        frames_num = floats_num // self.n_bands

        if self.sampling_strategy == 'whole':
            fp = np.memmap(melspectrogram_file, dtype='float16',
               mode='r', shape=(frames_num, self.n_bands))

        elif self.sampling_strategy == 'random_patch':
            if frames_num < self.patch_size:
                fp = np.memmap(melspectrogram_file, dtype='float16',
                               mode='r', shape=(frames_num, self.n_bands))

                melspectrogram = np.zeros([self.patch_size, self.n_bands])
                melspectrogram[:frames_num, :] = np.array(fp)
            else:
                # offset: idx * bands * bytes per float
                offset = random.randint(0, frames_num - self.patch_size) * self.n_bands * 2 
                fp = np.memmap(melspectrogram_file, dtype='float16', mode='r',
                               shape=(self.patch_size, self.n_bands), offset=offset)  
        else:
            raise(Exception('DiscotubeDataset: Sampling strategy not implemented'))

        # Put the data in a numpy ndarray and add channel axis         
        melspectrogram = np.expand_dims(np.array(fp, dtype='float32'), axis=0)

        del fp

        tags = self.gt[key].astype('float32')

        sample = {'melspectrogram': melspectrogram, 'tags': tags}

        if self.transform:
            sample = self.transform(sample)

        return sample
