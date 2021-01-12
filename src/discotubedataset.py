from pathlib import Path
import pickle
import random

import torch
import numpy as np
from torch.utils.data import Dataset
import logging


class DiscotubeDataset(Dataset):
    """DiscoTube dataset."""

    def __init__(self, pickle_file, root, conf, sampling_strategy='random_patch', transform=None):
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

        self.logger = logging.getLogger('TrainManager.DiscotubeDataset')

        if conf.seed is not None:
            self.seed = conf.seed
            np.random.seed(seed=self.seed)
            random.seed(a=self.seed, version=2)

        self.transform = transform

        self.n_bands = conf.y_size
        self.patch_size = conf.x_size
        self.patch_hop_size = self.patch_size
        self.hop_size = conf.hop_size

        if sampling_strategy == 'random_patch':
            self.overlap_add = False
        elif sampling_strategy == 'overlap_add':
            self.overlap_add = True
        else:
            self.logger.error(f'`{sampling_strategy}` not implemented. Use `random_patch` or `whole_trach`')
            raise

        if self.overlap_add:
            patch_hop_size_seconds = conf.patch_hop_size * conf.hop_size / conf.sample_rate
            self.max_patches_per_track = int(np.ceil(conf.evaluation_time / patch_hop_size_seconds))
            self.logger.debug(f'samples per validation track: {self.max_patches_per_track}')

            extended_gt = dict()

            for key, tags in self.gt.items():
                melspectrogram_file = Path(self.root, key)

                floats_num = melspectrogram_file.stat().st_size // 2  # each float16 has 2 bytes
                frames_num = floats_num // self.n_bands

                for hop in range(self.max_patches_per_track):
                    offset_idx = hop * self.patch_hop_size

                    if offset_idx < (frames_num - self.patch_size):
                        extended_gt[f'{key}-{hop}'] = (key, tags, offset_idx)
                    else:
                        break

            keys = list(extended_gt.keys())
            self.gt = extended_gt
            self.keys = list(self.gt.keys())

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        key = self.keys[idx]

        if self.overlap_add:
            key, tags, offset_idx = self.gt[self.keys[idx]]
            frames_to_read = self.patch_hop_size

        melspectrogram_file = Path(self.root, key)

        if not self.overlap_add:
            tags = self.gt[key]

            frames_num = melspectrogram_file.stat().st_size // (2 * self.n_bands)  # each float16 has 2 bytes
            frames_to_read = min(self.patch_size, frames_num)
            max_frame = max(frames_num - self.patch_size, 0)
            offset_idx = random.randint(0, max_frame)


        # offset: idx * bands * bytes per float
        offset = offset_idx * self.n_bands * 2
        fp = np.memmap(melspectrogram_file, dtype='float16', mode='r',
                       shape=(frames_to_read, self.n_bands), offset=offset)

        # put the data in a numpy ndarray
        melspectrogram = np.array(fp, dtype='float32')

        if frames_to_read < self.patch_size:
            padding_size = self.patch_size - frames_to_read
            melspectrogram = np.vstack([melspectrogram, np.zeros([padding_size, self.n_bands])])
            melspectrogram = np.roll(melspectrogram, padding_size // 2, axis=0)  # center the padding
            np.save(f'padded_melspetrogram.npy', melspectrogram)
            self.logger.debug(f'incomplete audio frame {key}. n of frames: {frames_to_read}. shape: {melspectrogram.shape}')

        del fp

        sample = {'melspectrogram': melspectrogram, 'tags': tags.astype('float32'), 'key': key}

        return sample
