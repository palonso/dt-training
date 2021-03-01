from pathlib import Path
import pickle
import random

import torch
import numpy as np
from torch.utils.data import Dataset
import logging


class MSDDataset(Dataset):
    """MSD dataset."""

    def __init__(self, pickle_file, root, conf, sampling_strategy='random_patch', transform=None):
        """
        Args:
            pickle_file (string): Path to the pickle file with annotations.
            root (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.logger = logging.getLogger('TrainManager.DiscotubeDataset')

        self.root = root
        self.sampling_strategy = sampling_strategy
        self.transform = transform

        self.n_bands = conf.y_size
        self.patch_size = conf.x_size
        self.patch_hop_size = self.patch_size  # No overlap
        self.hop_size = conf.hop_size

        stoframes = conf.sample_rate / self.hop_size

        self.gt = pickle.load(open(pickle_file, 'rb'))

        min_duration = conf.min_track_duration
        min_duration_frames = min_duration * stoframes

        # We can't discard short files in this dataset as most of them are already 30s previews 
        unexisting = set()
        for key in self.gt.keys():
            melspectrogram_file = Path(self.root, key)
            if not melspectrogram_file.is_file():
                unexisting.add(key)

        self.logger.debug(f'discarding ({len(unexisting)}) tracks without melspectrograms on disk')
        for k in unexisting:
            self.gt.pop(k, None)
        self.logger.debug(f'using ({len(self.gt.keys())}) tracks')

        # We can't discard short files in this dataset as most of them are already 30s previews 
        # short_files = set()
        # for key in self.gt.keys():
        #     melspectrogram_file = Path(self.root, key)

        #     floats_num = melspectrogram_file.stat().st_size // 2  # each float16 has 2 bytes
        #     frames_num = floats_num // self.n_bands
        #     if frames_num < min_duration_frames:
        #         short_files.add(key)

        # self.logger.debug(f'discarding ({len(short_files)}) tracks of less than {min_duration} seconds')

        # for k in short_files:
        #     self.gt.pop(k, None)

        self.keys = list(self.gt.keys())
        self.tracks = self.keys

        if conf.seed is not None:
            self.seed = conf.seed
            np.random.seed(seed=self.seed)
            random.seed(a=self.seed, version=2)

        # intro durations oscilates between 5 to 20 seconds
        # https://www.bbc.com/news/entertainment-arts-41500692
        # intro_outro_time = 30
        # discard this feature in MSD becase the previews
        # don't start from the beginning of the song
        intro_outro_time = 0

        self.intro_outro_offset = intro_outro_time * stoframes

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

                offset = max(0, frames_num // 2 - self.max_patches_per_track // 2)

                for hop in range(self.max_patches_per_track):
                    offset_idx = offset + hop * self.patch_hop_size

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
            max_frame = frames_num - self.patch_size - self.intro_outro_offset
            min_frame = self.intro_outro_offset

            if max_frame < min_frame:
                max_frame = frames_num - self.patch_size
                min_frame = 0
            offset_idx = random.randint(min_frame, max_frame)


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
