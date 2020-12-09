import torch
import numpy as np
from discotubedataset import DiscotubeDataset
from multilabel_balanced_sampler import MultilabelBalancedRandomSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import logging

def DataLoader(pickle_file, conf, mode='train'):
    num_replicas = conf.local_world_size

    if mode == 'train':
        batch_size = conf.train_batch_size
        sampler = 'multilabel_balanced_random'

        dataset = DiscotubeDataset(pickle_file=pickle_file,
            root=conf.root,
            conf=conf,
            sampling_strategy=conf.train_sampling_strategy)

        rank_sampler = DistributedSampler(dataset,
            num_replicas=num_replicas,
            rank=conf.local_rank)

        # get the indices for this rank.
        indices = [i for i in iter(rank_sampler)]

    elif mode == 'val':
        batch_size = conf.val_batch_size
        sampler = 'sequential'

        dataset = DiscotubeDataset(pickle_file=pickle_file,
            root=conf.root,
            conf=conf,
            sampling_strategy=conf.val_sampling_strategy)

        samples_per_rank = len(dataset) // num_replicas
        indices = range(conf.local_rank * samples_per_rank,
                        (conf.local_rank + 1 ) * samples_per_rank)

    logging.info(f'number of samples per process: {len(indices)}')

    rank_subset = torch.utils.data.Subset(dataset, indices)
    labels = np.array([i['tags'] for i in iter(rank_subset)])

    if sampler == 'multilabel_balanced_random':
        sampler = MultilabelBalancedRandomSampler(
            labels,
            class_choice="least_sampled")

    elif sampler == 'random':
        sampler = RandomSampler(rank_subset)

    elif sampler == 'sequential':
        sampler = SequentialSampler(rank_subset)

    else:
        raise Exception('dataloder: sampler type not implemented')

    loader = torch.utils.data.DataLoader(
        dataset=rank_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=sampler)

    return loader
