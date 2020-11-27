import torch
import numpy as np
from discotubedataset import DiscotubeDataset
from multilabel_balanced_sampler import MultilabelBalancedRandomSampler
from torch.utils.data.sampler import RandomSampler
import logging

def DataLoader(pickle_file, args, mode='train'):
    num_replicas = args.local_world_size

    if mode == 'train':
        sampling_strategy = args.train_sampling_strategy
        batch_size = args.train_batch_size
        sampler = 'multilabel_balanced_random'
        
    elif mode == 'val':
        sampling_strategy = args.val_sampling_strategy
        batch_size = args.val_batch_size
        sampler = 'random'
        # if not args.distributed_val:
        #     num_replicas = 1

    # Data loading code
    dataset = DiscotubeDataset(pickle_file=pickle_file,
        root=args.root,
        sampling_strategy=sampling_strategy)

    rank_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=num_replicas,
        rank=args.local_rank)

    # get the indices for this rank.
    indices = [i for i in iter(rank_sampler)]
    logging.info(f'number of samples per process: {len(indices)}')

    rank_subset = torch.utils.data.Subset(dataset, indices)
    labels = np.array([i['tags'] for i in iter(rank_subset)])

    if sampler == 'multilabel_balanced_random':
        sampler = MultilabelBalancedRandomSampler(
            labels, 
            class_choice="least_sampled")

    elif  sampler == 'random':
        sampler = RandomSampler(rank_subset)

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
