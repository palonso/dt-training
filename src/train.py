import os
from datetime import datetime
import configargparse
from pathlib import Path
import json
import logging
import sys

import scipy
import numpy as np
import torch.multiprocessing as mp
import torch
import torch.nn as nn
import torch.distributed as dist
from sklearn import metrics

import utils
from modelfactory import ModelFactory
from dataloader import dataloader
from tblogger import TBLogger


def train(rank, args):
    # for now one gpu per process
    args.rank = rank
    args.exp_id = f"{args.timestamp}_{args.exp_name}"
    gpu = rank
    exp_dir = Path(args.exp_base_dir, args.exp_id)
    args.exp_dir = str(exp_dir)

    exp_dir.mkdir(parents=True, exist_ok=True)

    logger = utils.get_logger(__name__, exp_dir / f"rank_{rank}.log")

    if rank == 0:
        console_handler = utils.get_console_handler(args.logging_level)
        logger.addHandler(console_handler)
        logger.debug(f'console logging level: {args.logging_level}')

    logger.info(f'starting experiment "{args.exp_id}"')
    logger.debug(f'experiment configuration: {vars(args)}')

    with open(str(exp_dir / f"config_{rank}.json"), "w" ) as f:
        json.dump(vars(args), f, indent=4)

    checkpoint_path = exp_dir / "model.checkpoint"

    tb_logger = TBLogger(log_dir=str(exp_dir / "tb_logs" / f"rank_{rank}"))

    logger.info(f'rank: {rank}. gpu: {gpu}. world_size: {args.world_size}')

    logger.debug('initiating process group...')
    logger.debug('ok!')
    dist.init_process_group(backend='nccl',
                            rank=rank,
                            world_size=args.world_size)

    torch.manual_seed(args.seed)
    model_factory = ModelFactory()
    model = model_factory.create("VGG")

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    logger.info(f'number of trainable params: {params}')

    torch.cuda.set_device(gpu)
    model.cuda(gpu)

    for m in model.parameters():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias:
                torch.nn.init.xavier_uniform_(m.bias)

    # define loss function (criterion) and optimizer
    criterion = nn.MultiLabelSoftMarginLoss().cuda(gpu)
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)

    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model,
        device_ids=[gpu])

    # Data loading code
    train_loader = dataloader(args.train_pickle, args, mode='train')

    if args.distributed_validation:
        logger.debug('using distributed validation')

    if gpu==0 or args.distributed_validation:
        val_loader = dataloader(args.val_pickle, args, mode='val')

    stats = {'Loss': [], 'AUC/ROC': [], 'AUC/PR': []}
    best_loss_vas = np.Inf

    start = datetime.now()
    total_step = len(train_loader)
    logger.info(f'number of steps per epoch: {total_step}')

    if args.just_one_batch:
        logger.warning('`just-one-batch` mode on. Intended for development purposes')
        train_iterator = [next(iter(train_loader))]
    else:
        train_iterator = train_loader
        logger.debug('entering the training loop')

    for epoch in range(args.epochs):
        logger.debug(f'starting epoch {epoch}')

        train_loss, loss_val = 0, np.NaN
        y_true_train, y_pred_train, y_true_val, y_pred_val = [], [], [], []
        roc_auc_train, pr_auc_train, roc_auc_val, pr_auc_val = np.NaN, np.NaN, np.NaN, np.NaN

        for i, sample in enumerate(train_iterator):
            specs = sample['melspectrogram'].cuda(non_blocking=True)
            tags = sample['tags'].cuda(non_blocking=True)

            # Forward pass
            logits, sigmoid = model(specs)
            loss = criterion(logits, tags)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

            y_true_train.append(tags.cpu().detach().numpy())
            y_pred_train.append(sigmoid.cpu().detach().numpy())

        y_true_train = np.vstack(y_true_train)
        y_pred_train = np.vstack(y_pred_train)

        roc_auc_train = metrics.roc_auc_score(y_true_train, y_pred_train, average='macro')
        pr_auc_train = metrics.average_precision_score(y_true_train, y_pred_train, average='macro')
        loss_train = train_loss / len(train_loader)

        logger.info(f'Step [{i + 1}/{total_step}]')

        # validation
        if not args.just_one_batch:
            if gpu==0 or args.distributed_validation:
                loss_val = 0
                with torch.no_grad():
                    for i, sample in enumerate(val_loader):
                        specs = sample['melspectrogram'].cuda(non_blocking=True)
                        tags = sample['tags'].cuda(non_blocking=True)
                        # Forward pass
                        logits, sigmoid = model(specs)
                        loss = criterion(logits, tags)

                        loss_val += loss.item()
                        y_true_val.append(tags.cpu().detach().numpy())
                        y_pred_val.append(sigmoid.cpu().detach().numpy())

                y_true_val = np.vstack(y_true_val)
                y_pred_val = np.vstack(y_pred_val)

                roc_auc_val = metrics.roc_auc_score(y_true_val, y_pred_val, average='macro')
                pr_auc_val = metrics.average_precision_score(y_true_val, y_pred_val, average='macro')
                loss_val = loss_val / len(val_loader)

        stats['Loss'].append({'train': loss_train, 'val': loss_val})
        stats['AUC/ROC'].append({'train': roc_auc_train, 'val': roc_auc_val})
        stats['AUC/PR'].append({'train': pr_auc_train, 'val': pr_auc_val})

        elapsed = str(datetime.now() - start)

        
        logger.info(f'Epoch [{epoch + 1:03}/{args.epochs}]: | '
                    f'Train Loss: {loss_train:.3f} | '
                    f'Val Loss: {loss_val:.3f} | '
                    f'Train ROC AUC: {roc_auc_train:.3f} | '
                    f'Val ROC AUC: {roc_auc_val:.3f} | '
                    f'Train AP: {pr_auc_train:.3f} | '
                    f'Val AP: {pr_auc_val:.3f} | '
                    f'Time : {elapsed}')

        tb_logger.write_epoch_stats(epoch, stats)

        # save model
        if rank == 0:
            if loss_val <= best_loss_vas:
                best_loss_vas = loss_val
                torch.save(model.state_dict(), str(checkpoint_path))
                logger.debug('lowest valiation loss achieved. Updating the model!')

        dist.barrier()
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        model.load_state_dict(
            torch.load(str(checkpoint_path), map_location=map_location))

    tb_logger.close()


if __name__ == '__main__':
    parser = configargparse.ArgumentParser(default_config_files=['config.ini'])
    parser.add('-c', '--config-file', is_config_file=True,
               help='config file path')
    parser.add('--master-addr',
               help='address of the master node') 
    parser.add('--master-port',
               help='port of the master node')
    parser.add('--nccl-socket-iframe', help='NCCL socket iframe')
    parser.add('--nccl-ib-disable', 
               help='Whether to use Infiniband')
    parser.add('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add('-np', '--nproc-per-node', default=1, type=int,
               help='ranking within the nodes')
    parser.add('-g', '--gpus-per-proc', default=1, type=int,
               help='number of gpus per process')
    parser.add('--epochs', type=int, metavar='N',
               help='number of total epochs to run')
    parser.add('--train-batch-size', type=int, metavar='N',
               help='train batch size')
    parser.add('--val-batch-size', type=int, metavar='N',
               help='val batch size')
    parser.add('--train-sampling-strategy',
               help='train sampling strategy')
    parser.add('--val-sampling-strategy',
               help='val sampling strategy')
    parser.add('--learning-rate', type=float, metavar='N',
               help='initial learning rate')
    parser.add('--seed')
    parser.add('--root', help='root data directory')
    parser.add('--train-pickle', help='pickle file with the training indices')
    parser.add('--val-pickle', help='pickle file with the validation indices')
    parser.add('--train', action='store_true')
    parser.add('--just-one-batch', action='store_true')
    parser.add('--distributed-validation', action='store_true')
    parser.add('--exp-name', help='the experiment name')
    parser.add('--exp-base-dir', help='the experiment directory')
    parser.add('--x-size', type=int)
    parser.add('--y-size', type=int)
    parser.add('--model-name')
    parser.add('--logging-level')

    args = parser.parse_args()

    args.world_size = args.nproc_per_node * args.nodes

    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    os.environ['NCCL_SOCKET_IFRAME'] = args.nccl_socket_iframe
    os.environ['NCCL_IB_DISABLE'] = args.nccl_ib_disable

    args.timestamp = datetime.now().strftime("%y%m%d-%H%M%S")

    mp.spawn(train, nprocs=args.nproc_per_node, args=(args,))
