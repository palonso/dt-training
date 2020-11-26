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


def train(args):
    args.local_world_size = args.nproc_per_node
    rank = args.local_rank

    args.exp_id = f"{args.timestamp}_{args.exp_name}"
    exp_dir = Path(args.exp_base_dir, args.exp_id)
    args.exp_dir = str(exp_dir)

    exp_dir.mkdir(parents=True, exist_ok=True)

    logger = utils.get_logger(__name__, exp_dir / f"rank_{rank}.log")

    if rank == 0:
        console_handler = utils.get_console_handler(args.logging_level)
        logger.addHandler(console_handler)
        logger.debug(f'console logging level: {args.logging_level}')

    logger.info(f'starting experiment "{args.exp_id}"')
    logger.debug(f'experiment folder: "{args.exp_dir}"')
    logger.debug(f'experiment configuration: {vars(args)}')
    logger.debug(f'pid: {os.getpid()}')

    with open(str(exp_dir / f"config_{rank}.json"), "w" ) as f:
        json.dump(vars(args), f, indent=4)

    checkpoint_path = exp_dir / "model.checkpoint"

    tb_logger = TBLogger(log_dir=str(exp_dir / "tb_logs" / f"rank_{rank}"))

    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK")
    }
    logger.debug(f"initializing process group with: {env_dict}")


    logger.debug('initiating process group...')
    dist.init_process_group(backend='nccl',
                            init_method='env://')
    logger.debug('ok!')

    torch.manual_seed(args.seed)
    
    # assign gpus
    # gpus_per_process = min(torch.cuda.device_count() // args.local_world_size, args.gpus_per_proc)
    # device_ids = list(range(rank * gpus_per_process, (rank + 1) * gpus_per_process))
    # logger.debug(f"using ({gpus_per_process}) GPUs. device_ids: {device_ids}")

    model_factory = ModelFactory()
    model = model_factory.create("VGG")
    torch.cuda.set_device(rank)
    model.cuda(rank)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    logger.info(f'number of trainable params: {params}')

    for m in model.parameters():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias:
                torch.nn.init.xavier_uniform_(m.bias)

    # define loss function (criterion) and optimizer
    criterion = nn.MultiLabelSoftMarginLoss().cuda(rank)
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)

    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model,
        device_ids=[rank],
        output_device=rank)

    # Data loading code
    train_loader = dataloader(args.train_pickle, args, mode='train')

    if args.distributed_validation:
        logger.debug('using distributed validation')

    # if rank == 0 or args.distributed_validation:
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
        logger.debug(f'starting epoch {epoch + 1}')

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
            logger.info(f'Step [{i + 1}/{total_step}]')

        y_true_train = np.vstack(y_true_train)
        y_pred_train = np.vstack(y_pred_train)

        roc_auc_train = metrics.roc_auc_score(y_true_train, y_pred_train, average='macro')
        pr_auc_train = metrics.average_precision_score(y_true_train, y_pred_train, average='macro')
        loss_train = train_loss / len(train_loader)


        if args.just_one_batch:
            validate = False
        # validation
        if not args.just_one_batch:
            # if rank == 0 or args.distributed_validation:
                loss_val = 0
                with torch.no_grad():
                    for i, sample in enumerate(val_loader):
                        specs = sample['melspectrogram'].cuda(non_blocking=True)
                        tags = sample['tags'].cuda(non_blocking=True)
                        # Forward pass
                        logits, sigmoid = model(specs)
                        loss = criterion(logits, tags)

                        loss_val += loss.item()

                        # gather the predictions in rank 0
                        if not args.distributed_validation:
                            tags_list = [torch.zeros_like(tags) for _ in range(args.local_world_size)]
                            sigmoid_list = [torch.zeros_like(sigmoid) for _ in range(args.local_world_size)]

                            # ideally we could use `gather` as we just need
                            # the data in the rank 0 but it's not implemented
                            # in our backend (NCCL) yet.
                            dist.all_gather(tags_list, tags)
                            dist.all_gather(sigmoid_list, sigmoid)

                            if rank == 0:
                                for t, s in zip(tags_list, sigmoid_list):
                                    y_true_val.append(t.cpu().detach().numpy())
                                    y_pred_val.append(s.cpu().detach().numpy())
                        else:
                            y_true_val.append(tags.cpu().detach().numpy())
                            y_pred_val.append(sigmoid.cpu().detach().numpy())

                if args.distributed_validation or rank == 0:
                    y_true_val = np.vstack(y_true_val)
                    y_pred_val = np.vstack(y_pred_val)
                    logger.debug(f"val tags shape {y_true_val.shape}")
                    logger.debug(f"val preds shape {y_pred_val.shape}")

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
            if loss_val <= best_loss_vas or epoch == 0:
                best_loss_vas = loss_val
                torch.save(model.state_dict(), str(checkpoint_path))
                logger.debug('lowest valiation loss achieved. Updating the model!')

        logger.debug('waiting on barrier to reload the model')
        dist.barrier()
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        model.load_state_dict(
            torch.load(str(checkpoint_path), map_location=map_location))
        logger.debug('reloaded best model')
        logger.debug('epoch finished')

    logger.debug('cleaning up process group')
    dist.destroy_process_group()
    logger.debug('done!')
    tb_logger.close()


if __name__ == '__main__':
    parser = configargparse.ArgumentParser(default_config_files=['config.ini'])

    parser.add('-c', '--config-file', is_config_file=True, help='config file path')

    # exp config
    parser.add('--exp-name', help='the experiment name')
    parser.add('--exp-base-dir', help='the experiment directory')
    parser.add('--timestamp', help='experiment timestamp',
               default=datetime.now().strftime("%y%m%d-%H%M%S"))
    parser.add('--logging-level', default='INFO', help='console logging level')
    # parser.add('--train', action='store_true')

    # distributed config
    parser.add('--distributed-validation', action='store_true')
    parser.add('--n-nodes', default=1, type=int)
    parser.add('--node-rank', default=0, type=int)
    parser.add('--local_rank', default=0, type=int)
    parser.add('--nproc-per-node', default=1, type=int,
               help='number of processes per node')

    # data config
    parser.add('--root', help='root data directory')
    parser.add('--train-pickle', help='pickle file with the training indices')
    parser.add('--val-pickle', help='pickle file with the validation indices')
    parser.add('--just-one-batch', action='store_true')
    parser.add('--train-sampling-strategy', help='train sampling strategy')
    parser.add('--val-sampling-strategy', help='val sampling strategy')

    # model config
    parser.add('--model-name')
    parser.add('--x-size', type=int)
    parser.add('--y-size', type=int)

    # train config
    parser.add('--seed', help='seed number')
    parser.add('--epochs', type=int, help='number of total epochs to run')
    parser.add('--learning-rate', type=float, help='initial learning rate')
    parser.add('--train-batch-size', type=int, help='train batch size')
    parser.add('--val-batch-size', type=int, help='val batch size')

    args, _ = parser.parse_known_args()

    train(args)
