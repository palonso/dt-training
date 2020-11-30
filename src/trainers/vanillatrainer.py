import sys
from datetime import datetime
import configargparse
from argparse import Namespace
from pathlib import Path
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from sklearn import metrics

sys.path.append("..")
import utils
from .trainer import Trainer
from modelfactory import ModelFactory
from dataloader import DataLoader


class VanillaTrainer(Trainer):
    def __init__(self, manager_conf, arg_line=None):
        super().__init__(manager_conf)

        conf, _ = self.__parse_args(arg_line)
        self.conf = Namespace(**vars(conf), **vars(manager_conf))

        self.logger = logging.getLogger('TrainManager.VanillaTrainer')

        # put common variables into the main class scope
        self.learning_rate = self.conf.learning_rate
        self.epochs = self.conf.epochs

        # validation logic
        if self.conf.distributed_val:
            self.logger.debug('using distributed validation')
        self.val_inference = not self.conf.just_one_batch
        self.val_scoring = self.val_inference and (self.i_am_chief or self.conf.distributed_val)

        # use a different seed in each rank so
        # the tey do not intialize to the same values
        torch.manual_seed(self.conf.seed + self.rank)

        # define and initliaze model
        self.__define_model()
        self.__model_initialization()
        self.__define_dataloaders()

        # define loss function (criterion) and optimizer
        self.criterion = nn.MultiLabelSoftMarginLoss().cuda(self.rank)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate)

    def __define_model(self):
        model = ModelFactory().create(self.conf.model_name)
        torch.cuda.set_device(self.rank)
        model.cuda(self.rank)

        # compute number of parameters
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info(f'number of trainable params: {params}')

        # wrap the model
        self.model = nn.parallel.DistributedDataParallel(model,
            device_ids=[self.rank],
            output_device=self.rank)

    def __model_initialization(self):
        for m in self.model.parameters():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias:
                    torch.nn.init.xavier_uniform_(m.bias)

    def __define_dataloaders(self):
        train_iterator = DataLoader(self.conf.train_pickle, self.conf, mode='train')
        if self.conf.just_one_batch:
            self.logger.warning('`just-one-batch` mode on. Intended for development purposes')
            self.train_iterator = [next(iter(train_iterator))]
        else:
            self.train_iterator = train_iterator

        self.train_steps = len(self.train_iterator)
        self.logger.info(f'number of train steps per epoch: {self.train_steps}')

        if self.val_inference:
            self.val_iterator = DataLoader(self.conf.val_pickle, self.conf, mode='val')

    def __train(self):
        loss_list, y_true, y_pred = [], [], []

        for i, sample in enumerate(self.train_iterator):
            specs = sample['melspectrogram'].cuda(non_blocking=True)
            tags = sample['tags'].cuda(non_blocking=True)

            # forward pass
            logits, sigmoid = self.model(specs)
            loss = self.criterion(logits, tags)

            # backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_list.append(loss.item())

            y_true.append(tags.cpu().detach().numpy())
            y_pred.append(sigmoid.cpu().detach().numpy())
            self.logger.info(f'Step [{i + 1}/{self.train_steps}]')

        loss = np.sum(loss_list) / len(self.train_iterator)
        return loss, np.vstack(y_true), np.vstack(y_pred)

    def __validate(self):
        loss_list, y_true, y_pred = [], [], []

        with torch.no_grad():
            for i, sample in enumerate(self.val_iterator):
                specs = sample['melspectrogram'].cuda(non_blocking=True)
                tags = sample['tags'].cuda(non_blocking=True)

                # forward pass
                logits, sigmoid = self.model(specs)
                loss = self.criterion(logits, tags)

                loss_list.append(loss.item())

                # gather the predictions in rank 0
                if not self.conf.distributed_val:
                    tag_list = [torch.zeros_like(tags) for _ in range(self.conf.local_world_size)]
                    sigmoid_list = [torch.zeros_like(sigmoid) for _ in range(self.conf.local_world_size)]

                    # ideally we could use `gather` as we just need
                    # the data in the chief but it is not implemented
                    # in our backend (NCCL) yet so we are using `all_gather`
                    # which makes them avaiable in al the ranks.
                    dist.all_gather(tag_list, tags)
                    dist.all_gather(sigmoid_list, sigmoid)

                    if self.i_am_chief:
                        for t, s in zip(tag_list, sigmoid_list):
                            y_true.append(t.cpu().detach().numpy())
                            y_pred.append(s.cpu().detach().numpy())
                else:
                    y_true.append(tags.cpu().detach().numpy())
                    y_pred.append(sigmoid.cpu().detach().numpy())

        self.logger.debug(f"val tags size {len(y_true)}")
        self.logger.debug(f"val preds size {len(y_pred)}")

        if y_true:
            y_true = np.vstack(y_true)
            y_pred = np.vstack(y_pred)

        loss = np.sum(loss_list) / len(self.val_iterator)
        return loss, y_true, y_pred

    def __compute_metrics(self, y_true, y_pred):
        roc_auc = metrics.roc_auc_score(y_true, y_pred, average='macro')
        pr_auc = metrics.average_precision_score(y_true, y_pred, average='macro')
        return roc_auc, pr_auc

    def run(self):
        self.logger.debug('entering the training loop')

        start = datetime.now()
        best_loss_val, loss_val = np.Inf, np.Inf
        stats = {'Loss': [], 'AUC/ROC': [], 'AUC/PR': []}

        for epoch in range(self.epochs):
            self.logger.debug(f'starting epoch {epoch + 1}')

            loss_train, y_true_train, y_pred_train = self.__train()
            roc_auc_train, pr_auc_train = self.__compute_metrics(y_true_train, y_pred_train)

            if self.val_inference:
                loss_val, y_true_val, y_pred_val = self.__validate()
                if self.val_scoring:
                    roc_auc_val, pr_auc_val = self.__compute_metrics(y_true_val, y_pred_val)

            elapsed = str(datetime.now() - start)
            epoch_log = [
                f'Epoch [{epoch + 1:03}/{self.epochs:03}]',
                f'Time : {elapsed}',
                f'Train Loss: {loss_train:.3f}',
                f'Train ROC AUC: {roc_auc_train:.3f}',
                f'Train AP AUC: {pr_auc_train:.3f}',
            ]

            if self.val_inference:
                stats['Loss'].append({'train': loss_train, 'val': loss_val})
                epoch_log.append(f'Val Loss: {loss_val:.3f}')
            else:
                stats['Loss'].append({'train': loss_train})

            if self.val_scoring:
                stats['AUC/ROC'].append({'train': roc_auc_train, 'val': roc_auc_val})
                stats['AUC/PR'].append({'train': pr_auc_train, 'val': pr_auc_val})
                epoch_log.append(f'Val ROC AUC: {roc_auc_val:.3f}')
                epoch_log.append(f'Val PR AUC: {pr_auc_val:.3f}')
            else:
                stats['AUC/ROC'].append({'train': roc_auc_train})
                stats['AUC/PR'].append({'train': pr_auc_train})

            self.logger.info(' | '.join(epoch_log))
            self.tensorboard.write_epoch_stats(epoch, stats)

            # save model on the chief model
            if self.i_am_chief:
                if (loss_val <= best_loss_val) or (epoch == 0):
                    best_loss_val = loss_val
                    torch.save(self.model.state_dict(), self.checkpoint_path)
                    self.logger.debug('lowest valiation loss achieved. Updating the model!')

            self.logger.debug('waiting on barrier to reload the model')
            dist.barrier()
            map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
            self.model.load_state_dict(
                torch.load(self.checkpoint_path, map_location=map_location))
            self.logger.debug('epoch finished')

    @staticmethod
    def __parse_args(arg_line):
        parser = configargparse.ArgumentParser()

        parser.add('--distributed-val', type=utils.str2bool, nargs='?',
            const=True, default=False, help="activate `distributed-val`")
        parser.add('--root', help='root data directory')
        parser.add('--train-pickle', help='pickle file with the training indices')
        parser.add('--val-pickle', help='pickle file with the validation indices')
        parser.add('--just-one-batch', type=utils.str2bool, nargs='?',
            const=True, default=False, help="activate  `just-one-batch`")
        parser.add('--train-sampling-strategy', help='train sampling strategy')
        parser.add('--val-sampling-strategy', help='val sampling strategy')
        parser.add('--model-name')
        parser.add('--x-size', type=int)
        parser.add('--y-size', type=int)
        parser.add('--seed', help='seed number', type=int)
        parser.add('--epochs', type=int, help='number of total epochs to run')
        parser.add('--learning-rate', type=float, help='initial learning rate')
        parser.add('--train-batch-size', type=int, help='train batch size')
        parser.add('--val-batch-size', type=int, help='val batch size')

        args, unknown_args = parser.parse_known_args(arg_line)
        return args, unknown_args
