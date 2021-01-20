import sys
from datetime import datetime
import configargparse
from argparse import Namespace
from pathlib import Path
import logging
import random
import yaml

import numpy as np
import torch.onnx
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn import metrics
from tqdm import tqdm

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
        super().save_config()

        self.logger = logging.getLogger('TrainManager.VanillaTrainer')
        self.logger.info(f"{conf}")

        # put common variables into the main class scope
        self.epochs = self.conf.epochs

        # validation logic
        if self.conf.distributed_val:
            self.logger.debug('using distributed validation')
        self.val_inference = not self.conf.just_one_batch
        self.val_scoring = self.val_inference and (self.i_am_chief or self.conf.distributed_val)

        # use a different seed in each rank so
        # the tey do not intialize to the same values
        torch.manual_seed(self.conf.seed + self.rank)
        np.random.seed(self.conf.seed + self.rank)
        random.seed(self.conf.seed + self.rank)

        # required for multi-gpu deterministic behavior
        # torch.set_deterministic(True)
        # os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'

        # define and initliaze model
        self.__define_model()
        self.__model_initialization()
        self.__define_dataloaders()

        # define loss function (criterion) and optimizer
        self.criterion = nn.MultiLabelSoftMarginLoss().cuda(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.conf.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=self.conf.lr_decay,
            patience=self.conf.lr_patience, verbose=self.i_am_chief)

    def __define_model(self):
        model = ModelFactory().create(self.conf.model_name, *self.conf.model_args, **self.conf.model_kwargs)
        torch.cuda.set_device(self.device)
        model.cuda(self.device)

        # compute number of parameters
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info(f'number of trainable params: {params}')

        # create a sigmoid layer to transform logits into activations
        self.sigmoid_layer = nn.Sigmoid()

        # wrap the model
        self.model = nn.parallel.DistributedDataParallel(model,
            device_ids=[self.device],
            output_device=self.device)

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
            self.val_steps = len(self.val_iterator)
            self.logger.info(f'number of validation steps per epoch: {self.val_steps}')

    def __train(self, compute_predictions=False):
        loss_list, y_true, y_pred = [], [], []

        # stdout is enought for the progress bar. Don't log this
        if self.i_am_chief:
            pbar = tqdm(self.train_iterator, desc='train iter')
        for i, sample in enumerate(self.train_iterator):
            specs = sample['melspectrogram'].cuda(non_blocking=True)
            tags = sample['tags'].cuda(non_blocking=True)

            # forward pass
            logits = self.model(specs)
            loss = self.criterion(logits, tags)

            # backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_list.append(loss.item())

            if compute_predictions:
                sigmoid = self.sigmoid_layer(logits)
                y_true.append(tags.cpu().detach().numpy())
                y_pred.append(sigmoid.cpu().detach().numpy())

            if self.i_am_chief:
                pbar.update()

        if self.i_am_chief:
            pbar.close()

        if compute_predictions:
            y_true = np.vstack(y_true)
            y_pred = np.vstack(y_pred)

        loss = np.mean(loss_list)
        return loss, y_true, y_pred

    def __validate(self, compute_predictions=False):
        loss_list, y_true, y_pred = [], [], []
        sigmoid_list, tags_list, keys_list = [], [], []
        unique_keys = set()

        with torch.no_grad():
            if self.i_am_chief:
                pbar = tqdm(self.val_iterator, desc='valid iter')
            for i, sample in enumerate(self.val_iterator):
                specs = sample['melspectrogram'].cuda(non_blocking=True)
                tags = sample['tags'].cuda(non_blocking=True)
                keys = sample['key']

                # forward pass
                logits = self.model(specs)
                loss = self.criterion(logits, tags)

                loss_list.append(loss.item())

                if compute_predictions:
                    sigmoid = self.sigmoid_layer(logits)
                    sigmoid_list.append(sigmoid)
                    tags_list.append(tags)
                    keys_list.append(keys)
                    unique_keys.update(set(keys))

                if self.i_am_chief:
                    pbar.update()

            if self.i_am_chief:
                pbar.close()

        if compute_predictions:
            # the amount of tracks among classes can deviate. give some margin.
            max_n_tracks = self.conf.tracks_per_rank + 3
            n_tracks =len(unique_keys)

            sigmoid = torch.cat(sigmoid_list, dim=0).cuda(self.device)
            tags = torch.cat(tags_list, dim=0).cuda(self.device)
            keys = np.hstack(keys_list)

            tracks_sigmoid = torch.zeros(max_n_tracks, self.conf.n_classes, dtype=torch.float32).cuda(self.device)
            tracks_tags = torch.zeros(max_n_tracks, self.conf.n_classes, dtype=torch.float32).cuda(self.device)

            indices = np.zeros([n_tracks, len(keys)]).astype(bool)
            key, idx = keys[0], 0
            for j, k in enumerate(keys):
                if k == key:
                    indices[idx, j] = True
                else:
                    key = k
                    idx += 1

            for j in range(n_tracks):
                track_indices = torch.nonzero(torch.tensor(indices[j])).squeeze().cuda(self.device)

                if track_indices.dim():
                    track_sigmoid = torch.index_select(sigmoid, 0, track_indices)

                    tracks_sigmoid[j, :] = torch.mean(track_sigmoid, dim=0)
                    tracks_tags[j, :] = tags[track_indices[0].item(), :]
                else:
                    self.logger.debug(f"track ({j}) without indices")

            # free GPU memory
            del sigmoid
            del tags
            torch.cuda.empty_cache()

            # gather the predictions in rank 0
            if not self.conf.distributed_val:
                tag_list = [torch.zeros(max_n_tracks, self.conf.n_classes).cuda(self.device) for i in range(self.conf.local_world_size)]
                sigmoid_list = [torch.zeros(max_n_tracks, self.conf.n_classes).cuda(self.device) for i in range(self.conf.local_world_size)]

                # all reduce tensors must have an uniform shape.
                # ideally we could use `gather` as we just need
                # the data in the chief but it is not implemented
                # in our backend (NCCL) yet so we are using `all_gather`
                # which makes them avaiable in al the ranks.
                dist.all_gather(tag_list, tracks_tags)
                dist.all_gather(sigmoid_list, tracks_sigmoid)

                if self.i_am_chief:
                    y_true = torch.cat(tag_list, dim=0).cpu().detach().numpy()
                    y_pred = torch.cat(sigmoid_list, dim=0).cpu().detach().numpy()

                    valid_entries = np.where(y_true.any(axis=1))[0]

                    y_true = y_true[valid_entries]
                    y_pred = y_pred[valid_entries]
            else:
                y_true = tracks_tags.cpu().detach().numpy()
                y_pred = tracks_sigmoid.cpu().detach().numpy()


        self.logger.debug(f"val tags size {len(y_true)}")
        self.logger.debug(f"val preds size {len(y_pred)}")

        loss = np.mean(loss_list)
        return loss, y_true, y_pred

    def __compute_metrics(self, y_true, y_pred):
        roc_auc = metrics.roc_auc_score(y_true, y_pred, average='macro')
        pr_auc = metrics.average_precision_score(y_true, y_pred, average='macro')
        return roc_auc, pr_auc

    def __export_to_onnx(self):
        dummy_input = torch.randn(1, self.conf.x_size, self.conf.y_size, device='cuda')
        model_with_sigmoid = nn.Sequential(self.model.module, nn.Sigmoid())
        torch.onnx.export(model_with_sigmoid, dummy_input,
                          self.checkpoint_path.with_suffix('.onnx'),
                          # verbose=True,
                          opset_version=11,
                          do_constant_folding=True,
                          input_names=['melspectrogram'],
                          output_names=['activations'],
                          dynamic_axes={'melspectrogram': {0: 'batch_size'},
                                        'activations': {0: 'batch_size'}}
                          )

    def run(self):
        self.logger.debug('entering the training loop')

        start = datetime.now()
        best_loss_val, loss_val = np.Inf, np.Inf
        stats = {'Loss': [], 'AUC/ROC': [], 'AUC/PR': [], 'LR': []}

        for epoch in range(self.epochs):
            self.logger.debug(f'starting epoch {epoch + 1}')

            compute_metrics = not bool(epoch % self.conf.metric_rate)

            self.logger.debug(f'compute metrics: {compute_metrics}')

            self.logger.debug('training phase')
            loss_train, y_true_train, y_pred_train = self.__train(compute_predictions=compute_metrics)
            if compute_metrics:
                roc_auc_train, pr_auc_train = self.__compute_metrics(y_true_train, y_pred_train)

            if self.val_inference:
                self.logger.debug('validation phase')
                loss_val, y_true_val, y_pred_val = self.__validate(compute_predictions=compute_metrics)
                self.scheduler.step(loss_val)

                if self.val_scoring and compute_metrics:
                    roc_auc_val, pr_auc_val = self.__compute_metrics(y_true_val, y_pred_val)

            elapsed = str(datetime.now() - start)
            epoch_log = [
                f'Epoch [{epoch + 1:03}/{self.epochs:03}]',
                f'Time : {elapsed}',
                f'Train Loss: {loss_train:.3f}',
            ]

            if compute_metrics:
                epoch_log.append(f'Train ROC AUC: {roc_auc_train:.3f}')
                epoch_log.append(f'Train AP AUC: {pr_auc_train:.3f}')

            if self.val_inference:
                stats['Loss'].append({'train': loss_train, 'val': loss_val})
                epoch_log.append(f'Val Loss: {loss_val:.3f}')
            else:
                stats['Loss'].append({'train': loss_train})

            if compute_metrics:
                if self.val_scoring:
                    stats['AUC/ROC'].append({'train': roc_auc_train, 'val': roc_auc_val})
                    stats['AUC/PR'].append({'train': pr_auc_train, 'val': pr_auc_val})
                    epoch_log.append(f'Val ROC AUC: {roc_auc_val:.3f}')
                    epoch_log.append(f'Val PR AUC: {pr_auc_val:.3f}')
                else:
                    stats['AUC/ROC'].append({'train': roc_auc_train})
                    stats['AUC/PR'].append({'train': pr_auc_train})

            stats['LR'].append({'lr': self.optimizer.param_groups[0]['lr']})

            self.logger.info(' | '.join(epoch_log))
            self.tensorboard.write_epoch_stats(epoch, stats)

            # save model on the chief model
            if self.i_am_chief:
                if (loss_val <= best_loss_val) or (epoch == 0):
                    best_loss_val = loss_val
                    torch.save(self.model.state_dict(), self.checkpoint_path)
                    self.__export_to_onnx()
                    self.logger.info('lowest validation loss achieved. Updating the model!')

            self.logger.debug('waiting on barrier to reload the model')
            dist.barrier()
            map_location = {'cuda:%d' % self.conf.gpu_shift: 'cuda:%d' % self.device}
            self.model.load_state_dict(
                torch.load(self.checkpoint_path, map_location=map_location))

            if epoch == self.conf.weight_decay_delay:
                self.logger.info(f'setting weight decay to {self.conf.weight_decay}')
                self.optimizer.param_groups[0]['weight_decay'] = self.conf.weight_decay

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
            const=True, default=False, help="activate `just-one-batch`")
        parser.add('--train-sampling-strategy', help='train sampling strategy')
        parser.add('--val-sampling-strategy', help='val sampling strategy')
        parser.add('--min-track-duration', help='minimum track duration in seconds', type=float)
        parser.add('--model-name')
        parser.add('--model-args', action='append', default=[])
        parser.add('--model-kwargs', type=yaml.safe_load)
        parser.add('--x-size', type=int, help='mel band frames')
        parser.add('--y-size', type=int, help='mel band bins')
        parser.add('--n-classes', type=int, help='number of classes')
        parser.add('--sample-rate', type=int, help='audio analysis sample rate')
        parser.add('--hop-size', type=int, help='audio analysis hop size')
        parser.add('--frame-size', type=int, help='audio analysis frame size')
        parser.add('--patch-hop-size', type=int, help='hop size between adjacent patches for validation')
        parser.add('--evaluation-time', type=float, help='max time for evalution in seconds')
        parser.add('--seed', help='seed number', type=int)
        parser.add('--epochs', type=int, help='number of total epochs to run')
        parser.add('--lr', type=float, help='initial learning rate')
        parser.add('--lr-decay', type=float, help='learning rate decay')
        parser.add('--lr-patience', type=int, help='learning rate patience')
        parser.add('--weight-decay', type=float, help='weight decay')
        parser.add('--weight-decay-delay', type=int, help='number of epochs before applying weight decay')
        parser.add('--train-batch-size', type=int, help='train batch size')
        parser.add('--val-batch-size', type=int, help='val batch size')
        parser.add('--metric-rate', type=int, help='number of epochs to wait to update the AUC metrics')

        args, unknown_args = parser.parse_known_args(arg_line)
        return args, unknown_args
