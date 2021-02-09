from pathlib import Path
import os
from collections import OrderedDict
from argparse import ArgumentParser
from argparse import Namespace
import json

import torch
import numpy as np
from tqdm import tqdm
from torch import nn
import joblib

from modelfactory import ModelFactory
from pathlib import Path
from dataloader import DataLoader

from sklearn import metrics 


def predict(args):
    experiment_folder = Path(args.experiment_folder)
    test_pickle = args.test_pickle
    patch_hop_size = args.patch_hop_size
    label_binarizer = args.label_binarizer
    device = 0
    config_file = experiment_folder / 'config_0.json'
    checkpoint_path = experiment_folder / 'model.checkpoint'
    distributed_val = False
    top_n = args.top_n

    with open(config_file, 'r') as f:
        conf = json.load(f)
    conf = Namespace(**conf)
    # don't split the dataset
    conf.local_world_size = 1

    encoder = joblib.load(label_binarizer)
    labels = encoder.classes_

    state_dict = torch.load(checkpoint_path, map_location={'cuda:%d' % 0: 'cuda:%d' % 0})
    model = ModelFactory().create(conf.model_name, *conf.model_args, **conf.model_kwargs)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    
    torch.cuda.set_device(device)
    model.cuda(device)
    model.eval()

    # get files to process
    test_iterator = DataLoader(test_pickle, conf, mode='val')
    test_steps = len(test_iterator)

    y_true, y_pred = [], []
    sigmoid_list, tags_list, keys_list = [], [], []
    unique_keys = set()

    sigmoid_layer = nn.Sigmoid()

    pbar = tqdm(test_iterator, desc='valid iter')
    with torch.no_grad():
        for i, sample in enumerate(test_iterator):
            specs = sample['melspectrogram'].cuda(non_blocking=True)
            tags = sample['tags'].cuda(non_blocking=True)
            keys = sample['key']

            # forward pass
            logits = model(specs)

            sigmoid = sigmoid_layer(logits)
            sigmoid_list.append(sigmoid)
            tags_list.append(tags)
            keys_list.append(keys)
            unique_keys.update(set(keys))
            pbar.update()
    pbar.close()

    n_tracks =len(unique_keys)
    print('ntrachs: ', n_tracks)

    sigmoid = torch.cat(sigmoid_list, dim=0).cuda(device)
    tags = torch.cat(tags_list, dim=0).cuda(device)
    keys = np.hstack(keys_list)

    tracks_sigmoid = torch.zeros(n_tracks, conf.n_classes, dtype=torch.float32).cuda(device)
    tracks_tags = torch.zeros(n_tracks, conf.n_classes, dtype=torch.float32).cuda(device)

    indices = np.zeros([n_tracks, len(keys)], dtype=bool)
    key, idx = keys[0], 0
    for j, k in enumerate(keys):
        if k == key:
            indices[idx, j] = True
        else:
            key = k
            idx += 1

    for j in range(n_tracks):
        track_indices = torch.nonzero(torch.tensor(indices[j])).squeeze().cuda(device)

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
    if distributed_val:
        pass
    else:
        y_true = tracks_tags.cpu().detach().numpy()
        y_pred = tracks_sigmoid.cpu().detach().numpy()

    print(f"test tags size {y_true.shape}")
    print(f"test preds size {y_pred.shape}")

    results = [f'Metrics for the test set for experiment "{conf.exp_name}"\n']
    results.append(f'Test set: "{test_pickle}"\n')

    roc_auc = metrics.roc_auc_score(y_true, y_pred, average='macro')
    pr_auc = metrics.average_precision_score(y_true, y_pred, average='macro')
    results.append(f'ROC AUC: {roc_auc}')
    results.append(f'PR AUC: {pr_auc}\n')

    indices = np.argsort(y_pred, axis=1)[:, -top_n:]
    y_pred_top1 = np.zeros(y_pred.shape, dtype=int)
    np.put_along_axis(y_pred_top1, indices, 1, 1)

    y_true = y_true.astype('int')

    results.append(f"top-({top_n}) classification results")
    report = metrics.classification_report(y_true, y_pred_top1, target_names=labels)
    results.append(report)

    confussion_matrix = metrics.multilabel_confusion_matrix(y_true, y_pred_top1)
    cm_lines = ['\n\nMultilabel (1 vs all) confussion matrices\n']
    for i,label in enumerate(labels):
        cm_lines.append(label)
        cm_lines.append(np.array_str(confussion_matrix[i], max_line_width=None, precision=3, suppress_small=None))
    results += cm_lines

    with open(experiment_folder / 'test_set_results', 'w') as f:
        f.write('\n'.join(results))


if __name__ == "__main__":
    parser = ArgumentParser('Script for validation')

    parser.add_argument('--experiment-folder', help='experiment folder (as created by train.py)')
    parser.add_argument('--test-pickle', help='pickle file with the indices to test (as created by the preprocessing notebooks)')
    parser.add_argument('--label-binarizer', help='label binarizer to retrieve label names (as created by the preprocessing notebooks)')
    parser.add_argument('--patch-hop-size', default=0, type=int, help='path hop size. Use 0 to avoid overlap')
    parser.add_argument('--top-n', help='top_n classification results', type=int, default=5)

    predict(parser.parse_args())
