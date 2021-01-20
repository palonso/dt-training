from pathlib import Path
import os
from collections import OrderedDict
from argparse import ArgumentParser
from argparse import Namespace
import json

import torch
import numpy as np
from tqdm import tqdm

from modelfactory import ModelFactory

MODEL_OUTPUT = {}
def set_hook():
    def hook(model, input, output):
        MODEL_OUTPUT['model_output'] = output.detach()
    return hook


def predict(args):
    in_dir = args.in_dir
    out_dir = args.out_dir
    checkpoint_path = args.checkpoint_path
    config_file = args.config_file
    layer_name = args.layer_name
    patch_hop_size = args.patch_hop_size
    compress = args.compress
    dry_run = args.dry_run
    extension = args.extension
    force = args.force

    with open(config_file, 'r') as f:
        conf = json.load(f)
    conf = Namespace(**conf)

    print(f'from "{in_dir}" to "{out_dir}"')
    print(f'extracting layer "{layer_name}"')

    state_dict = torch.load(checkpoint_path, map_location={'cuda:%d' % 0: 'cuda:%d' % 0})
    model = ModelFactory().create(conf.model_name, *conf.model_args, **conf.model_kwargs)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()

    # set hook to the layer of interest
    eval(f'model.{layer_name}').register_forward_hook(set_hook())

    # Find files to process
    files_to_process = []
    files_processed = []
    for root, dirs, files in os.walk(in_dir):
        rel_root = Path(root).relative_to(in_dir)
        for file in files:
            if file.endswith(f".{extension}"):
                files_to_process.append(Path(root, file))
                files_processed.append(Path(out_dir, rel_root, file))

    if not files_to_process:
        print('no files to process found. Skipping!')
    else:
        print(f'processing ({len(files_to_process)}) files')

    if not patch_hop_size:
        patch_hop_size = conf.y_size

    pbar = tqdm(files_to_process)
    for src, tgt in zip(files_to_process, files_processed):
        # print(f'{src} -> {tgt}')
        if not dry_run:
            if not tgt.exists() or force:
                tgt.parent.mkdir(parents=True, exist_ok=True)

                # read data
                fp = np.memmap(src, dtype='float16', mode='r')
                melspectrogram = np.array(fp, dtype='float32').reshape(-1, conf.y_size)
                del fp

                if compress:
                    melspectrogram = np.log10(10000 * melspectrogram + 1)

                # batchize
                npatches = int(np.ceil((melspectrogram.shape[0] - conf.x_size) / patch_hop_size) + 1)
                batch = np.zeros([npatches, conf.x_size, conf.y_size])
                for i in range(npatches):
                    last_frame = min(i * patch_hop_size + conf.x_size, melspectrogram.shape[0])
                    actual_size = last_frame - i * patch_hop_size
                    batch[i, :actual_size] = melspectrogram[i * patch_hop_size: last_frame]
                    if actual_size < conf.x_size:
                        print(f'batch {i}/{npatches} with {actual_size} samples')

                batch = torch.Tensor(batch)

                _ = model(batch)
                output = MODEL_OUTPUT['model_output'].cpu().numpy().astype('float16')

                fp = np.memmap(tgt, dtype='float16', mode='w+', shape=output.shape)
                fp[:] = output[:]
                del fp

        pbar.update()
    pbar.close()
    print('done!')


if __name__ == "__main__":
    parser = ArgumentParser('Script for batch inference from precomputed melbands.')
    parser.add_argument('in_dir', help='input base folder with the mmap spectrograms')
    parser.add_argument('out_dir', help='output base folder where to store the outputs')
    parser.add_argument('checkpoint_path', help='Pytorch checkpoint')
    parser.add_argument('config_file', help='model configuration file')
    parser.add_argument('layer_name', help='layer to retrieve from the model')
    parser.add_argument('--extension', default='mmap', help='extension of the mmap melspectrograms')
    parser.add_argument('--patch-hop-size', default=0, type=int, help='path hop size. Use 0 to avoid overlap')
    parser.add_argument('--compress', action='store_true', help='whether to apply log compression to the melspectrograms')
    parser.add_argument('--dry-run', action='store_true', help='dry run')
    parser.add_argument('--force', action='store_true', help='whether to recompute the outputs if they already exist')

    predict(parser.parse_args())
