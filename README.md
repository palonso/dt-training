# dt-training

Scripts for training models based on the DiscoTube Dataset (WIP). It utilizes Pytorch's [Distritubed Data Parallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) and supports one or multiple GPUs (devices) in one or several machines (nodes).

## Preprocessing
`src/preprocessing` contains scripts to compute the mel-spectrograms used as input. This folder also contains Jupyter Notebooks to create training, validation, and testing splits for 400 or 500 DiscoTube Style Tags.  

## Configuration
All experiments are configured through [config files](src/config.ini) via `config_argparse`

## Training
Experiments can be launched in two ways (from `src/`):
1. For single GPU experiments use:

```bash
python train.py -c config.ini
```

2. For multi-GPU or multi-node experiments use:

```bash
python launch.py -c config.ini
```
The latter is just a wrapper of Pytorch's [launch script](https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py) that spawns up `train.py` on each device/node.

## Validation
Once a model is trained, it can be validated in the testing split with `validate.py`

```
usage: Script for validation [-h] [--experiment-folder EXPERIMENT_FOLDER]
                             [--test-pickle TEST_PICKLE]
                             [--label-binarizer LABEL_BINARIZER]
                             [--patch-hop-size PATCH_HOP_SIZE] [--top-n TOP_N]

optional arguments:
  -h, --help            show this help message and exit
  --experiment-folder EXPERIMENT_FOLDER
                        experiment folder (as created by train.py)
  --test-pickle TEST_PICKLE
                        pickle file with the indices to test (as created by the preprocessing notebooks)
  --label-binarizer LABEL_BINARIZER
                        label binarizer to retrieve label names 
  --patch-hop-size PATCH_HOP_SIZE
                        path hop size. Use 0 to avoid overlap (as created by the preprocessing notebooks). Default = 0
  --top-n TOP_N         top_n classification results. Default = 5
```
