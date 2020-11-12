#!/bin/sh

eval "$(conda shell.bash hook)"
conda activate pt

python melspectrogram_batch.py --jobs 8 /mnt/projects/discotube/discotube-2020-09/audio/ /scratch/palonso/data/discotube/discotube-specs
