import os
import json
import logging
from pathlib import Path
from datetime import datetime

import configargparse
import torch
import torch.distributed as dist

import utils
from trainerfactory import TrainerFactory


class TrainManager:
    def __init__(self, arg_line):
        self.conf, unknown_args = self.__parse_args(arg_line)

        self.conf.exp_id = f"{self.conf.timestamp}_{self.conf.exp_name}"

        # confgure the experiment dir
        self.exp_dir = Path(self.conf.exp_base_dir, self.conf.exp_id)
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.conf.exp_dir = str(self.exp_dir)

        # put common variables into the class namespace
        self.rank = self.conf.local_rank
        self.i_am_chief = self.rank == 0
        self.conf.device = self.rank + self.conf.gpu_shift

        # local_world_size is as more familiar name
        self.conf.local_world_size = self.conf.nproc_per_node

        # start loggers
        self.logger = utils.get_logger('TrainManager', self.exp_dir / f"rank_{self.rank}.log")
        if self.i_am_chief:
            self.__chief_to_console()
        self.__log_config()

        self.logger.debug('initiating process group...')
        dist.init_process_group(backend='nccl',
                                init_method='env://')
        self.logger.debug('ok!')

        self.trainer = TrainerFactory().create(self.conf.trainer,
                                               self.conf,
                                               arg_line=unknown_args)

    def __chief_to_console(self):
        console_handler = utils.get_console_handler(self.conf.logging_level)
        self.logger.addHandler(console_handler)
        self.logger.debug(f'console logging level: {self.conf.logging_level}')

    def __log_config(self):
        self.logger.info(f'starting experiment "{self.conf.exp_id}"')
        self.logger.debug(f'experiment folder: "{self.conf.exp_dir}"')
        self.logger.debug(f'experiment configuration: {vars(self.conf)}')
        self.logger.debug(f'pid: {os.getpid()}')

        env_dict = { key: os.environ.get(key)
            for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK")
            }
        self.logger.debug(f"initializing process group with: {env_dict}")

        with open(str(self.exp_dir / f"config_{self.rank}.json"), "w" ) as f:
            json.dump(vars(self.conf), f, indent=4)

    def __delete__(self):
        self.tensorboard.close()

        self.logger.debug('cleaning up process group')
        dist.destroy_process_group()
        self.logger.debug('done!')

    @staticmethod
    def __parse_args(arg_line):
        parser = configargparse.ArgumentParser()

        parser.add('--exp-name', help='experiment name')
        parser.add('--exp-base-dir', help='experiment directory')
        parser.add('--timestamp', help='experiment timestamp',
            default=datetime.now().strftime("%y%m%d-%H%M%S"))
        parser.add('--logging-level', help='console logging level',
            default='INFO')
        parser.add('--node-rank', default=0, type=int)
        parser.add('--local_rank', default=0, type=int)
        parser.add('--gpu-shift', default=0, type=int)
        parser.add('--nproc-per-node', help='number of processes per node',
            default=1, type=int)
        parser.add('--trainer', default='VanillaTrainer', help='trainer type')

        args, unknown_args = parser.parse_known_args(arg_line)
        return args, unknown_args

    def run(self):
        self.trainer.run()
