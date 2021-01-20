import logging
from pathlib import Path
import sys
import json

sys.path.append("..")
from tblogger import TBLogger


class Trainer:
    def __init__(self, conf):
        self.conf = conf
        self.logger = logging.getLogger('TrainManager.Trainer')

        # put common variables into the main class scope
        self.rank = self.conf.local_rank
        self.device = self.conf.device
        self.checkpoint_path = Path(self.conf.exp_dir, "model.checkpoint")

        self.i_am_chief = self.rank == 0

        self.tensorboard = TBLogger(log_dir=str(Path(self.conf.exp_dir, "tb_logs", f"rank_{self.rank}")))

    def save_config(self):
        with open(Path(self.conf.exp_dir, f"config_{self.rank}.json"), "w" ) as f:
            json.dump(vars(self.conf), f, indent=4)
