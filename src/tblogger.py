from torch.utils.tensorboard import SummaryWriter


class TBLogger:
    def __init__(self, log_dir, **args):
        self.writer = SummaryWriter(log_dir=log_dir, **args)

    def write_epoch_stats(self, n_iter, d):
        for k, v in d.items():
            if v:
                self.writer.add_scalars(k, v[-1], n_iter)

    def close(self):
        self.writer.close()
