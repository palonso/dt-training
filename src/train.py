import os
from datetime import datetime

import configargparse

from trainmanager import TrainManager


def train(args, unknown_args):
    # the following environment variables have
    # to be set when running in standalone.
    # when runnig though pytorch `launch.py` the
    # should be already set.
    if not os.environ.get('MASTER_ADDR'):
        os.environ['MASTER_ADDR'] = args.master_addr
    if not os.environ.get('MASTER_PORT'):
        os.environ['MASTER_PORT'] = args.master_port
    if not os.environ.get('RANK'):
        os.environ['RANK'] = '0'
    if not os.environ.get('LOCAL_RANK'):
        os.environ['LOCAL_RANK'] = '0'
    if not os.environ.get('WORLD_SIZE'):
        os.environ['WORLD_SIZE'] = '1'

    # additional options
    os.environ['NCCL_SOCKET_IFRAME'] = args.nccl_socket_iframe
    os.environ['NCCL_IB_DISABLE'] = args.nccl_ib_disable

    train_manager = TrainManager(unknown_args)
    train_manager.run()


if __name__ == '__main__':
    parser = configargparse.ArgumentParser(default_config_files=['config.ini'])

    parser.add('-c', '--config-file', is_config_file=True, help='config file path')

    parser.add('--master-addr', help='address of the master node')
    parser.add('--master-port', help='port of the master node')
    parser.add('--nccl-socket-iframe', help='NCCL socket iframe')
    parser.add('--nccl-ib-disable', help='whether to use Infiniband interface')

    train(*parser.parse_known_args())
