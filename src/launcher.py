import os
import subprocess
import sys
from datetime import datetime

import configargparse


def launcher(args):
    timestamp = datetime.now().strftime("%y%m%d-%H%M%S")

    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    os.environ['NCCL_SOCKET_IFRAME'] = args.nccl_socket_iframe
    os.environ['NCCL_IB_DISABLE'] = args.nccl_ib_disable
    
    cmd = [
            sys.executable, '-m',
            'torch.distributed.launch',
            '--nnodes', str(args.n_nodes),
            '--node_rank', str(args.node_rank),
            '--nproc_per_node', str(args.nproc_per_node),
            '--master_addr', args.master_addr,
            '--master_port', args.master_port,
            'train.py',
            '--timestamp', timestamp,
        ]

    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    parser = configargparse.ArgumentParser(default_config_files=['config.ini'])

    parser.add('-c', '--config-file', is_config_file=True, help='config file path')
    parser.add('--master-addr', help='address of the master node')
    parser.add('--master-port', help='port of the master node')
    parser.add('--nccl-socket-iframe', help='NCCL socket iframe')
    parser.add('--nccl-ib-disable', help='whether to use Infiniband interface')
    parser.add('--n-nodes', default=1, type=int, help='number of nodes (machines) to use')
    parser.add('--node-rank', default=0, type=int, help='ranking within the nodes')
    parser.add('--nproc-per-node', default=1, type=int, help='number of processes per node')


    args, _ = parser.parse_known_args()

    launcher(args)
