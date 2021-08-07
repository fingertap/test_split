import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from test_split.loss import LinearFocalLoss, RegressionLoss
from test_split.model import BucketModel, RegressionModel
from test_split.data import MyDataset
from test_split.engine import train_epoch, val_epoch, info

parser = argparse.ArgumentParser()
parser.add_argument('data', help='path to info.csv file')
parser.add_argument('-b', dest='batch_size', type=int)
parser.add_argument('-m', dest='model', default='bucket',
                    choices=['bucket', 'regression'],
                    help='which kind of model to use')
parser.add_argument('-n', dest='nb', type=int, default=100,
                    help='number of pixel buckets')
parser.add_argument('-p', dest='parallel', action='store_true',
                    help='parallel compute')
parser.add_argument('--vf', '--validate-fraction',
                    dest='vf', type=float, default=0.2)
parser.add_argument('--epochs', type=int, default=100)


def main_worker(device, args, world_size):
    if args.parallel:
        info(f'Start initialize process group at {device}')
        dist.init_process_group(
            rank=device,
            backend='nccl',
            world_size=world_size,
            init_method='file:///tmp/asdfaosdifjaxc'
        )
    info(f'Start training at GPU {device}.')
    do_train(device, args)
    if args.parallel:
        dist.destroy_process_group()


def do_train(device, args):
    train_data, val_data = MyDataset.read_csv(args.data)\
                                    .train_test_split(args.vf)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=DistributedSampler(train_data) if args.parallel else None
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        sampler=DistributedSampler(val_data) if args.parallel else None
    )

    if args.model == 'bucket':
        model = BucketModel().to(device)
        criterion = LinearFocalLoss().to(device)
        # criterion = nn.CrossEntropyLoss().to(device)
    else:
        model = RegressionModel().to(device)
        criterion = RegressionLoss().to(device)
    if args.parallel:
        model = DDP(model, device_ids=[device])

    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(args.epochs):
        train_epoch(
            data=train_loader,
            model=model,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch
        )
        val_epoch(
            data=val_loader,
            model=model,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch
        )


def main(args):
    if args.parallel:
        world_size = torch.cuda.device_count()  # We use a single machine
        mp.spawn(
            main_worker,
            nprocs=world_size,
            args=(args, world_size),
            join=True
        )
    else:
        main_worker(0, args, 1)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
