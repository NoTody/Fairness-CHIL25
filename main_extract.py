import os
import json
import random
import argparse
import wandb

from logger import create_logger
from config import get_config

from src.utils.misc import *
from src.data.datasets import *

import numpy as np

from monai.config import print_config

from engine_extract import *

import torch
import torch.distributed as dist


print_config()


def parse_option():
    parser = argparse.ArgumentParser('MONAI training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file')
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line",
        default=None,
        nargs='+',
    )

    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')
    parser.add_argument('--dist-backend', default='nccl', help='used to set up distributed backend')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument("--seed", type=int, help='seed')
    parser.add_argument("--use_amp", action='store_true')

    # wandb configs
    parser.add_argument("--use_wandb", action='store_true')
    parser.add_argument("--wandb_project", type=str, default="TBRecon")
    
    # model parameters
    parser.add_argument("--model_load_path", type=str, help='path to trained model')
    parser.add_argument("--loss", type=str, help='loss function')
    parser.add_argument("--base_lr", type=float, help='base learning rate')
    parser.add_argument("--min_lr", type=float, help='minimum learning rate')
    parser.add_argument("--weight_decay", type=float, help='weight decay')
    parser.add_argument("--batch_size", type=int, help='batch size')
    parser.add_argument("--num_workers", type=int, help='number of workers for dataloader')
    parser.add_argument("--max_epochs", type=int, help='max epoch')

    # dataset parameters
    parser.add_argument('--seg_type', type=str, help='segmentation type')
    parser.add_argument('--train_csv_path', type=str, help='path to train csv file')
    parser.add_argument('--val_csv_path', type=str, help='path to val csv file')
    parser.add_argument('--test_csv_path', type=str, help='path to test csv file')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(config, wandb_run):
    # Device setting
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters

    # Get transforms
    imtrans = extract_transforms(config, mode='test')
    imvals = extract_transforms(config, mode='test')
    imtests = extract_transforms(config, mode='test')
    
    # Create data loaders
    train_loader, val_loader, test_loader = \
        get_dataloaders(config, augs=[imtrans, imvals, imtests])

    # Create model
    model = get_model(config).to(device)
    
    # Load model with wrong size weights unloaded
    load_epoch = 0
    load_epoch, loaded_state_dict, model = load_model(config, model, logger)

    # Convert all BatchNorm layers to SyncBatchNorm layers
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # Use DistributedDataParallel for distributed training
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], \
        broadcast_buffers=False, find_unused_parameters=False)

    torch.backends.cudnn.benchmark = True

    trainer(
        config=config,
        model=model,
        train_loader=val_loader,
        logger=logger,
        device=device,
        wandb_run=wandb_run,
    )

    logger.info(f"train completed ...")


def init_seed(seed):
    random_seed = seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


if __name__ == "__main__":
    #torch.autograd.set_detect_anomaly(True)
    args, config = parse_option()
    # init distributed training
    init_distributed_mode(args)
    seed = config.SEED + dist.get_rank()
    init_seed(seed)
    # create logger
    logger = create_logger(output_dir='./log', dist_rank=dist.get_rank(), name=config.LOG.FILENAME)
    # print arguments
    logger.info(f"Arguments: {config}")

    # output config settings
    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, f"{config.LOG.FILENAME}.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    # init wandb
    wandb_run = None
    if config.WANDB.WANDB_ENABLE and dist.get_rank() == 0:
        wandb_run = wandb.init(
                # Set the project where this run will be logged
                name = config.LOG.FILENAME,
                project=config.WANDB.PROJECT,
                # Track hyperparameters and run metadata
                config={
                    "learning_rate": config.TRAIN.BASE_LR,
                    "batch_size": config.DATA.BATCH_SIZE,
                    "epochs": config.TRAIN.MAX_EPOCHS,
                    "backbone": config.MODEL.BACKBONE,
                }
            )

    # run main training
    main(config, wandb_run)
    