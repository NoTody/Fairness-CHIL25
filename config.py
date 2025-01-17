import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data Settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU
_C.DATA.BATCH_SIZE = 8
# Base path
_C.DATA.BASE_PATH = '/home/data/TBRecon_data'
# Train csv path
_C.DATA.TRAIN_CSV_PATH = '/data/TBRecon/dataset/train.csv'
# Val csv path
_C.DATA.VAL_CSV_PATH = '/data/TBRecon/dataset/val.csv'
# Test csv path
_C.DATA.TEST_CSV_PATH = '/data/TBRecon/dataset/test.csv'
# Cache directory
_C.DATA.CACHE_DIR = '/data/TBRecon_data/TBRecon_cache_dir'
# Pin memory
_C.DATA.PIN_MEMORY = True
# Number of workers
_C.DATA.NUM_WORKERS = 4
# Cache rate
_C.DATA.CACHE_RATE = 1.0
# Crop rate
_C.DATA.CROP_RATE = 0.5
# Segmentation type
_C.DATA.SEG_TYPE = 'bone'

# -----------------------------------------------------------------------------
# Model Settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.BACKBONE  = 'unet'
# Pretrained model path to load
_C.MODEL.PRETRAINED = None
# Model save directoy
_C.MODEL.DIR = '/home/model_saved'
# Model save name
_C.MODEL.SAVE_NAME = 'debug.pt'
# Region of Interest
_C.MODEL.ROI = [96, 96, 96]
# Shift range
_C.MODEL.SHIFT_RANGE = [20, 20, 20]
# Spacial dims
_C.MODEL.SPATIAL_DIMS = 3
# Number of Labels
_C.MODEL.NUM_CLASSES = 2
# Number of samples
_C.MODEL.NUM_SAMPLES = 4
# Sliding windows batch size
_C.MODEL.SW_BATCH_SIZE = 2
# Sliding windows overlap
_C.MODEL.INFER_OVERLAP = 0.5
# Use epoch number from loaded checkpoint
_C.MODEL.USE_LOAD_EPOCH = False
# Clip gradients
_C.MODEL.CLIP_GRAD = 1.0

# -----------------------------------------------------------------------------
# Training Settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
# Epoch
_C.TRAIN.MAX_EPOCHS = 200
# Warmup Epoch
_C.TRAIN.WARMUP_EPOCHS = 20
# Validation every N epochs
_C.TRAIN.VAL_EVERY = 10
# Base learning rate
_C.TRAIN.BASE_LR = 1e-4
# Minimum learning rate
_C.TRAIN.MIN_LR = 1e-8
# Weight decay
_C.TRAIN.WEIGHT_DECAY = 1e-5
# Loss type
_C.TRAIN.LOSS = 'dice'

# -----------------------------------------------------------------------------
# SSL Settings
# -----------------------------------------------------------------------------
_C.SSL = CN()
_C.SSL.EMBEDDING_DIM = 768

# -----------------------------------------------------------------------------
# MM Settings
# -----------------------------------------------------------------------------
_C.MM = CN()
_C.MM.TEXT_LEN = 128
_C.MM.TEMPERATURE = 0.1
_C.MM.EMBEDDING_DIM = 768
_C.MM.PROJECTION_DIM = 768
_C.MM.POOL = 'cls'

# -----------------------------------------------------------------------------
# Logging Settings
# -----------------------------------------------------------------------------
_C.LOG = CN()
# Logging file save name
_C.LOG.FILENAME = 'unet3d_TBRecon'

# -----------------------------------------------------------------------------
# wandb Settings
# -----------------------------------------------------------------------------
_C.WANDB = CN()
# enable wandb
_C.WANDB.WANDB_ENABLE = False
# wandb project name
_C.WANDB.PROJECT = 'unet3d_TBRecon'

# -----------------------------------------------------------------------------
# Misc Settings
# -----------------------------------------------------------------------------
# Training Mode (select from 'ssl_pretrain', 'mm_pretrain', 'finetune')
_C.MODE = 'ssl_pretrain'
# Seed to ensure reproducibility
_C.SEED = 42
# Enable Pytorch automatic mixed precision
_C.AMP_ENABLE = False
# local rank for distributed training
_C.LOCAL_RANK = 0
# Path to output folder
_C.OUTPUT = ''
# Tag of experiment
_C.TAG = 'default'

def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    def _check_args(name):
        if hasattr(args, name) and eval(f'args.{name}'):
            return True
        return False

    # merge from specific arguments
    if _check_args('batch_size'):
        config.DATA.BATCH_SIZE = args.batch_size
    if _check_args('num_workers'):
        config.DATA.NUM_WORKERS = args.num_workers
    if _check_args('seg_type'):
        config.DATA.SEG_TYPE = args.seg_type
    if _check_args('train_csv_path'):
        config.DATA.TRAIN_CSV_PATH = args.train_csv_path
    if _check_args('val_csv_path'):
        config.DATA.VAL_CSV_PATH = args.val_csv_path
    if _check_args('test_csv_path'):
        config.DATA.TEST_CSV_PATH = args.test_csv_path
    if _check_args('loss'):
        config.TRAIN.LOSS = args.loss
    if _check_args('max_epochs'):
        config.TRAIN.MAX_EPOCHS = args.max_epochs
    if _check_args('base_lr'):
        config.BASE_LR = args.base_lr
    if _check_args('min_lr'):
        config.MIN_LR = args.min_lr
    if _check_args('weight_decay'):
        config.WEIGHT_DECAY = args.weight_decay
    if _check_args('seed'):
        config.SEED = args.seed
    if _check_args('use_amp'):
        config.AMP_ENABLE = args.use_amp
    if _check_args('use_wandb'):
        config.WANDB.WANDB_ENABLE = args.use_wandb
    if _check_args('wandb_project'):
        config.WANDB.PROJECT = args.wandb_project
    if _check_args('backbone'):
        config.MODEL.BACKBONE = args.backbone
    if _check_args('model_load_path'):
        config.MODEL.PRETRAINED = args.model_load_path

    # set local rank for distributed training
    config.LOCAL_RANK = args.local_rank

    # output folder
    #config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.TYPE, config.TAG)
    config.OUTPUT = os.path.join(config.OUTPUT)

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config
