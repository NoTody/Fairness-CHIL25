import os
import json
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist

from src.models.unet import UNet3D
from src.models.ssl_head import SSLHead

from monai.networks.layers import Norm, trunc_normal_
from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT
from monai.networks.nets import UNETR, SwinUNETR
from monai.utils import ensure_tuple_rep
from monai.losses import DiceLoss, FocalLoss, DiceCELoss, \
    GeneralizedDiceLoss, GeneralizedDiceFocalLoss


# pack dataset
def create_dataset(images, labels, reports):
    dataset = []
    
    if labels != None:
        for img, label in zip(images, labels):
            sample_dict = dict()
            sample_dict['image'] = [img]
            sample_dict['name'] = img.split('/')[-1]
            sample_dict['label'] = [label]
            dataset.append(sample_dict)
    elif reports != None:
        for img, report in zip(images, reports):
            sample_dict = dict()
            sample_dict['image'] = [img]
            sample_dict['name'] = img.split('/')[-1]
            sample_dict['report'] = [report]
            dataset.append(sample_dict)
    else:
        for img in images:
            sample_dict = dict()
            sample_dict['image'] = [img]
            sample_dict['name'] = img.split('/')[-1]
            dataset.append(sample_dict)

    return dataset


# get model
def get_model(config):
    roi = config.MODEL.ROI
    if config.MODEL.BACKBONE == "unet":
        img_model = UNet3D(
            n_class=config.MODEL.NUM_CLASSES,
        )
    elif config.MODEL.BACKBONE == "unetr":
        img_model = UNETR(
            in_channels=1,
            out_channels=config.MODEL.NUM_CLASSES,
            img_size=roi,
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0,
        )
    elif config.MODEL.BACKBONE == "swinunetr":
        img_model = SwinUNETR(
            img_size=roi,
            in_channels=1,
            out_channels=config.MODEL.NUM_CLASSES,
            feature_size=48,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=True,
        )
    elif config.MODEL.BACKBONE == "ssl_swinunetr":
        img_model = SSLHead(
            config=config,
            swinunetr=True,
        )
    elif config.MODEL.BACKBONE == "ssl_dino":
        img_model = SSLHead(
            config=config,
            swinunetr=False,
        )
    else:
        raise ValueError(f"Backbone {config.MODEL.BACKBONE} not supported")

    return img_model


# load model
def load_model(config, model, logger=None, mode='train'):
    load_epoch = 0
    loaded_state_dict = None
    if config.MODEL.PRETRAINED != None:
        if config.MODEL.BACKBONE == "swinunetr" or "ssl" in config.MODEL.BACKBONE or mode == "test":
            loaded_state_dict = torch.load(config.MODEL.PRETRAINED, map_location=torch.device('cpu'))
            current_model_dict = model.state_dict()
            
            if 'epoch' in loaded_state_dict.keys() and config.MODEL.USE_LOAD_EPOCH:
                load_epoch = loaded_state_dict['epoch']

            if 'state_dict' in loaded_state_dict.keys():
                model_state_dict = loaded_state_dict['state_dict']
            else:
                model_state_dict = loaded_state_dict
            
            new_state_dict={k:v if v.size()==current_model_dict[k].size() else current_model_dict[k] \
                            for k,v in zip(current_model_dict.keys(), model_state_dict.values())}
            
            msg = model.load_state_dict(new_state_dict, strict=False)
            logger.info(f"Load Pretrained Model: {msg} for Achitecture: {config.MODEL.BACKBONE}")
        elif config.MODEL.BACKBONE == "unet":
            loaded_state_dict = torch.load(config.MODEL.PRETRAINED)['net']
            store_dict = model.state_dict()
            amount = 0
            for key in loaded_state_dict.keys():
                new_key = '.'.join(key.split('.')[2:])
                if new_key in store_dict.keys():
                    store_dict[new_key] = loaded_state_dict[key]   
                    amount += 1

            msg = model.load_state_dict(store_dict)
            logger.info(f"Load Pretrained Model: {msg} for Achitecture: {config.MODEL.BACKBONE}")
        else:
            raise ValueError(f"Backbone {config.MODEL.BACKBONE} not supported")

    return load_epoch, loaded_state_dict, model


# load optimizer
def load_optimizer(config, optimizer, scheduler, loaded_state_dict, logger=None):
    # load optimizer/scheduler state
    if config.MODEL.PRETRAINED != None:
        if 'optimizer' in loaded_state_dict.keys():
            optimizer_state = loaded_state_dict['optimizer']
            #print(f"optimizer_state: {optimizer_state}")
            msg = optimizer.load_state_dict(optimizer_state)
            logger.info(f"Load optimizer state: {msg}")

        if 'scheduler' in loaded_state_dict.keys():
            scheduler_state = loaded_state_dict['scheduler']
            #print(f"optimizer_state: {scheduler_state}")
            msg = scheduler.load_state_dict(scheduler_state)
            logger.info(f"Load scheduler state: {msg}")

    return optimizer, scheduler

# get loss
def get_loss(config):
    if config.TRAIN.LOSS == 'dice':
        loss = DiceLoss(to_onehot_y=True, softmax=True)
    elif config.TRAIN.LOSS == 'focal':
        loss = FocalLoss(to_onehot_y=True, softmax=True)
    elif config.TRAIN.LOSS == 'generalized_dice':
        loss = GeneralizedDiceLoss(to_onehot_y=True, softmax=True)
    elif config.TRAIN.LOSS == 'dice_focal':
        loss = GeneralizedDiceFocalLoss(to_onehot_y=True, softmax=True)
        #loss = DiceCELoss(to_onehot_y=True, softmax=True)
    else:
        raise ValueError(f"Loss type {config.TRAIN.LOSS} not supported")

    return loss

# get text embedding
def get_embed(text_encodings, model, pool):
    text_inp = text_encodings['input_ids'].cuda()
    outputs = model(text_inp)
    text_hidden_states = outputs.last_hidden_state
    if pool == 'cls':
        text_embed = text_hidden_states[:, 0, :]
    elif pool == 'full':
        text_embed = text_hidden_states
    elif pool == 'global_pool':
        text_embed = (text_hidden_states * text_encodings['attention_mask'].unsqueeze(-1))[:, 1:, :]
        text_embed = torch.mean(text_embed, dim=1)
    else:
        raise ValueError(f"Pooling type {pool} not supported")

    return text_embed

# save checkpoint
def save_checkpoint(model, epoch, optimizer, scheduler, filename="model.pt", \
        best_acc=0, dir_add=None, logger=None):
    state_dict = model.state_dict()
    optimizer_state = optimizer.state_dict()
    scheduler_state = scheduler.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, \
        "optimizer": optimizer_state, "scheduler": scheduler_state, \
        "state_dict": state_dict}
    filename = os.path.join(dir_add, filename)
    torch.save(save_dict, filename)
    logger.info(f"Saving checkpoint {filename}")


# datafold reader
def datafold_read(datalist, basedir, fold=0, key="training"):
    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[key]

    for d in json_data:
        for k in d:
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]
                
    tr = []
    val = []
    for d in json_data:
        if "fold" in d and d["fold"] == fold:
            val.append(d)
        else:
            tr.append(d)

    return tr, val


# average meter
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


# reduce tensor
def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

# distributed training setup
def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


# distributed training initialization
def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        return

    torch.cuda.set_device(gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        rank, args.dist_url, gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=world_size, rank=rank)
    torch.distributed.barrier()
    setup_for_distributed(rank == 0)


# get rank
def get_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    else:
        return 0

def all_gather(x):
    x_list = FullGatherLayer.apply(x)
    return torch.cat(x_list, dim=0)


class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


# Define a hook function that replaces NaN gradients with a small epsilon value
def replace_nan_gradients_hook(grad):
    epsilon = 1e-6
    if torch.isnan(grad).any():
        grad = torch.where(torch.isnan(grad), torch.full_like(grad, epsilon), grad)
    return grad


def remove_nan_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            eps = 1e-6
            param.data[~torch.isfinite(param.data)] = eps
            param.grad.data[~torch.isfinite(param.grad.data)] = eps
            has_nan = torch.isnan(param.grad.data).any()
            if has_nan:
                print("1")
                eps = 1e-6
                param.grad.data[~torch.isfinite(param.grad.data)] = eps
                # # Use torch.where to replace NaNs with eps
                # param.grad.data = torch.where(torch.isnan(param.grad.data),
                #                               torch.full_like(param.grad.data, eps),
                #                               param.grad.data)


class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self, model, head):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        self.backbone = model
        self.head = head

    def forward(self, x):
        # convert to list
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx, output = 0, torch.empty(0).to(x[0].device)
        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(x[start_idx: end_idx]))
            # The output is a tuple with XCiT model. See:
            # https://github.com/facebookresearch/xcit/blob/master/xcit.py#L404-L405
            if isinstance(_out, tuple):
                _out = _out[0]
            # accumulate outputs
            output = torch.cat((output, _out))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        return self.head(output)


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x
    

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule
