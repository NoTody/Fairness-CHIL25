import torch
import time
import pickle

import torch.distributed as dist

from src.utils.ops import aug_rand, rot_rand
from src.utils.misc import replace_nan_gradients_hook


def pretrain_one_step(
    config,
    data,
    model, 
    use_amp,
):

    # mix-precision
    with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
        rot_p, contrastive_p, rec_x = model(data)

    return rot_p, contrastive_p, rec_x

def train_epoch(
    config,
    model, 
    loader, 
    logger=None, 
    device=None, 
    use_amp=False,
    scaler=None,
):
    model.eval()
    start_time = time.time()

    tensor_list = []

    for _ in range(5):
        for idx, batch_data in enumerate(loader):
            data = batch_data['image'].to(device)

            _, contrastive_p, _ = pretrain_one_step(
                config,
                data,
                model, 
                use_amp,
            )

            contrastive_p = contrastive_p.detach().cpu()
            tensor_list.append(contrastive_p)

            logger.info(f"{idx+1}/{len(loader)}, time {time.time() - start_time}s")
            start_time = time.time()

    final_tensor = torch.stack(tensor_list, dim=0).numpy()

    with open('./embedding_pickles/tbrecon_scratch_pretrain.pkl', 'wb') as f:
        pickle.dump(final_tensor, f)

    print(f"Tensor saved with shape {final_tensor.shape}")


def trainer(
    config,
    model,
    train_loader,
    logger=None,
    device=None,
    wandb_run=None,
):
    use_amp = config.AMP_ENABLE
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    epoch_time = time.time()
    train_epoch(
        config,
        model,
        train_loader,
        logger=logger,
        device=device,
        use_amp=use_amp,
        scaler=scaler,
    )
    logger.info(
        f"Final training, time {time.time() - epoch_time}s"
    )

    return
