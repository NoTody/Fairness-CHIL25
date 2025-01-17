import torch
import time

import torch.distributed as dist

from src.utils.ops import aug_rand, rot_rand
from src.utils.misc import AverageMeter, save_checkpoint, reduce_tensor, remove_nan_gradients, replace_nan_gradients_hook


def pretrain_one_step(
    config,
    data,
    model, 
    loss_func, 
    use_amp,
):
    x1, rot1 = rot_rand(config, data)
    x2, rot2 = rot_rand(config, data)
    x1_augment = aug_rand(config, x1)
    x2_augment = aug_rand(config, x2)
    x1_augment = x1_augment
    x2_augment = x2_augment
    
    # mix-precision
    with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
        rot1_p, contrastive1_p, rec_x1 = model(x1_augment)
        rot2_p, contrastive2_p, rec_x2 = model(x2_augment)
        rot_p = torch.cat([rot1_p, rot2_p], dim=0)
        rots = torch.cat([rot1, rot2], dim=0)
        imgs_recon = torch.cat([rec_x1, rec_x2], dim=0)
        imgs = torch.cat([x1, x2], dim=0)
        loss, losses_tasks = loss_func(rot_p, rots, contrastive1_p, contrastive2_p, imgs_recon, imgs)
    
    return loss, losses_tasks

def train_epoch(
    config,
    model, 
    batch_size, 
    loader, 
    optimizer, 
    epoch, 
    max_epochs, 
    loss_func, 
    logger=None, 
    device=None, 
    use_amp=False,
    scaler=None,
):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    run_task_loss = [AverageMeter() for _ in range(3)]

    count = 0

    for param in model.parameters():
        param.register_hook(replace_nan_gradients_hook)
        
    for idx, batch_data in enumerate(loader):

        data = batch_data['image'].to(device)

        optimizer.zero_grad()

        #with record_function("## forward ##"):
        loss, losses_tasks = pretrain_one_step(
            config,
            data,
            model, 
            loss_func,
            use_amp,
        )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        # gradient clipping
        if config.MODEL.CLIP_GRAD:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MODEL.CLIP_GRAD)
            
        scaler.step(optimizer)
        scaler.update()

        loss = reduce_tensor(loss)
        run_loss.update(loss.item(), n=batch_size)

        for i in range(len(losses_tasks)):
            loss_task = reduce_tensor(losses_tasks[i])
            run_task_loss[i].update(loss_task.item(), n=batch_size)

        task_losses = [float(run_task_loss[i].avg) for i in range(len(losses_tasks))]

        logger.info(f"Epoch {epoch+1}/{max_epochs} {idx+1}/{len(loader)}, loss: {run_loss.avg}, \
                task_losses: {task_losses}, time {time.time() - start_time}s")
        start_time = time.time()

    return run_loss.avg, task_losses


def val_epoch(
    config,
    model, 
    batch_size, 
    loader, 
    epoch, 
    max_epochs, 
    loss_func, 
    logger=None, 
    device=None, 
    use_amp=False,
    scaler=None,
):
    model.eval()
    start_time = time.time()
    run_loss = AverageMeter()
    run_task_loss = [AverageMeter() for _ in range(3)]

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data = batch_data['image'].to(device)

            loss, losses_tasks = pretrain_one_step(
                config,
                data,
                model, 
                loss_func,
                use_amp,
            )

            loss = reduce_tensor(loss)
            run_loss.update(loss.item(), n=batch_size)

            for i in range(len(losses_tasks)):
                loss_task = reduce_tensor(losses_tasks[i])
                run_task_loss[i].update(loss_task.item(), n=batch_size)

            task_losses = [float(run_task_loss[i].avg) for i in range(len(losses_tasks))]

            logger.info(f"Val Epoch {epoch+1}/{max_epochs} {idx+1}/{len(loader)}, loss: {run_loss.avg}, \
                    task_losses: {task_losses}, time {time.time() - start_time}s")
            start_time = time.time()
            torch.cuda.empty_cache()

    return run_loss.avg, task_losses


def trainer(
    config,
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    scheduler,
    start_epoch=0,
    max_epochs=100,
    val_every=10,
    logger=None,
    device=None,
    wandb_run=None,
):
    batch_size = config.DATA.BATCH_SIZE
    use_amp = config.AMP_ENABLE
    val_loss_min = 1.0e6
    loss_epochs = []
    trains_epoch = []
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    for epoch in range(start_epoch, max_epochs):
        logger.info(f"Epoch: {epoch+1}")
        epoch_time = time.time()
        train_loss, _ = train_epoch(
            config,
            model,
            batch_size,
            train_loader,
            optimizer,
            epoch=epoch,
            max_epochs=max_epochs,
            loss_func=loss_func,
            logger=logger,
            device=device,
            use_amp=use_amp,
            scaler=scaler,
        )
        scheduler.step()
        logger.info(
            f"Final training  {epoch+1}/{max_epochs}, loss: {train_loss}, \
            time {time.time() - epoch_time}s"
        )
        if wandb_run != None and dist.get_rank() == 0:
            wandb_run.log({'Training Loss': train_loss, \
                           'Training lr': optimizer.param_groups[0]["lr"], \
                            'Epoch': epoch+1})

        save_checkpoint(
            model,
            epoch,
            optimizer,
            scheduler,
            best_acc=val_loss_min,
            dir_add=config.MODEL.DIR,
            filename="latest_" + config.MODEL.SAVE_NAME,
            logger=logger,
        )

        if (epoch + 1) % val_every == 0 or epoch == 0:
            loss_epochs.append(train_loss)
            trains_epoch.append(int(epoch+1))
            epoch_time = time.time()
            val_loss, val_loss_tasks = val_epoch(
                config,
                model,
                batch_size,
                val_loader,
                epoch=epoch,
                loss_func=loss_func,
                max_epochs=max_epochs,
                logger=logger,
                device=device,
                use_amp=use_amp,
                scaler=scaler,
            )
            
            logger.info(
                f"Final validation stats {epoch+1}/{max_epochs} \
                Loss: {val_loss}, Loss_Tasks: {val_loss_tasks}, \
                time: {time.time() - epoch_time}s"
            )
            
            if wandb_run != None and dist.get_rank() == 0:
                task_names = ["Rotation", "Contrastive", "Reconstruction"]
                # per class dice
                for i in range(len(val_loss_tasks)):
                    wandb_run.log({f'Validation {task_names[i]} Loss {i+1}': val_loss_tasks[i]})
                # average dice
                wandb_run.log({'Validation Total Loss': val_loss})

            if val_loss < val_loss_min:
                logger.info(f"new best ({val_loss_min} --> {val_loss}). ")
                val_loss_min = val_loss
                save_checkpoint(
                    model,
                    epoch,
                    optimizer,
                    scheduler,
                    best_acc=val_loss_min,
                    dir_add=config.MODEL.DIR,
                    filename="best_" + config.MODEL.SAVE_NAME,
                    logger=logger,
                )

    logger.info(f"Training Finished !, Best Loss: {val_loss_min}")
    
    return (
        val_loss_min,
        loss_epochs,
        trains_epoch,
    )


def tester(
    config,
    model,
    test_loader,
    loss_func,
    logger=None,
    device=None,
    wandb_run=None,
):
    batch_size = config.DATA.BATCH_SIZE
    use_amp = config.AMP_ENABLE
    test_loss_min = 1.0e6
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    epoch, max_epochs = 1, 1

    test_loss, test_loss_tasks = val_epoch(
        config,
        model,
        batch_size,
        test_loader,
        epoch=epoch,
        loss_func=loss_func,
        max_epochs=max_epochs,
        logger=logger,
        device=device,
        use_amp=use_amp,
        scaler=scaler,
    )
    
    if wandb_run != None and dist.get_rank() == 0:
        task_names = ["Rotation", "Contrastive", "Reconstruction"]
        # per class dice
        for i in range(len(test_loss_tasks)):
            wandb_run.log({f'Validation {task_names[i]} Loss {i+1}': test_loss_tasks[i]})
        # average dice
        wandb_run.log({'Validation Total Loss': test_loss})
    
    return test_loss_min