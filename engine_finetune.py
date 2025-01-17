import torch
import time
import pickle
import os

import numpy as np

#from scipy.ndimage import binary_closing, binary_opening

import torch.nn.functional as F
import torch.distributed as dist

from monai.data import decollate_batch

from src.utils.misc import AverageMeter, save_checkpoint, reduce_tensor


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

    for idx, batch_data in enumerate(loader):
        data, target = batch_data['image'].to(device), \
            batch_data['label'].to(device)

        optimizer.zero_grad()
        
        #print("2")
        # mix-precision
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
            logits = model(data)
            loss = loss_func(logits, target)
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        
        # gradient clipping
        if config.MODEL.CLIP_GRAD:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MODEL.CLIP_GRAD)

        scaler.step(optimizer)
        scaler.update()
        loss = reduce_tensor(loss)
        run_loss.update(loss.item(), n=batch_size)
        logger.info(f"Epoch {epoch+1}/{max_epochs} {idx+1}/{len(loader)}, loss: {run_loss.avg}, \
                time {time.time() - start_time}s")
        start_time = time.time()
        torch.cuda.empty_cache()

    return run_loss.avg


def val_epoch(
    config,
    model,
    loader,
    epoch,
    max_epochs,
    acc_func,
    model_inferer=None,
    post_pred=None,
    post_label=None,
    logger=None,
    device=None,
    use_amp=False,
    scaler=None,
):
    model.eval()
    start_time = time.time()

    num_classes = config.MODEL.NUM_CLASSES-1
    run_accs = [AverageMeter() for _ in range(num_classes)]
    
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data, target = batch_data['image'].to(device), \
                batch_data['label'].to(device)
                
            # mix-precision
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                logits = model_inferer(data)
                
            # calculate dice
            val_outputs = decollate_batch(logits)
            val_labels = decollate_batch(target)
            
            val_outputs = [post_pred(val_output.cpu()) for val_output in val_outputs]
            val_labels = [post_label(val_labels.cpu()) for val_labels in val_labels]

            acc_func.reset()
            acc_func(y_pred=val_outputs, y=val_labels)
            acc, not_nans = acc_func.aggregate()
            acc = acc.to(device)
            acc = reduce_tensor(acc)

            for i in range(num_classes):
                run_accs[i].update(acc[i].cpu().numpy(), n=not_nans.cpu().numpy())

            dices = [run_accs[i].avg[0] for i in range(num_classes)]
            
            logger.info(
                f"Val {epoch+1}/{max_epochs} {idx+1}/{len(loader)}, dice: {dices}, \
                time {time.time() - start_time}s")
            start_time = time.time()
            torch.cuda.empty_cache()

    return dices


def test_epoch(
    config,
    model,
    loader,
    acc_func,
    model_inferer=None,
    post_pred=None,
    post_label=None,
    logger=None,
    device=None,
    use_amp=False,
    scaler=None,
):
    import statistics
    model.eval()
    start_time = time.time()
    num_classes = config.MODEL.NUM_CLASSES-1
    run_accs = [AverageMeter() for _ in range(num_classes)]
    list_accs = [[] for _ in range(num_classes)]
    
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data, target, name = batch_data['image'].to(device), \
                batch_data['label'].to(device), \
                batch_data['name'][0]
                
            # mix-precision
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                logits = model_inferer(data)

            # calculate dice
            test_outputs = decollate_batch(logits)
            test_labels = decollate_batch(target)

            test_outputs = [post_pred(test_output.cpu()) for test_output in test_outputs]
            test_labels = [post_label(test_labels.cpu()) for test_labels in test_labels]

            acc_func.reset()
            acc_func(y_pred=test_outputs, y=test_labels)
            acc, not_nans = acc_func.aggregate()
            acc = acc.to(device)
            acc = reduce_tensor(acc)

            for i in range(num_classes):
                run_accs[i].update(acc[i].cpu().numpy(), n=not_nans.cpu().numpy())
                list_accs[i].append(acc[i].cpu().item())

            cur_dices = [acc[i].cpu().item() for i in range(num_classes)]

            logger.info(
                f"Test {idx+1}/{len(loader)}, dice: {cur_dices}, \
                time {time.time() - start_time}s")
            start_time = time.time()

            # save data
            test_inputs, test_labels = data, target
            
            test_inputs_np = test_inputs.squeeze().cpu().numpy()
            test_labels_np = test_labels.squeeze().cpu().numpy()
            logits_np = F.softmax(logits.squeeze(), dim=0).cpu().numpy()

            widget_state = {
                'dices': cur_dices,
                'inputs_np': test_inputs_np,
                'labels_np': test_labels_np,
                'logits_np': logits_np,
            }

            base_path = "./test_output_OAI_binary_scratch"
            f_name = f"{name}.pkl"
            save_path = os.path.join(base_path, f_name)

            logger.info(f"output path: {save_path}")

            with open(save_path, 'wb') as f:
                pickle.dump(widget_state, f)

    dices = [run_accs[i].avg[0] for i in range(num_classes)]

    for i in range(num_classes):
        print(f"{i}: {list_accs[i]}")

    dices_std = [statistics.stdev(list_accs[i]) for i in range(num_classes)]
    print(f"std: {dices_std}")

    return dices


def trainer(
    config,
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    scheduler,
    model_inferer=None,
    start_epoch=0,
    max_epochs=100,
    val_every=10,
    post_pred=None,
    post_label=None,
    logger=None,
    device=None,
    wandb_run=None,
):
    batch_size = config.DATA.BATCH_SIZE
    use_amp = config.AMP_ENABLE
    val_acc_max = 0.0
    dices = []
    dices_avg = []
    loss_epochs = []
    trains_epoch = []
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(start_epoch, max_epochs):
        logger.info(f"Epoch: {epoch+1}")
        epoch_time = time.time()
        train_loss = train_epoch(
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
            best_acc=val_acc_max,
            dir_add=config.MODEL.DIR,
            filename="latest_" + config.MODEL.SAVE_NAME,
            logger=logger,
        )

        if (epoch + 1) % val_every == 0 or epoch == 0:
            loss_epochs.append(train_loss)
            trains_epoch.append(int(epoch+1))
            epoch_time = time.time()
            val_acc = val_epoch(
                config,
                model,
                val_loader,
                epoch=epoch,
                max_epochs=max_epochs,
                acc_func=acc_func,
                model_inferer=model_inferer,
                post_pred=post_pred,
                post_label=post_label,
                logger=logger,
                device=device,
                use_amp=use_amp,
                scaler=scaler,
            )

            val_avg_acc = sum(val_acc) / len(val_acc) 
            
            logger.info(
                f"Final validation stats {epoch+1}/{max_epochs} \
                Dice: {val_acc}, Dice_Avg: {val_avg_acc}, \
                time: {time.time() - epoch_time}s"
            )
            
            if wandb_run != None and dist.get_rank() == 0:
                # per class dice
                for i in range(len(val_acc)):
                    wandb_run.log({f'Validation Dice {i+1}': val_acc[i]})
                # average dice
                wandb_run.log({'Validation Average Dice': val_avg_acc})
            
            dices.append(val_acc)
            dices_avg.append(val_avg_acc)

            if val_avg_acc > val_acc_max:
                logger.info(f"new best ({val_acc_max} --> {val_avg_acc}). ")
                val_acc_max = val_avg_acc
                save_checkpoint(
                    model,
                    epoch,
                    optimizer,
                    scheduler,
                    best_acc=val_acc_max,
                    dir_add=config.MODEL.DIR,
                    filename="best_" + config.MODEL.SAVE_NAME,
                    logger=logger,
                )

    logger.info(f"Training Finished !, Best Accuracy: {val_acc_max}")
    
    return (
        val_acc_max,
        dices,
        dices_avg,
        loss_epochs,
        trains_epoch,
    )


def tester(
    config,
    model,
    test_loader,
    acc_func,
    model_inferer=None,
    post_pred=None,
    post_label=None,
    logger=None,
    device=None,
    wandb_run=None,
):
    use_amp = config.AMP_ENABLE
    test_acc_max = 0.0
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    test_acc = test_epoch(
        config,
        model,
        test_loader,
        acc_func=acc_func,
        model_inferer=model_inferer,
        post_pred=post_pred,
        post_label=post_label,
        logger=logger,
        device=device,
        use_amp=use_amp,
        scaler=scaler,
    )

    test_acc = sum(test_acc) / len(test_acc) 

    print(f"test_acc: {test_acc}")

    test_acc_max = test_acc
    
    if wandb_run != None:
        wandb_run.log({'Test Dice': test_acc_max})
    
    return test_acc_max