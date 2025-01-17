import os
import pandas as pd

from src.data.transforms import *

from monai import data
from functools import partial

import torch
import torch.distributed as dist

from transformers import AutoTokenizer

from src.utils.misc import create_dataset

def collate_fn_batch_encoding(batch, tokenizer, text_len):
    images, texts = [], []
    for sample in batch:
        images.append(sample[0]['image'])
        texts.append(sample[0]['report'][0])
    
    images = torch.cat(images, dim=0)
    images = images.unsqueeze(dim=1) # add channel dim
    # print(f"texts: {len(texts)}")
    # print("-------------------------------------")
    #images, texts = zip(*batch)
    #image_encodings, text_encodings = images, texts
    # combine images

    # combine texts

    #print(list(texts))

    text_encodings = tokenizer(
            list(texts),
            max_length=text_len,
            padding="max_length",
            truncation=True,
            return_special_tokens_mask=False,
            return_tensors="pt")
    
    #print(f"text_encodings: {text_encodings}")
    #text_encodings.append(text_encoding)

    image_encodings = images
    
    return image_encodings, text_encodings

############################
# Fine-Tuning
############################

def get_dataloaders(config, augs=[], filter=False):
    imtrans, imvals, imtests = augs[0], augs[1], augs[2]
    
    batch_size = config.DATA.BATCH_SIZE
    cache_dir = config.DATA.CACHE_DIR
    seg_type = config.DATA.SEG_TYPE
    
    # Load data
    df_train, df_val, df_test = pd.read_csv(config.DATA.TRAIN_CSV_PATH), \
        pd.read_csv(config.DATA.VAL_CSV_PATH), \
        pd.read_csv(config.DATA.TEST_CSV_PATH)
    
    if filter==True:
        #df_test = df_test[df_test['gender']=='M']
        df_test = df_test
        #df_test = df_test[(df_test['weight']>70) & (df_test['weight']<90)]
        #df_test = df_test[(df_test['age']>30) & (df_test['age']<50)]
        #df_test = df_test[df_test['sex']=='M']
    #df_test = df_test[df_test['race']=='White']
    #df_test = df_test[df_test['age']<=70]

    #df_test = df_test[(df_test['BMI']>28) & (df_test['BMI']<34)]
        
    # df_train = df_train[df_train['dataset']=='TBRecon']
    # df_val = df_val[df_val['dataset']=='TBRecon']
    # df_test = df_test[df_test['dataset']=='TBRecon']

    if 'ssl' in config.MODE:
        img_train, dataset_train = list(df_train['img_path']), list(df_train['dataset'])
        img_val, dataset_val = list(df_val['img_path']), list(df_val['dataset'])
        img_test, dataset_test = list(df_test['img_path']), list(df_test['dataset'])
    elif 'mm' in config.MODE:
        img_train, report_train = list(df_train['img_path']), list(df_train['report'])
        img_val, report_val = list(df_train['img_path']), list(df_train['report'])
        img_test, report_test = list(df_train['img_path']), list(df_train['report'])
    elif 'extract' in config.MODE:
        img_train = list(df_train['img_path'])
        img_val = list(df_val['img_path'])
        img_test = list(df_test['img_path']) 
    else:
        img_train, mask_train = list(df_train['img_path']), list(df_train[f'{seg_type}_path'])
        img_val, mask_val = list(df_val['img_path']), list(df_val[f'{seg_type}_path'])
        img_test, mask_test = list(df_test['img_path']), list(df_test[f'{seg_type}_path'])

    base_path = config.DATA.BASE_PATH

    if 'ssl' in config.MODE:
        base_path_oai = '/data/mskacquisition/howard_temp/OAI_data'
        base_path_tbrecon = '/data/mskacquisition/howard_temp/TBRecon_data'

        img_train = [os.path.join(base_path_oai, path) if dataset == 'OAI' \
                     else os.path.join(base_path_tbrecon, path) \
                        for path, dataset in zip(img_train, dataset_train)]
        img_val = [os.path.join(base_path_oai, path) if dataset == 'OAI' \
                     else os.path.join(base_path_tbrecon, path) \
                        for path, dataset in zip(img_val, dataset_val)]
        img_test = [os.path.join(base_path_oai, path) if dataset == 'OAI' \
                     else os.path.join(base_path_tbrecon, path) \
                        for path, dataset in zip(img_test, dataset_test)]

        train_files = create_dataset(img_train, None, None)
        val_files = create_dataset(img_val, None, None)
        test_files = create_dataset(img_test, None, None)
    elif 'mm' in config.MODE:
        img_train = [os.path.join(base_path, path) for path in img_train]
        img_val = [os.path.join(base_path, path) for path in img_val]
        img_test = [os.path.join(base_path, path) for path in img_test]

        train_files = create_dataset(img_train, None, report_train)
        val_files = create_dataset(img_val, None, report_val)
        test_files = create_dataset(img_test, None, report_test)
    elif 'extract' in config.MODE:
        img_train = [os.path.join(base_path, path) for path in img_train]
        img_val = [os.path.join(base_path, path) for path in img_val]
        img_test = [os.path.join(base_path, path) for path in img_test]

        train_files = create_dataset(img_train, None, None)
        val_files = create_dataset(img_val, None, None)
        test_files = create_dataset(img_test, None, None)
    else:
        img_train, mask_train = [os.path.join(base_path, path) for path in img_train], \
            [os.path.join(base_path, path) for path in mask_train]
        img_val, mask_val = [os.path.join(base_path, path) for path in img_val], \
            [os.path.join(base_path, path) for path in mask_val]
        img_test, mask_test = [os.path.join(base_path, path) for path in img_test], \
            [os.path.join(base_path, path) for path in mask_test]

        train_files = create_dataset(img_train, mask_train, None)
        val_files = create_dataset(img_val, mask_val, None)
        test_files = create_dataset(img_test, mask_test, None)

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    
    if config.MODE == 'mm_pretrain':
        collate_fn = collate_fn_batch_encoding
        token_model = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        tokenizer = AutoTokenizer.from_pretrained(token_model, use_fast=True)
    # Create Dataloaders
    # Train
    # train_ds = data.Dataset(
    #     data=train_files, 
    #     transform=imtrans, 
    # )
    train_ds = data.PersistentDataset(
        data=train_files, 
        transform=imtrans, 
        cache_dir=cache_dir,
    )
    sampler_train = torch.utils.data.distributed.DistributedSampler(
        train_ds,
        shuffle=True,
        num_replicas=num_tasks,
        rank=global_rank,
    )
    train_loader = data.ThreadDataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler_train,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        # collate_fn=partial(collate_fn, tokenizer=tokenizer, text_len=config.MM.TEXT_LEN) \
        #     if config.MODE == 'mm_pretrain' else None,
    )
    
    # Validate
    # val_ds = data.Dataset(
    #     data=val_files, 
    #     transform=imvals, 
    # )
    val_ds = data.PersistentDataset(
        data=val_files, 
        transform=imvals, 
        cache_dir=cache_dir, 
    )
    sampler_val = torch.utils.data.distributed.DistributedSampler(
        dataset=val_ds,
        shuffle=False,
        num_replicas=num_tasks,
        rank=global_rank,
    )
    val_loader = data.ThreadDataLoader(
        dataset=val_ds,
        batch_size=1,
        sampler=sampler_val,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        # collate_fn=partial(collate_fn, tokenizer=tokenizer, text_len=config.MM.TEXT_LEN) \
        #     if config.MODE == 'mm_pretrain' else None,
    )
    
    # Test
    # test_ds = data.Dataset(
    #     data=test_files, 
    #     transform=imtests, 
    # )
    test_ds = data.PersistentDataset(
        data=test_files, 
        transform=imtests, 
        cache_dir=cache_dir, 
    )
    sampler_test = torch.utils.data.distributed.DistributedSampler(
        dataset=test_ds,
        shuffle=False,
        num_replicas=num_tasks,
        rank=global_rank,
    )
    test_loader = data.ThreadDataLoader(
        dataset=test_ds,
        batch_size=1,
        sampler=sampler_test,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        # collate_fn=partial(collate_fn, tokenizer=tokenizer, text_len=config.MM.TEXT_LEN) \
        #     if config.MODE == 'mm_pretrain' else None,
    )
    
    return train_loader, val_loader, test_loader
