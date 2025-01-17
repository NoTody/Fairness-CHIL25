# Fairness-Musculoskeletal Segmentation

This repostiory contain code for pre-training and evaluating Musculoskeletal Imaging Segmentation with self-supervised learning for
model fairness evaluation.

## Dependencies
Check requirements.txt for dependencies of this repository or install them by running
```
pip install -r ./requirements.txt
```

## Train Model for Downstream
### Fine-tuning
Example for fine-tuning with 1 GPU, check
```
./configs/finetune/*.yaml
```
for fine-tuning method for other datasets and pre-defined hyper-parameters
```
torchrun --nnodes 1 --nproc_per_node 1 --master_port 12345 ./main_finetune.py --local_rank 0 \
    --model_name "vit" --num_workers 4 --max_epochs 800 --base_lr 1e-4 \
    --cfg ./configs/finetune/swinunetr_TBRecon_cartilage.yaml --use_amp \
    --optimizer "AdamW" --scheduler "cosine" --weight_decay 0.01 --grad_clip 1.0 \
    --model_load_path <Pre-Trained Model Path Here>
```

## Pre-train Model
### Pre-training
Example for pre-tuning with 1 GPU (our training used 4 GPUs), check
```
./configs/finetune/swinunetr_TBRecon_cartilage.yaml
```
for other pre-defined hyper-parameters
```
torchrun --nnodes 1 --nproc_per_node 1 --master_port 12345 ./main_pretrain_ssl.py --local_rank 0 \
    --model_name "vit" --num_workers 4 --max_epochs 800 --base_lr 1e-4 \
    --cfg ./configs/ssl/swinunetr_TBRecon.yaml --use_amp \
    --optimizer "AdamW" --scheduler "cosine" --weight_decay 0.01 --grad_clip 1.0 \
    --model_load_path <Pre-Trained Model Path Here>
```

## Fairness Evaluation
Evaluate model fairness ''compute_fairness'' function
in 
```
./fariness_eval.py
```
### Usage
Assume having list of some performance scores from model 1 and model 2 with two sub-cohorts (A, B) or three sub-cohorts (A, B, C)
```
compute_fairness(listA1, listA2, listB1, listB2, n_iterations=1000)
```
for two sub-cohorts case
or
```
compute_fairness(listA1, listA2, listB1, listB2, listC1, listC2, n_iterations=1000)
```
for three sub-cohorts case