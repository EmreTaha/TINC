# TINC: Temporally Informed Non-Contrastive Learning for Disease Progression Modeling in Retinal OCT Volumes
This repository is the official Pytorch implementation of MICCAI 2022 paper [TINC: Temporally Informed Non-Contrastive Learning for Disease Progression Modeling in Retinal OCT Volumes](https://arxiv.org/abs/2206.15282) by Taha Emre, Arunava Chakravarty, Antoine Rivail, Sophie Riedl, Ursula Schmidt-Erfurth and Hrvoje Bogunović.

TINC is a non-contrastive method, that uses temporal information between OCT volumes acquired at different times. Modified epsilon-insensitive loss is used instead of commonly used MSE loss for similarity, captures this temporal information implicitly. Later on, we showed that this information is beneficial in predicting disease stage progression using OCT volumes.

## Requirements
- Torch
- Torchvision
- Numpy
- Pandas

## Dataset

 [Harbor](https://clinicaltrials.gov/ct2/show/NCT00891735) dataset is used for self-supervised pre-training and supervised evaluation. Unfortunately we cannot release the dataset.

Make sure that your folder structure is similar to ours. Such as .../pat_id/visit_day/Bscan_number.png. Visit day is in days. Such as 000, 030 ... 720.

## Training

### Self-supervised training
this script is specifically for TINC, but we included model and loss definitions that are used in testing

    python src/pretrain.py --save_dir=./saved_models --norm_label --max_iters=400 --warmup_iters=10 --lin_iters=50 --optim=AdamW --batch_size=256 --lr=3e-3 --grad_norm_clip --exclude_nb
    
## Citation

Please consider citing TINC paper if it is useful for you:

```
@inproceedings{emre2022tinc,
    title={TINC: Temporally Informed Non-Contrastive Learning for Disease Progression Modeling in Retinal OCT Volumes},
    author={Emre, Taha and Chakravarty, Arunava and Rivail, Antoine and Riedl, Sophie and Schmidt-Erfurth, Ursula and Bogunović, Hrvoje},
    booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
    year={2022},
    organization={Springer}
}
```