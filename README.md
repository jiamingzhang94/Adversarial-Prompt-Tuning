
# ECCV 2024: Adversarial Prompt Tuning for Vision-Language Models (AdvPT)

Welcome to the official code repository for the paper "Adversarial Prompt Tuning for Vision-Language Models". Our work introduces AdvPT, an innovative approach that leverages learnable text prompts and aligns them with adversarial image embeddings. This method aims to address the vulnerabilities inherent in Vision-Language Models (VLMs) without the necessity of extensive parameter training or modifying the existing model architecture.


## Step 1: Environment Setup
To configure the environment, follow the installation guide from [CoOp](https://github.com/KaiyangZhou/CoOp#how-to-install).

## Step 2: Data Preparation
We have copied the data download script from CoOp. You can follow the steps in [DATASETS.md](https://github.com/jiamingzhang94/Adversarial-Prompt-Tuning/blob/master/DATASETS.md) to prepare the datasets.

## Step 3: Adversarial Prompt Tuning
For complete training and evaluation, use the script:
```bash
./scripts/main.sh
```
You can modify this script to suit your needs. It calls `train.py`, with most parameters documented in the code. If you already have a model checkpoint, use:
```bash
python train.py --root YOUR_DATA --trainer AdvPT --dataset-config-file configs/datasets/oxford_flowers.yaml --config-file configs/trainers/AdvPT/rn50.yaml --output-dir YOUR_OUTPUT --model-dir YOUR_WEIGHTS --eval-only
```

## Citation
If you find this work useful, please cite:
```bibtex
@inproceedings{zhang2024adversarial, 
  title={Adversarial Prompt Tuning for Vision-Language Models},
  author={Zhang, Jiaming and Ma, Xingjun and Wang, Xin and Qiu, Lingyu and Wang, Jiaqi and Jiang, Yu-Gang and Sang, Jitao},
  booktitle={European Conference on Computer Vision},
  year={2024}
}
```

