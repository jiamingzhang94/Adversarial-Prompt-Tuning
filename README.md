# ECCV 2024: Adversarial Prompt Tuning for Vision-Language Models (AdvPT)

> **🎉 [2026.03] News:** Our extended journal version **"NAP-Tuning: Neural Augmented Prompt Tuning for Adversarially Robust Vision-Language Models"** has been accepted by **IEEE TPAMI 2026**! 
> 
> 🔥 The new repository with enhanced methods, updated code, and better performance is now available here: **[jiamingzhang94/nap-tuing](https://github.com/jiamingzhang94/nap-tuing)**.

*Note: This repository contains the original code for our ECCV 2024 conference paper (AdvPT). For our latest extended work with stronger adversarial robustness (NAP-Tuning, TPAMI 2026), please visit the [NAP-Tuning repository](https://github.com/jiamingzhang94/nap-tuing).*

Welcome to the official code repository for the paper "Adversarial Prompt Tuning for Vision-Language Models". Our work introduces AdvPT, an innovative approach that leverages learnable text prompts and aligns them with adversarial image embeddings. This method aims to address the vulnerabilities inherent in Vision-Language Models (VLMs) without the necessity of extensive parameter training or modifying the existing model architecture.


## Step 1: Environment Setup
To configure the environment, follow the installation guide from [CoOp](https://github.com/KaiyangZhou/CoOp#how-to-install).

## Step 2: Data Preparation
We have copied the data download script from CoOp. You can follow the steps in [DATASETS.md](https://github.com/jiamingzhang94/Adversarial-Prompt-Tuning/blob/master/DATASETS.md) to prepare the datasets on `YOUR_DATA` directory.

## Step 3: Adversarial Prompt Tuning
### Complete training and PGD evaluation
Use the script:
```bash
./scripts/main.sh
```
You can modify this script to suit your needs. It calls `train.py`, with most parameters documented in the code. 

### White-box attack evaluation
If you want to evaluate a model checkpoint under PGD-40 attack, e.g., on `Flowers` dataset and `ResNet50` model, use:
```bash
python train.py --root YOUR_DATA --trainer AdvPT --dataset-config-file configs/datasets/oxford_flowers.yaml --config-file configs/trainers/AdvPT/rn50.yaml --output-dir YOUR_OUTPUT --model-dir YOUR_WEIGHTS --eval-only
```

### Black-box attack evaluation
If you want a black-box evaluation, e.g., RAP attack, use:
```bash
python black.py --root YOUR_DATA --dataset OxfordFlowers
python train.py --root YOUR_DATA --trainer AdvPT --dataset-config-file configs/datasets/oxford_flowers.yaml --config-file configs/trainers/AdvPT/rn50.yaml --output-dir YOUR_OUTPUT --model-dir YOUR_WEIGHTS --eval-black --black-attack RAP
```

## Citation
If you find this work useful, please cite:

**Journal Version (TPAMI 2026):**
```bibtex
@article{zhang2026naptuning,
  title={NAP-Tuning: Neural Augmented Prompt Tuning for Adversarially Robust Vision-Language Models},
  author={Zhang, Jiaming and Wang, Xin and Ma, Xingjun and Qiu, Lingyu and Jiang, Yu-Gang and Sang, Jitao},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2026}
}
```

**Conference Version (ECCV 2024):**
```bibtex
@inproceedings{zhang2024adversarial, 
  title={Adversarial Prompt Tuning for Vision-Language Models},
  author={Zhang, Jiaming and Ma, Xingjun and Wang, Xin and Qiu, Lingyu and Wang, Jiaqi and Jiang, Yu-Gang and Sang, Jitao},
  booktitle={European Conference on Computer Vision},
  year={2024}
}
```
