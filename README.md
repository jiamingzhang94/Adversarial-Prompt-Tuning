
# ECCV 2024: Adversarial Prompt Tuning for Vision-Language Models (AdvPT)

Welcome to the official code repository for the paper "Adversarial Prompt Tuning for Vision-Language Models". Our work introduces AdvPT, an innovative approach that leverages learnable text prompts and aligns them with adversarial image embeddings. This method aims to address the vulnerabilities inherent in Vision-Language Models (VLMs) without the necessity of extensive parameter training or modifying the existing model architecture.

**Note:** This repository is in its preliminary version and represents our first effort in organizing the code. If you encounter any bugs or issues, please feel free to report them in the 'Issues' section of this GitHub repository. A more refined version of the code will be released following the publication of the paper.

## Getting Started

This project is built upon the framework of CoOp (available on GitHub at [CoOp on GitHub](https://github.com/KaiyangZhou/CoOp)). To set up the required environment and download the necessary datasets, please refer to the installation guide at [CoOp Installation Guide](https://github.com/KaiyangZhou/CoOp#how-to-install).

To run the project, execute the script using the following command:

```bash
./scripts/main.sh
```

We are excited to see how you will utilize AdvPT in your research and applications. Stay tuned for further updates and feel free to contribute to the development of this project!

## Citation
If you find this work useful in your research, please consider citing:

```bibtex
@article{zhang2023adversarial,
  title={Adversarial Prompt Tuning for Vision-Language Models},
  author={Zhang, Jiaming and Ma, Xingjun and Wang, Xin and Qiu, Lingyu and Wang, Jiaqi and Jiang, Yu-Gang and Sang, Jitao},
  journal={arXiv preprint arXiv:2311.11261},
  year={2023}
}
```
