from skimage.restoration import denoise_wavelet
import os
import logging
import yaml
import random
import numpy as np
import time
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

def super_resolution(inputs, mean, std):
    outputs = torch.empty_like(inputs)
    num_inputs = len(inputs)
    inputs *= std.to(inputs.device)
    inputs += mean.to(inputs.device)
    inputs = torch.clamp(inputs, 0, 1)
    mean = mean.view(-1).tolist()
    std = std.view(-1).tolist()
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std)
    ])
    for i in range(num_inputs):
        img = inputs[i].cpu().numpy()
        img_bayes = denoise_wavelet(img.transpose(1, 2, 0), convert2ycbcr=True, channel_axis=-1,
                                    method='BayesShrink', mode='soft', sigma=0.02)
        img = transform(img_bayes)
        outputs[i] = img
    return outputs
