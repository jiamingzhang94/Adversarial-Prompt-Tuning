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
import runners.utils as utils
from runners.diffpure_ddpm import Diffusion
from runners.diffpure_guided import GuidedDiffusion
from runners.diffpure_sde import RevGuidedDiffusion
from runners.diffpure_ode import OdeGuidedDiffusion
from runners.diffpure_ldsde import LDGuidedDiffusion

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

def diffpure(inputs, mean, std, args):
    outputs = torch.empty_like(inputs)
    num_inputs = len(inputs)
    inputs *= std.to(inputs.device)
    inputs += mean.to(inputs.device)
    inputs = torch.clamp(inputs, 0, 1)
    mean = mean.view(-1).tolist()
    std = std.view(-1).tolist()
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean, std)
    ])

    # load model
    args_, config_ = parse_args_and_config(args)
    ngpus = torch.cuda.device_count()
    print('starting the model and loader...')
    model = SDE_Adv_Model(args_, config_)
    model.reset_counter()
    if ngpus > 1:
        model = torch.nn.DataParallel(model)
    model = model.eval().to(config_.device)

    batch_size = 2*ngpus
    for i in range(0, num_inputs, batch_size):
        start = i
        end = min(i + batch_size, num_inputs)
        img_batch = inputs[start:end]
        img_batch_diffpure = model(img_batch).detach().cpu()
        # print(i, img_batch_diffpure.shape, torch.max(img_batch_diffpure),torch.min(img_batch_diffpure))
        outputs[start:end] = transform(img_batch_diffpure)

        del img_batch_diffpure
        del img_batch
    
    torch.cuda.empty_cache()

    return outputs

def parse_args_and_config(args):
    log_dir = os.path.join('./runners/diffpure_log')
    os.makedirs(log_dir, exist_ok=True)
    args.log_dir = log_dir

    # parse config file
    print(os.path.join(args.config))
    with open(os.path.join(args.config), 'r') as f:
        config = yaml.safe_load(f)
    new_config = utils.dict2namespace(config)

    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config
    
class SDE_Adv_Model(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.device = config.device

        # diffusion model
        print(f'diffusion_type: {args.diffusion_type}')
        if args.diffusion_type == 'ddpm':
            self.runner = GuidedDiffusion(args, config, device=self.device)
        elif args.diffusion_type == 'sde':
            self.runner = RevGuidedDiffusion(args, config, device=self.device)
        elif args.diffusion_type == 'ode':
            self.runner = OdeGuidedDiffusion(args, config, device=self.device)
        elif args.diffusion_type == 'ldsde':
            self.runner = LDGuidedDiffusion(args, config, device=self.device)
        elif args.diffusion_type == 'celebahq-ddpm':
            self.runner = Diffusion(args, config, device=self.device)
        else:
            raise NotImplementedError('unknown diffusion type')

        # use `counter` to record the the sampling time every 5 NFEs (note we hardcoded print freq to 5,
        # and you may want to change the freq)
        self.register_buffer('counter', torch.zeros(1, device=self.device))
        self.tag = None

    def reset_counter(self):
        self.counter = torch.zeros(1, dtype=torch.int, device=self.device)

    def set_tag(self, tag=None):
        self.tag = tag

    def forward(self, x):
        counter = self.counter.item()
        if counter % 100 == 0:
            print(f'diffusion times: {counter}')

        # imagenet [3, 224, 224] -> [3, 256, 256] -> [3, 224, 224]
        if 'imagenet' in self.args.domain:
            x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)

        start_time = time.time()
        x_re = self.runner.image_editing_sample((x - 0.5) * 2, bs_id=counter, tag=self.tag)
        minutes, seconds = divmod(time.time() - start_time, 60)

        if 'imagenet' in self.args.domain:
            x_re = F.interpolate(x_re, size=(224, 224), mode='bilinear', align_corners=False)

        if counter % 100 == 0:
            print(f'x shape (before diffusion models): {x.shape}')
            print(f'x shape (before classifier): {x_re.shape}')
            print("Sampling time per batch: {:0>2}:{:05.2f}".format(int(minutes), seconds))

        out = (x_re + 1) * 0.5

        self.counter += 1

        return out

