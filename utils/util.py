import torch
from .enumType import NormType
from torchvision import transforms
import timm
# from models import *
import ruamel.yaml as yaml
from pathlib import Path
from torchvision.models import ResNet101_Weights, ResNet50_Weights, ViT_B_16_Weights, MobileNet_V2_Weights, EfficientNet_B0_Weights, DenseNet121_Weights
from torchvision._internally_replaced_utils import load_state_dict_from_url


normalize_list = {'clip': transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
                  'general': transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                  'imagenet': transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])}

def clamp_by_l2(x, max_norm):
    norm = torch.norm(x, dim=(1,2,3), p=2, keepdim=True)
    factor = torch.min(max_norm / norm, torch.ones_like(norm))
    return x * factor

def random_init(x, norm_type, epsilon):
    delta = torch.zeros_like(x)
    if norm_type == NormType.Linf:
        delta.data.uniform_(0.0, 1.0)
        delta.data = delta.data * epsilon
    elif norm_type == NormType.L2:
        delta.data.uniform_(0.0, 1.0)
        delta.data = delta.data - x
        delta.data = clamp_by_l2(delta.data, epsilon)
    return delta


def is_image_file(filename):
    IMG_EXTENSIONS = [
        '.jpg', '.JPG', '.jpeg', '.JPEG',
        '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
    ]
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_model(name, num_classes=2, model_config='config/surrogate.yaml'):
    """ num_classes is useless, which is just for deploying hook function """
    model_config = read_yaml(model_config)
    if name == 'resnet18':
        model = resnet18(num_classes=num_classes)
    elif name == 'resnet101':
        model = resnet101(num_classes=num_classes)
    elif name == 'resnet50':
        model = resnet50(num_classes=num_classes)
    elif name == 'efficientnet_b0':
        model = efficientnet_b0(num_classes=num_classes)
    elif name == 'mobilenet_v2':
        model = mobilenet_v2(num_classes=num_classes)
    elif name == 'densenet121':
        model = densenet121(num_classes=num_classes)
    elif name == 'regnet_x_1_6gf':
        model = regnet_x_1_6gf(num_classes=num_classes)
    elif name == 'ViT-B16':
        model = vit_b_16(num_classes=num_classes)
    elif name == 'Clip-RN50':
        model = ClipResnet(name='RN50', num_classes=num_classes)
    elif name == 'Clip-RN50-arcface':
        model = ClipModel(name='RN50', num_classes=num_classes)
        state_dict = torch.load('/mnt/user/code/CLIP_Finetune/checkpoints/clip/RN50.pth', map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
    elif name == 'Clip-RN50x64-arcface':
        model = ClipModel(name='RN50x64', num_classes=num_classes)
        state_dict = torch.load('/mnt/user/code/CLIP_Finetune/checkpoints/clip/RN50x64.pth', map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
    elif name == 'resnet50-arcface':
        model = resnet50(num_classes=num_classes)
        state_dict = torch.load('/mnt/user/code/CLIP_Finetune/checkpoints/imagenet/resnet50.pth', map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
    elif name == 'resnet50-scratch':
        model = resnet50(num_classes=num_classes)
        state_dict = torch.load('/mnt/user/code/CLIP_Finetune/checkpoints/scratch/resnet50.pth', map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
    elif name == 'Clip-ViT-B32':
        model = ClipViT(name='ViT-B/32', num_classes=num_classes)
    elif name == 'Clip-ViT-B16-arcface':
        model = ClipViT(name='ViT-B/16', num_classes=num_classes)
        state_dict = torch.load('/mnt/user/code/CLIP_Finetune/checkpoints/clip/ViT-B-16.pth', map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
    elif name == 'OpenClip-ViT-H14-arcface':
        model = ClipOpen(name='ViT-H-14', num_classes=num_classes)
        state_dict = torch.load('/mnt/user/code/CLIP_Finetune/checkpoints/clip/ViT-H-14.pth', map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
    elif name == 'convnext_xxlarge-arcface':
        model = ClipOpen(name='hf-hub:laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup', num_classes=num_classes)
        state_dict = torch.load('/mnt/user/code/CLIP_Finetune/checkpoints/open_clip/convnext_xxlarge.pth', map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
    elif name == 'Clip-ViT-B16':
        model = ClipViT(name='ViT-B/16', num_classes=num_classes)
    elif name == 'BeiT_v2-B16':
        model = BeitViT(name='beitv2_base_patch16_224', num_classes=num_classes)
    elif name == 'ALBEF-ViT-B16':
        model = ALBEFViT(name='ViT-B/16', num_classes=num_classes)
    elif name == 'ALBEF-VE-ViT-B16':
        model = ALBEFViT(name='ViT-B/16', num_classes=1000)
    elif name == 'MAE-ViT-B16':
        model = MAEViT(name='ViT-B/16', num_classes=num_classes)
    elif name == 'SimCLR-RN50':
        model = resnet50(num_classes=num_classes)
    elif name == 'MoCo-ViT-B16':
        model = MoCoViT(name='ViT-B/16', num_classes=num_classes)
    else:
        raise (f'Model {name} Not Found')

    # load weights

    if name == 'resnet101':
        state_dict = load_state_dict_from_url(ResNet101_Weights.IMAGENET1K_V1.url, model_dir='cache')
        del state_dict['fc.weight'], state_dict['fc.bias']
        model.load_state_dict(state_dict, strict=False)
    elif name == 'resnet50':
        state_dict = load_state_dict_from_url(ResNet50_Weights.IMAGENET1K_V1.url, model_dir='cache')
        del state_dict['fc.weight'], state_dict['fc.bias']
        model.load_state_dict(state_dict, strict=False)
    elif name == 'ViT-B16':
        state_dict = load_state_dict_from_url(ViT_B_16_Weights.IMAGENET1K_V1.url, model_dir='cache')
        del state_dict['heads.head.weight'], state_dict['heads.head.bias']
        model.load_state_dict(state_dict, strict=False)
    elif name == 'efficientnet_b0':
        state_dict = load_state_dict_from_url(EfficientNet_B0_Weights.IMAGENET1K_V1.url, model_dir='cache')
        del state_dict['classifier.weight'], state_dict['classifier.bias']
        model.load_state_dict(state_dict, strict=False)
    elif name == 'mobilenet_v2':
        state_dict = load_state_dict_from_url(MobileNet_V2_Weights.IMAGENET1K_V1.url, model_dir='cache')
        del state_dict['classifier.weight'], state_dict['classifier.bias']
        model.load_state_dict(state_dict, strict=False)
    elif name == 'densenet121':
        state_dict = load_state_dict_from_url(DenseNet121_Weights.IMAGENET1K_V1.url, model_dir='cache')
        del state_dict['classifier.weight'], state_dict['classifier.bias']
        model.load_state_dict(state_dict, strict=False)
    elif name == 'Clip-RN50':
        model.load_pretrain(tmp_dir='cache')
    elif name == 'Clip-ViT-B32':
        model.load_pretrain(tmp_dir='cache')
    elif name == 'Clip-ViT-B16':
        model.load_pretrain(tmp_dir='cache')
    elif name == 'BeiT_v2-B16':
        visual_encoder = timm.models.create_model('beitv2_base_patch16_224', pretrained=True, num_classes=0)
        model.visual_encoder = visual_encoder
    elif name == 'ALBEF-ViT-B16':
        state_dict = torch.load(model_config[name]['path'], map_location='cpu')['model']
        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.visual_encoder)
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
        model.load_state_dict(state_dict, strict=False)
    elif name == 'ALBEF-VE-ViT-B16':
        state_dict = torch.load(model_config[name]['path'], map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
    elif name == 'MAE-ViT-B16':
        state_dict = torch.load(model_config[name]['path'], map_location='cpu')['model']
        model.visual_encoder.load_state_dict(state_dict, strict=False)
    elif name == 'SimCLR-RN50':
        state_dict = torch.load(model_config[name]['path'], map_location='cpu')['state_dict']
        new_state_dict = model.state_dict()
        del new_state_dict['fc.weight'], new_state_dict['fc.bias']
        for k in new_state_dict.keys():
            new_state_dict[k] = state_dict['convnet.'+ k]
        model.load_state_dict(new_state_dict, strict=False)
    elif name == 'MoCo-ViT-B16':
        state_dict = torch.load(model_config[name]['path'], map_location='cpu')['state_dict']
        new_state_dict = model.visual_encoder.state_dict()
        del new_state_dict['head.weight'], new_state_dict['head.bias']
        for k in new_state_dict.keys():
            new_state_dict[k] = state_dict['module.base_encoder.'+ k]
        model.load_state_dict(new_state_dict, strict=False)
    # else:
    #     raise (f'Model {name} Not Found')

    return model


def read_yaml(path):
    return yaml.load(open(path, 'r'), Loader=yaml.Loader)

def dir_check(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def distance(A, B):
    prod = A @ B.T

    prod_A = A @ A.T
    norm_A = prod_A.diag().unsqueeze(1).expand_as(prod)

    prod_B = B @ B.T
    norm_B = prod_B.diag().unsqueeze(0).expand_as(prod)

    res = norm_A + norm_B - 2 * prod
    return res

def interpolate_pos_embed(pos_embed_checkpoint, visual_encoder):
    # interpolate position embedding
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = visual_encoder.patch_embed.num_patches
    num_extra_tokens = visual_encoder.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)

    if orig_size != new_size:
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        print('reshape position embedding from %d to %d' % (orig_size ** 2, new_size ** 2))

        return new_pos_embed
    else:
        return pos_embed_checkpoint
