import torch
import torch.nn.functional as F
from utils.enumType import NormType
from utils.util import random_init, clamp_by_l2
from utils.hook import SingleModelHook


class PGD():
    def __init__(self, epsilon, *args, **kwargs):
        self.epsilon = epsilon

        self.name = None
        self.num_iters = kwargs.get('num_iters', 10)
        self.norm_type = kwargs.get('norm_type', NormType.Linf)
        self.preprocess = kwargs.get('preprocess', lambda x: x)
        self.bounding = kwargs.get('bounding', (0, 1))


    def run(self, net, image, target=None, scaler=1, feature_layer='fc', *args):
        hook = SingleModelHook(net, feature_layer, use_inp=True)

        criterion = torch.nn.KLDivLoss(reduction='batchmean')

        with torch.no_grad():
            if target is None:
                net(self.preprocess(image))
                clean_embeddings = hook.get_hooked_value().detach()
                hook.clear()
            elif target.ndim == 4:
                net(self.preprocess(target))
                clean_embeddings = hook.get_hooked_value().detach()
                hook.clear()
            elif target.ndim == 2:
                clean_embeddings = target
            else:
                raise ('Error when init clean embedding')

        iteration = self.attack(image)

        for i in range(self.num_iters):
            image_adv = next(iteration)
            net(image_adv)

            loss = criterion(hook.get_hooked_value().log_softmax(dim=-1), clean_embeddings.softmax(dim=-1))
            loss = loss * scaler

            loss.backward()
            hook.clear()

        hook.remove()
        image_adv = next(iteration)
        return image_adv

    def input_diversity(self, image):
        return image

    def update_grad(self):
        self.grad = self.delta.grad.clone()

    def normalize_grad(self):
        if self.norm_type == NormType.Linf:
            return torch.sign(self.grad)
        elif self.norm_type == NormType.L2:
            return self.grad / torch.norm(self.grad, dim=(1, 2, 3), p=2, keepdim=True)

    def project(self, delta, epsilon):
        if self.norm_type == NormType.Linf:
            return torch.clamp(delta, -epsilon, epsilon)
        elif self.norm_type == NormType.L2:
            return clamp_by_l2(delta, epsilon)

    def attack(self, image):
        self.delta = random_init(image, self.norm_type, self.epsilon)

        self.grad = torch.zeros_like(self.delta)

        epsilon_per_iter = self.epsilon / self.num_iters * 1.25

        for i in range(self.num_iters):
            self.delta = self.delta.detach()
            self.delta.requires_grad = True

            image_diversity = self.input_diversity(image + self.delta)
            image_diversity = self.preprocess(image_diversity)

            yield image_diversity

            self.update_grad()
            norm_grad = self.normalize_grad()
            self.delta = self.delta.data + epsilon_per_iter * norm_grad

            # constraint 1: epsilon
            self.delta = self.project(self.delta, self.epsilon)
            # constraint 2: image range
            self.delta = torch.clamp(image + self.delta, *self.bounding) - image

        yield torch.clamp((image + self.delta).detach(), *self.bounding)

class DI(PGD):
    def __init__(self, epsilon, *args, **kwargs):
        super(DI, self).__init__(epsilon, *args, **kwargs)
        self.resize_rate = kwargs.get('resize_rate', 1.10)
        self.diversity_prob = kwargs.get('diversity_prob', 0.3)

    def input_diversity(self, x):
        assert self.resize_rate >= 1.0
        assert self.diversity_prob >= 0.0 and self.diversity_prob <= 1.0

        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)
        # print(img_size, img_resize, resize_rate)
        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)
        padded = F.interpolate(padded, size=[img_size, img_size])
        ret = padded if torch.rand(1) < self.diversity_prob else x
        return ret

class MI(PGD):
    def __init__(self, epsilon, *args, **kwargs):
        super(MI, self).__init__(epsilon, *args, **kwargs)
        self.momentum = kwargs.get('momentum', 0.9)

    def update_grad(self):
        grad = self.delta.grad.clone()
        self.grad = self.grad * self.momentum + grad


class DIMI(DI, MI):
    def __init__(self, epsilon, *args, **kwargs):
        super(DIMI, self).__init__(epsilon, *args, **kwargs)

    def input_diversity(self, x):
        return DI.input_diversity(self, x)

    def update_grad(self):
        MI.update_grad(self)


class PGDAuxiliary(PGD):
    def run(self, net, image, *args):
        sur, aux = net
        sur_hook = SingleModelHook(sur, 'fc', use_inp=True)
        aux_hook = SingleModelHook(aux, 'fc', use_inp=True)

        criterion = torch.nn.KLDivLoss(reduction='batchmean')

        with torch.no_grad():
            sur(self.preprocess(image))
            aux(self.preprocess(image))

            clean_embeddings = sur_hook.get_hooked_value().detach()
            #aux_embeddings = aux_hook.get_hooked_value().detach()
            aux_embeddings = torch.randn_like(clean_embeddings)
            sur_hook.clear()
            aux_hook.clear()

        iteration = self.attack(image)

        for i in range(self.num_iters):
            image_adv = next(iteration)

            sur(image_adv)

            loss = criterion(sur_hook.get_hooked_value().log_softmax(dim=-1), clean_embeddings.softmax(dim=-1))
            loss += criterion(sur_hook.get_hooked_value().log_softmax(dim=-1), aux_embeddings.softmax(dim=-1))

            loss.backward()

            sur_hook.clear()
            aux_hook.clear()

        image_adv = next(iteration)

        return image_adv


class PGDEnsemble(PGD):
    def run(self, net, image, *args):
        sur, aux = net
        sur_hook = SingleModelHook(sur, 'fc', use_inp=True)
        aux_hook = SingleModelHook(aux, 'fc', use_inp=True)

        criterion = torch.nn.KLDivLoss(reduction='batchmean')
        #criterion = torch.nn.CosineEmbeddingLoss()

        with torch.no_grad():
            sur(self.preprocess(image))
            aux(self.preprocess(image))

            clean_embeddings = sur_hook.get_hooked_value().detach()
            aux_embeddings = aux_hook.get_hooked_value().detach()
            sur_hook.clear()
            aux_hook.clear()

        iteration = self.attack(image)

        for i in range(self.num_iters):
            image_adv = next(iteration)

            sur(image_adv)
            aux(image_adv)

            loss = criterion(sur_hook.get_hooked_value().log_softmax(dim=-1), clean_embeddings.softmax(dim=-1))
            loss -= criterion(aux_hook.get_hooked_value().log_softmax(dim=-1), aux_embeddings.softmax(dim=-1))

            #adv_embeddings = features['feat'] / features['feat'].norm(dim=-1, keepdim=True)
            #loss = criterion(clean_embeddings, adv_embeddings, torch.ones(len(image)).to(image.device))
            loss.backward()

            sur_hook.clear()
            aux_hook.clear()

        image_adv = next(iteration)

        return image_adv


class ILA(PGD):
    def __init__(self, epsilon, *args, **kwargs):
        super(ILA, self).__init__(epsilon, *args, **kwargs)
        self.coeff = kwargs.get('coeff', 1.0)

    def run(self, net, image, target=None, scaler=1, feature_layer='fc', *args):
        image_adv = PGD.run(self, net, image, target, scaler, feature_layer, *args)

        target = image_adv
        scaler = 1

        hook = SingleModelHook(net, feature_layer, use_inp=True)

        with torch.no_grad():
            net(self.preprocess(image))
            clean_embeddings = hook.get_hooked_value().detach()
            hook.clear()

            net(self.preprocess(target))
            target_embeddings = hook.get_hooked_value().detach()
            hook.clear()

        iteration = self.attack(image)

        for i in range(self.num_iters):
            image_adv = next(iteration)
            net(image_adv)

            loss = self._mid_layer_target_loss(target_embeddings, hook.get_hooked_value(), clean_embeddings)
            loss = loss * scaler

            loss.backward()
            hook.clear()

        hook.remove()
        image_adv = next(iteration)
        return image_adv

    def _mid_layer_target_loss(self, target_embeddings, adversarial_embeddings, clean_embeddings):
        x = (target_embeddings - clean_embeddings).view(1, -1)
        y = (adversarial_embeddings - clean_embeddings).view(1, -1)

        x_norm = x / x.norm()
        if (y==0).all():
            y_norm = y
        else:
            y_norm = y / y.norm()

        angle_loss = torch.mm(x_norm, y_norm.transpose(0, 1))
        magnitude_gain = y.norm() / x.norm()
        return angle_loss + magnitude_gain * self.coeff