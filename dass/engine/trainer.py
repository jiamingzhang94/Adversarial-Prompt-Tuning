import time
import numpy as np
import os.path as osp
import datetime
from collections import OrderedDict
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from dass.data import DataManager
from dass.optim import build_optimizer, build_lr_scheduler
from dass.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, mkdir_if_missing, resume_from_checkpoint,
    load_pretrained_weights
)
from dass.modeling import build_head, build_backbone
from dass.evaluation import build_evaluator
from attack.attackFeature import PGD
from attack.purification import super_resolution
import clip
from torch import randperm
import os


def get_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


class SimpleNet(nn.Module):
    """A simple neural network composed of a CNN backbone
    and optionally a head such as mlp for classification.
    """

    def __init__(self, cfg, model_cfg, num_classes, **kwargs):
        super().__init__()
        self.backbone = build_backbone(
            model_cfg.BACKBONE.NAME,
            verbose=cfg.VERBOSE,
            pretrained=model_cfg.BACKBONE.PRETRAINED,
            **kwargs,
        )
        fdim = self.backbone.out_features

        self.head = None
        if model_cfg.HEAD.NAME and model_cfg.HEAD.HIDDEN_LAYERS:
            self.head = build_head(
                model_cfg.HEAD.NAME,
                verbose=cfg.VERBOSE,
                in_features=fdim,
                hidden_layers=model_cfg.HEAD.HIDDEN_LAYERS,
                activation=model_cfg.HEAD.ACTIVATION,
                bn=model_cfg.HEAD.BN,
                dropout=model_cfg.HEAD.DROPOUT,
                **kwargs,
            )
            fdim = self.head.out_features

        self.classifier = None
        if num_classes > 0:
            self.classifier = nn.Linear(fdim, num_classes)

        self._fdim = fdim

    @property
    def fdim(self):
        return self._fdim

    def forward(self, x, return_feature=False):
        f = self.backbone(x)
        if self.head is not None:
            f = self.head(f)

        if self.classifier is None:
            return f

        y = self.classifier(f)

        if return_feature:
            return y, f

        return y


class TrainerBase:
    """Base class for iterative trainer."""

    def __init__(self):
        self._models = OrderedDict()
        self._optims = OrderedDict()
        self._scheds = OrderedDict()
        self._writer = None

    def register_model(self, name="model", model=None, optim=None, sched=None):
        if self.__dict__.get("_models") is None:
            raise AttributeError(
                "Cannot assign model before super().__init__() call"
            )

        if self.__dict__.get("_optims") is None:
            raise AttributeError(
                "Cannot assign optim before super().__init__() call"
            )

        if self.__dict__.get("_scheds") is None:
            raise AttributeError(
                "Cannot assign sched before super().__init__() call"
            )

        assert name not in self._models, "Found duplicate model names"

        self._models[name] = model
        self._optims[name] = optim
        self._scheds[name] = sched

    def get_model_names(self, names=None):
        names_real = list(self._models.keys())
        if names is not None:
            names = tolist_if_not(names)
            for name in names:
                assert name in names_real
            return names
        else:
            return names_real

    def save_model(
            self, epoch, directory, is_best=False, val_result=None, model_name=""
    ):
        names = self.get_model_names()

        for name in names:
            model_dict = self._models[name].state_dict()

            optim_dict = None
            if self._optims[name] is not None:
                optim_dict = self._optims[name].state_dict()

            sched_dict = None
            if self._scheds[name] is not None:
                sched_dict = self._scheds[name].state_dict()

            save_checkpoint(
                {
                    "state_dict": model_dict,
                    "epoch": epoch + 1,
                    "optimizer": optim_dict,
                    "scheduler": sched_dict,
                    "val_result": val_result
                },
                osp.join(directory, name),
                is_best=is_best,
                model_name=model_name,
            )

    def resume_model_if_exist(self, directory):
        names = self.get_model_names()
        file_missing = False

        for name in names:
            path = osp.join(directory, name)
            if not osp.exists(path):
                file_missing = True
                break

        if file_missing:
            print("No checkpoint found, train from scratch")
            return 0

        print(f"Found checkpoint at {directory} (will resume training)")

        for name in names:
            path = osp.join(directory, name)
            start_epoch = resume_from_checkpoint(
                path, self._models[name], self._optims[name],
                self._scheds[name]
            )

        return start_epoch

    def load_model(self, directory, epoch=None):
        if not directory:
            print(
                "Note that load_model() is skipped as no pretrained "
                "model is given (ignore this if it's done on purpose)"
            )
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(f"No model at {model_path}")

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]
            val_result = checkpoint["val_result"]
            print(
                f"Load {model_path} to {name} (epoch={epoch}, val_result={val_result:.1f})"
            )
            self._models[name].load_state_dict(state_dict)

    def set_model_mode(self, mode="train", names=None):
        names = self.get_model_names(names)

        for name in names:
            if mode == "train":
                self._models[name].train()
            elif mode in ["test", "eval"]:
                self._models[name].eval()
            else:
                raise KeyError

    def update_lr(self, names=None):
        names = self.get_model_names(names)

        for name in names:
            if self._scheds[name] is not None:
                self._scheds[name].step()

    def detect_anomaly(self, loss):
        if not torch.isfinite(loss).all():
            raise FloatingPointError("Loss is infinite or NaN!")

    def init_writer(self, log_dir):
        if self.__dict__.get("_writer") is None or self._writer is None:
            print(f"Initialize tensorboard (log_dir={log_dir})")
            self._writer = SummaryWriter(log_dir=log_dir)

    def close_writer(self):
        if self._writer is not None:
            self._writer.close()

    def write_scalar(self, tag, scalar_value, global_step=None):
        if self._writer is None:
            # Do nothing if writer is not initialized
            # Note that writer is only used when training is needed
            pass
        else:
            self._writer.add_scalar(tag, scalar_value, global_step)

    def train(self, start_epoch, max_epoch, adv_training=False):
        """Generic training loops."""
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch

        # self.before_train()
        if adv_training:
            self.before_adv_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            if adv_training:
                self.run_epoch_adv()
            else:
                self.run_epoch()
            self.after_epoch()
        self.after_train()


    def before_train(self):
        pass

    def after_train(self):
        pass

    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    def run_epoch(self):
        raise NotImplementedError

    def run_epoch_adv(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def parse_batch_train(self, batch):
        raise NotImplementedError

    def parse_batch_test(self, batch):
        raise NotImplementedError

    def forward_backward(self, batch):
        raise NotImplementedError

    def forward_backward_adv(self, batch_dict):
        raise NotImplementedError

    def model_inference(self, input):
        raise NotImplementedError

    def model_zero_grad(self, names=None):
        names = self.get_model_names(names)
        for name in names:
            if self._optims[name] is not None:
                self._optims[name].zero_grad()

    def model_backward(self, loss):
        self.detect_anomaly(loss)
        loss.backward()

    def model_update(self, names=None):
        names = self.get_model_names(names)
        for name in names:
            if self._optims[name] is not None:
                self._optims[name].step()

    def model_backward_and_update(self, loss, names=None):
        self.model_zero_grad(names)
        self.model_backward(loss)
        self.model_update(names)


class ClipModel(torch.nn.Module):
    def __init__(self, model, num_classes=1000):
        super(ClipModel, self).__init__()
        self.model = model

        # temp, self.preprocess_val = clip.load(self.name, 'cpu')
        self.visual_encoder = self.model

        output_dim = self.visual_encoder.output_dim
        # add a fc for attacking embedding
        self.fc = torch.nn.Linear(output_dim, 2)

    def forward(self, image):
        x = self.visual_encoder(image)
        x = self.fc(x)
        return x


class SimpleTrainer(TrainerBase):
    """A simple trainer class implementing generic functions."""

    def __init__(self, cfg):
        super().__init__()
        self.check_cfg(cfg)

        if torch.cuda.is_available() and cfg.USE_CUDA:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Save as attributes some frequently used variables
        self.start_epoch = self.epoch = 0
        self.max_epoch = cfg.OPTIM.MAX_EPOCH
        self.output_dir = cfg.OUTPUT_DIR

        self.cfg = cfg
        self.build_data_loader()
        self.build_model()
        self.evaluator = build_evaluator(cfg, lab2cname=self.lab2cname)
        self.best_result = -np.inf

    def check_cfg(self, cfg):
        """Check whether some variables are set correctly for
        the trainer (optional).

        For example, a trainer might require a particular sampler
        for training such as 'RandomDomainSampler', so it is good
        to do the checking:

        assert cfg.DATALOADER.SAMPLER_TRAIN == 'RandomDomainSampler'
        """
        pass

    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        batch_size = self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE
        dm = DataManager(self.cfg, batch_size)

        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader

        # optional
        self.adv = 'notransform_noshuffle'
        batch_size = self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE
        dm = DataManager(self.cfg, batch_size, self.adv)
        self.train_loader_x_notransform_noshuffle = dm.train_loader_x

        self.adv = 'noshuffle'
        batch_size = self.cfg.DATALOADER.TRAIN_X.BATCH_EMBEDDING_SIZE
        dm = DataManager(self.cfg, batch_size, self.adv)
        self.train_loader_x_noshuffle = dm.train_loader_x

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}

        self.dm = dm

    def build_model(self):
        """Build and register model.

        The default builds a classification model along with its
        optimizer and scheduler.

        Custom trainers can re-implement this method if necessary.
        """
        cfg = self.cfg

        print("Building model")
        self.model = SimpleNet(cfg, cfg.MODEL, self.num_classes)
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
        self.model.to(self.device)
        print(f"# params: {count_num_param(self.model):,}")
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("model", self.model, self.optim, self.sched)

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Detected {device_count} GPUs (use nn.DataParallel)")
            self.model = nn.DataParallel(self.model)

    def train(self, adv_training=False):
        super().train(self.start_epoch, self.max_epoch, adv_training)

    def before_train(self):
        directory = self.cfg.OUTPUT_DIR
        if self.cfg.RESUME:
            directory = self.cfg.RESUME
        self.start_epoch = self.resume_model_if_exist(directory)

        # Initialize summary writer
        writer_dir = osp.join(self.output_dir, "tensorboard")
        mkdir_if_missing(writer_dir)
        self.init_writer(writer_dir)

        # Remember the starting time (for computing the elapsed time)
        self.time_start = time.time()

    def before_adv_train(self, attack='PGD'):
        pkl_path = '/share/test/user/share1/train/{}_{}.pkl'.format(self.cfg.DATASET.NAME,
                                                                     self.cfg.MODEL.BACKBONE.NAME.replace(
                                                                         "/",
                                                                         "_"))
        if os.path.isfile(pkl_path):
            self.train_pkl = torch.load(pkl_path).to('cpu')
            print('loaded train_pkl')
            return
        train_eps = self.cfg.DATASET.TRAIN_EPS
        normalize = transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        clip_model, _ = clip.load(self.cfg.MODEL.BACKBONE.NAME, device='cpu')
        surrogate = ClipModel(model=get_model(clip_model.visual), num_classes=2).eval().to(self.device)

        if attack == 'PGD':
            attacker = PGD(train_eps / 255., preprocess=normalize, num_iters=10)
        else:
            attacker = None

        embedding_dim = surrogate.fc.in_features
        self.train_pkl = torch.empty(size=[len(self.train_loader_x_notransform_noshuffle.dataset), embedding_dim])
        for batch_idx, batch in enumerate(self.train_loader_x_notransform_noshuffle):
            inputs = batch['img'].to(self.device)
            images_adv = attacker.run(surrogate, inputs, scaler=1, feature_layer='fc')
            assert torch.max(images_adv - inputs) < (train_eps / 255. + 1e-6)
            assert torch.min(images_adv - inputs) > (-train_eps / 255 - 1e-6)
            images_adv = normalize(images_adv)
            with torch.no_grad():
                embedding = clip_model.encode_image(images_adv)

            self.train_pkl[batch_idx * self.train_loader_x_notransform_noshuffle.batch_size: (
                                                                                                         batch_idx + 1) * self.train_loader_x_notransform_noshuffle.batch_size] = embedding.cpu()

        torch.save(self.train_pkl, pkl_path)
        print('generated train_pkl')
        del surrogate
        torch.cuda.empty_cache()

    def before_adv_test(self, attack='PGD'):
        pkl_path = '/share/test/user/share1/test/{}_{}_{}.pkl'.format(self.cfg.DATASET.NAME,
                                                                              self.cfg.MODEL.BACKBONE.NAME.replace("/",
                                                                                                                   "_"),
                                                                              attack)
        mean_value, std_value = [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]
        mean = torch.tensor(mean_value).view(-1, 1, 1).to(self.device)
        std = torch.tensor(std_value).view(-1, 1, 1).to(self.device)
        normalize = transforms.Normalize(mean_value, std_value)
        self.mean, self.std = mean, std
        if os.path.isfile(pkl_path):
            self.test_pkl = torch.load(pkl_path)
            return
        self.test_pkl = torch.empty(size=[len(self.test_loader.dataset), 3, 224, 224])
        test_eps = self.cfg.DATASET.TEST_EPS

        if attack == 'PGD':
            temp_model, _ = clip.load(self.cfg.MODEL.BACKBONE.NAME, device='cpu')
            surrogate = ClipModel(model=get_model(temp_model.visual), num_classes=2).eval().to(self.device)
            attacker = PGD(test_eps / 255., preprocess=normalize, num_iters=40)
            for batch_idx, batch in enumerate(self.test_loader):
                inputs = batch['img'].to(self.device)
                inputs *= std
                inputs += mean
                images_adv = attacker.run(surrogate, inputs, scaler=1, feature_layer='fc')
                assert torch.max(images_adv - inputs) < (test_eps / 255. + 1e-6)
                assert torch.min(images_adv - inputs) > (-test_eps / 255 - 1e-6)
                images_adv = normalize(images_adv)
                self.test_pkl[batch_idx * self.test_loader.batch_size: (
                                                                                   batch_idx + 1) * self.test_loader.batch_size] = images_adv.cpu()
            torch.save(self.test_pkl, pkl_path)
            del surrogate

        else:
            raise NameError
        torch.cuda.empty_cache()

    def before_black_test(self, attack='RAP'):
        pkl_path = '/share/test/user/share1/test/{}_{}.pkl'.format(self.cfg.DATASET.NAME,
                                                                           attack)
        mean_value, std_value = [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]
        mean = torch.tensor(mean_value).view(-1, 1, 1)
        std = torch.tensor(std_value).view(-1, 1, 1)
        normalize = transforms.Normalize(mean_value, std_value)
        self.mean, self.std = mean, std
        if os.path.isfile(pkl_path):
            self.test_pkl = torch.load(pkl_path)
            self.test_pkl = (self.test_pkl - mean) / std
            return
        else:
            raise NameError

    def purify(self, baseline='super-resolution'):
        if baseline == 'super-resolution':
            inputs = self.test_pkl
            mean, std = self.mean.squeeze(0), self.std.squeeze(0)
            outputs = super_resolution(inputs, mean, std)
        else:
            raise NameError

        self.test_pkl = outputs.to(self.device)

    def after_train(self):
        print("Finish training")

        # do_test = not self.cfg.TEST.NO_TEST
        # if do_test:
        #     if self.cfg.TEST.FINAL_MODEL == "best_val":
        #         print("Deploy the model with the best val performance")
        #         self.load_model(self.output_dir)
        #     else:
        #         print("Deploy the last-epoch model")
        #     self.test()

        # Show elapsed time
        # elapsed = round(time.time() - self.time_start)
        # elapsed = str(datetime.timedelta(seconds=elapsed))
        # print(f"Elapsed: {elapsed}")

        # Close writer
        self.close_writer()

    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )

        if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
            curr_result = self.test(split="val")
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.save_model(
                    self.epoch,
                    self.output_dir,
                    val_result=curr_result,
                    model_name="model-best.pth.tar"
                )

        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)

    @torch.no_grad()
    def test_adv(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        array_to_pkl = self.test_pkl
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(-1, 1, 1).to(self.device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(-1, 1, 1).to(self.device)

        print(f"Evaluate on the *{split}* set")
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            input_adv = array_to_pkl[batch_idx * data_loader.batch_size: (batch_idx + 1) * data_loader.batch_size]
            input_adv = input_adv.to(input.device)

            # claim small noise
            x_adv = input_adv*std + mean
            x = input*std + mean
            noise = x_adv-x
            noise = torch.clamp(noise, -16 / 255.0, 16/255.0)
            x_adv = x+noise
            x_adv = torch.clamp(x_adv, 0, 1)
            assert (torch.max(x_adv - x) < (16/255.0 + 1e-6))
            assert (torch.min(x_adv - x) > (-16 / 255.0 - 1e-6))
            input_adv = (x_adv-mean)/std

            output = self.model_inference(input_adv)
            self.evaluator.process(output, label.to(input.device))

        results = self.evaluator.evaluate()
        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    def model_inference(self, input):
        return self.model(input)

    def parse_batch_test(self, batch):
        input = batch["img"]
        label = batch["label"]

        input = input.to(self.device)
        label = label.to(self.device)

        return input, label

    def get_current_lr(self, names=None):
        names = self.get_model_names(names)
        name = names[0]
        return self._optims[name].param_groups[0]["lr"]


class TrainerXU(SimpleTrainer):
    """A base trainer using both labeled and unlabeled data.

    In the context of domain adaptation, labeled and unlabeled data
    come from source and target domains respectively.

    When it comes to semi-supervised learning, all data comes from the
    same domain.
    """

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        # Decide to iterate over labeled or unlabeled dataset
        len_train_loader_x = len(self.train_loader_x)
        len_train_loader_u = len(self.train_loader_u)
        if self.cfg.TRAIN.COUNT_ITER == "train_x":
            self.num_batches = len_train_loader_x
        elif self.cfg.TRAIN.COUNT_ITER == "train_u":
            self.num_batches = len_train_loader_u
        elif self.cfg.TRAIN.COUNT_ITER == "smaller_one":
            self.num_batches = min(len_train_loader_x, len_train_loader_u)
        else:
            raise ValueError

        train_loader_x_iter = iter(self.train_loader_x)
        train_loader_u_iter = iter(self.train_loader_u)

        end = time.time()
        for self.batch_idx in range(self.num_batches):
            try:
                batch_x = next(train_loader_x_iter)
            except StopIteration:
                train_loader_x_iter = iter(self.train_loader_x)
                batch_x = next(train_loader_x_iter)

            try:
                batch_u = next(train_loader_u_iter)
            except StopIteration:
                train_loader_u_iter = iter(self.train_loader_u)
                batch_u = next(train_loader_u_iter)

            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch_x, batch_u)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                                     self.max_epoch - self.epoch - 1
                             ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

    def parse_batch_train(self, batch_x, batch_u):
        input_x = batch_x["img"]
        label_x = batch_x["label"]
        input_u = batch_u["img"]

        input_x = input_x.to(self.device)
        label_x = label_x.to(self.device)
        input_u = input_u.to(self.device)

        return input_x, label_x, input_u


class TrainerX(SimpleTrainer):
    """A base trainer using labeled data only."""

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)

        # mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(-1, 1, 1).to(self.device)
        # std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(-1, 1, 1).to(self.device)
        # normalize = transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        #
        # surrogate = ClipModel(model=self.clip_model.visual, num_classes=2).to(self.device)
        # attacker = PGD(16 / 255., preprocess=normalize, num_iters=10)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                                     self.max_epoch - self.epoch - 1
                             ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

    def run_adv_training(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(-1, 1, 1).to(self.device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(-1, 1, 1).to(self.device)

        train_eps = self.cfg.DATASET.TRAIN_EPS
        normalize = transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        # clip_model, _ = clip.load(self.cfg.MODEL.BACKBONE.NAME, device='cpu')
        surrogate = ClipModel(model=self.model.image_encoder, num_classes=2).eval().to(self.device)
        attacker = PGD(train_eps / 255., preprocess=normalize, num_iters=10)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            data_time.update(time.time() - end)
            inputs = batch['img'].to(self.device)
            inputs *= std
            inputs += mean
            self.model.image_encoder.eval()
            images_adv = attacker.run(surrogate, inputs, scaler=1, feature_layer='fc')
            self.model.image_encoder.train()
            assert torch.max(images_adv - inputs) < (train_eps / 255. + 1e-6)
            assert torch.min(images_adv - inputs) > (-train_eps / 255 - 1e-6)
            images_adv = normalize(images_adv)
            batch['img'] = images_adv
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                                     self.max_epoch - self.epoch - 1
                             ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

    def run_epoch_adv(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        # self.num_batches = len(self.train_loader_x)
        self.num_batches = len(self.train_loader_x_noshuffle)

        seed = torch.random.seed()
        torch.random.manual_seed(seed)

        # shuffle
        length = randperm(len(self.train_loader_x_noshuffle.dataset.data_source)).tolist()
        self.train_loader_x_noshuffle.dataset.data_source = [self.train_loader_x_noshuffle.dataset.data_source[i] for i
                                                             in length]
        self.train_pkl = self.train_pkl[torch.LongTensor(length)]

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x_noshuffle):

            data_time.update(time.time() - end)

            images_adv = self.train_pkl[self.batch_idx * self.train_loader_x_noshuffle.batch_size: (
                                                                                                           self.batch_idx + 1) * self.train_loader_x_noshuffle.batch_size]
            batch_dict = {'batch': batch, 'images_adv': images_adv.to(self.device)}

            loss_summary = self.forward_backward_adv(batch_dict)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                                     self.max_epoch - self.epoch - 1
                             ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        domain = batch["domain"]

        input = input.to(self.device)
        label = label.to(self.device)
        domain = domain.to(self.device)

        return input, label, domain