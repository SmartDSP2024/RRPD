# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for DiffPure. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import argparse
import logging
import yaml
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torchattacks

import random
import numpy as np

import torch
import torch.nn as nn

import utils
from utils import str2bool, get_image_classifier, get_dataloader
from torchvision.utils import save_image

from runners.diffpure_guided import GuidedDiffusion
from tqdm import tqdm
from thop import profile
from PIL import Image

mstar_label_to_class = {
    0: "2S1",
    1: "BMP2_SN_9556",
    2: "BRDM_2",
    3: "BTR70_SN_C71",
    4: "BTR_60",
    5: "D7",
    6: "T62",
    7: "T72_SN_132",
    8: "ZIL131",
    9: "ZSU_23_4"
}

class rrpdClassifier(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.classifiers = nn.ModuleList(get_image_classifier(args, device))

    def forward(self, x):
        outputs = [model(x) for model in self.classifiers]
        random_factors = torch.rand(len(self.classifiers), device=x.device)
        random_factors /= random_factors.sum()
        weighted_outputs = [output * factor for output, factor in zip(outputs, random_factors)]
        return sum(weighted_outputs) / len(self.classifiers)



class SDE_Adv_Model(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args

        # image classifier
        self.classifier = rrpdClassifier(args, config.device).to(config.device)

        # diffusion model
        print(f'diffusion_type: {args.diffusion_type}')
        if args.diffusion_type == 'ddpm':
            self.runner = GuidedDiffusion(args, config, device=config.device)
        else:
            raise NotImplementedError('unknown diffusion type')

        # use `counter` to record the the sampling time every 5 NFEs (note we hardcoded print freq to 5,
        # and you may want to change the freq)
        self.register_buffer('counter', torch.zeros(1, device=config.device))
        self.tag = None


    def forward(self, x):
        counter = self.counter.item()
        x = x.repeat(1, 3, 1, 1)
        x_re = self.runner.image_editing_sample((x - 0.5) * 2, bs_id=counter, tag=self.tag)
        x_re = x_re[:, [0], :, :]
        out = self.classifier((x_re + 1) * 0.5)
        self.counter += 1
        return out, x_re


def purify(args, config):
    print(f'stage: {args.stage}, start attack...')
    train_path = './datasets/%s/train' % (args.domain)
    test_path = './datasets/%s/test' % (args.domain)
    train_loader, test_loader = get_dataloader(dataset=args.domain, bs=args.adv_batch_size, size=args.size, train_path=train_path, test_path=test_path, shuffle=False)
    data_loader = train_loader if args.stage == 'train' else test_loader
    pure_model = SDE_Adv_Model(args, config)
    pure_model = pure_model.eval().to(config.device)
    pbar = tqdm(data_loader, total=len(data_loader), desc=f"Purify", ncols=120)

    cnt = 0
    flag = 0
    for ii, (inputs, labels) in tqdm(enumerate(pbar)):
        inputs = inputs.to(config.device)
        labels = labels.to(config.device)

        pure_labels, pure_outputs = pure_model(inputs)
        
        
        # 保存图像
        for idx in range(pure_outputs.size(0)):
            label = labels[idx].item()
            if label != flag:
                print(f"Category {mstar_label_to_class[flag]} has all been saved, for a total of {cnt}.")
                flag = label
                cnt = 0
            
            img = pure_outputs[idx]
            label_dir = os.path.join(args.output_dir, mstar_label_to_class[label])
            os.makedirs(label_dir, exist_ok=True)
            save_image(img, os.path.join(label_dir, f"image_{cnt}.jpeg"))
            cnt += 1
    print(f"Category {mstar_label_to_class[flag]} has all been saved, for a total of {cnt}.")


def robustness_eval(args, config):
    middle_name = '_'.join([args.diffusion_type, args.attack_version]) if args.attack_version in ['stadv', 'standard', 'rand'] \
        else '_'.join([args.diffusion_type, args.attack_version, args.attack_type])
    log_dir = os.path.join(args.image_folder, args.classifier_name, middle_name,
                           'seed' + str(args.seed), 'data' + str(args.data_seed))
    os.makedirs(log_dir, exist_ok=True)
    args.log_dir = log_dir
    logger = utils.Logger(file_name=f'{log_dir}/log.txt', file_mode="w+", should_flush=True)

    print('starting the model and loader...')
    purify(args, config)
    logger.close()


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    # diffusion models
    parser.add_argument('--config', type=str, default='imagenet.yml', help='Path to the config file')
    parser.add_argument('--data_seed', type=int, default=0, help='Random seed')
    parser.add_argument('--seed', type=int, default=2024, help='Random seed')
    parser.add_argument('--exp', type=str, default='./exp_results', help='Path for saving running related data.')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('-i', '--image_folder', type=str, default='images', help="The folder name of samples")
    parser.add_argument('--ni', action='store_true', help="No interaction. Suitable for Slurm Job launcher")
    parser.add_argument('--sample_step', type=int, default=1, help='Total sampling steps')
    parser.add_argument('--t', type=int, default=125, help='Sampling noise scale')
    parser.add_argument('--t_delta', type=int, default=15, help='Perturbation range of sampling noise scale')
    parser.add_argument('--rand_t', type=str2bool, default=False, help='Decide if randomize sampling noise scale')
    parser.add_argument('--diffusion_type', type=str, default='ddpm', help='[ddpm, sde]')
    parser.add_argument('--score_type', type=str, default='guided_diffusion', help='[guided_diffusion, score_sde]')
    parser.add_argument('--eot_iter', type=int, default=20, help='only for rand version of autoattack')
    parser.add_argument('--use_bm', action='store_true', help='whether to use brownian motion')

    # LDSDE
    parser.add_argument('--sigma2', type=float, default=1e-3, help='LDSDE sigma2')
    parser.add_argument('--lambda_ld', type=float, default=1e-2, help='lambda_ld')
    parser.add_argument('--eta', type=float, default=5., help='LDSDE eta')
    parser.add_argument('--step_size', type=float, default=1e-3, help='step size for ODE Euler method')

    # adv
    parser.add_argument('--domain', type=str, default='mstar', help='[acd, mstar, opensar]')
    parser.add_argument('--classifier_name', type=str, default='resnet', help='which classifier to use')
    parser.add_argument('--partition', type=str, default='val')
    parser.add_argument('--adv_batch_size', type=int, default=16)
    parser.add_argument('--attack_type', type=str, default='square')
    parser.add_argument('--lp_norm', type=str, default='Linf', choices=['Linf', 'L2'])
    parser.add_argument('--attack_version', type=str, default='standard')
    parser.add_argument('--size', default=128, type=int)

    parser.add_argument('--num_sub', type=int, default=64, help='imagenet subset')
    parser.add_argument('--adv_eps', type=float, default=0.05)
    parser.add_argument('--moe', type=bool, default=False)
    parser.add_argument('--stage', default='test', type=str, help='attack target, tarin or test')
    parser.add_argument('--output_dir', default='./purified_datasets', type=str)
    args = parser.parse_args()
   
    # parser.add_argument('--gpu_ids', type=str, default='0')

    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
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

    args.image_folder = os.path.join(args.exp, args.image_folder)
    os.makedirs(args.image_folder, exist_ok=True)

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


if __name__ == '__main__':
    args, config = parse_args_and_config()
    args.output_dir = os.path.join(args.output_dir, args.domain, args.stage)
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    robustness_eval(args, config)
