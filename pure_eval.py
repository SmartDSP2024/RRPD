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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torchattacks

import random
import numpy as np

import torch
import torch.nn as nn

import utils
from utils import str2bool, get_image_classifier, load_data

from runners.diffpure_guided import GuidedDiffusion
from tqdm import tqdm
from thop import profile
import time

class rpddClassifier(nn.Module):
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
        self.classifier = rpddClassifier(args, config.device).to(config.device)
        # modified by zjw
        total_params = sum(p.numel() for p in self.classifier.parameters())
        print(f"The parameters of Classifier is {total_params}")
        # modified by zjw

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
        # if self.moe:
        #     self.classifier1 = get_image_classifier(args).to(config.device)


    def forward(self, x):
        counter = self.counter.item()
        # 如果输入需要grad
        # if x.requires_grad:
        #     print(1)

        # 64, 1, 128, 128 -> 64, 3, 128, 128
        x = x.repeat(1, 3, 1, 1)
        # imagenet [3, 224, 224] -> [3, 256, 256] -> [3, 224, 224]
        # x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)

        x_re = self.runner.image_editing_sample((x - 0.5) * 2, bs_id=counter, tag=self.tag)

        # x_re = F.interpolate(x_re, size=(128, 128), mode='bilinear', align_corners=False)
        x_re = x_re[:,[0],:,:]

        
        out = self.classifier((x_re + 1) * 0.5)

        self.counter += 1

        return out


def eval_autoattack(args, config, test_loader, adv_batch_size, log_dir):

    attack_list = ['apgd-ce', 'apgd-dlr']
    
    # ---------------- apply the attack to classifier ----------------
    # print(f'apply the attack to classifier [{args.lp_norm}]...')
    pure_model = SDE_Adv_Model(args, config)
    pure_model = pure_model.eval().to(config.device)

    
    adv_pure_correct = 0
    total_nums = 0
    pbar = tqdm(test_loader, total=len(test_loader), desc=f"AutoAttack Evaluation", ncols=120)
    for ii, (inputs,labels) in tqdm(enumerate(pbar)):
        inputs = inputs.to(config.device)
        labels = labels.to(config.device)

        
        start_time = time.time()
        adv_pure_outputs = pure_model(inputs)
        end_time = time.time()
        elapse = end_time - start_time
        print(f"推理一次花了{elapse}秒")
        

        adv_pure_correct += int((torch.max(adv_pure_outputs.data,1)[1] == labels).sum())
        total_nums += len(adv_pure_outputs)
        pbar.set_postfix_str(f"Accuracy: {100 * adv_pure_correct / total_nums:.2f}%")

    print(f'adv: rpdd accuracy: {100. * adv_pure_correct / total_nums}')

    # predict
    
    # adversary_resnet = AutoAttack(classifier, norm=args.lp_norm, eps=args.adv_eps,
    #                               version=attack_version, attacks_to_run=attack_list,
    #                               log_path=f'{log_dir}/log_resnet.txt', device=config.device)
    # if attack_version == 'custom':
    #     adversary_resnet.apgd.n_restarts = 1
    #     adversary_resnet.fab.n_restarts = 1
    #     adversary_resnet.apgd_targeted.n_restarts = 1
    #     adversary_resnet.fab.n_target_classes = 9
    #     adversary_resnet.apgd_targeted.n_target_classes = 9
    #     adversary_resnet.square.n_queries = 5000
    # if attack_version == 'rand':
    #     adversary_resnet.apgd.eot_iter = args.eot_iter
    #     print(f'[classifier] rand version with eot_iter: {adversary_resnet.apgd.eot_iter}')
    # print(f'{args.lp_norm}, epsilon: {args.adv_eps}')

    # x_adv_resnet = adversary_resnet.run_standard_evaluation(x_val, y_val, bs=adv_batch_size)
    # print(f'x_adv_resnet shape: {x_adv_resnet.shape}')
    # torch.save([x_adv_resnet, y_val], f'{log_dir}/x_adv_resnet_sd{args.seed}.pt')

    # del x_adv_resnet
    # del adversary_resnet
    # del classifier
    # gc.collect()






def robustness_eval(args, config):
    middle_name = '_'.join([args.diffusion_type, args.attack_version]) if args.attack_version in ['stadv', 'standard', 'rand'] \
        else '_'.join([args.diffusion_type, args.attack_version, args.attack_type])
    log_dir = os.path.join(args.image_folder, args.classifier_name, middle_name,
                           'seed' + str(args.seed), 'data' + str(args.data_seed))
    os.makedirs(log_dir, exist_ok=True)
    args.log_dir = log_dir
    logger = utils.Logger(file_name=f'{log_dir}/log.txt', file_mode="w+", should_flush=True)

    # ngpus = torch.cuda.device_count()
    # adv_batch_size = args.adv_batch_size * ngpus
    adv_batch_size = args.adv_batch_size
    # print(f'ngpus: {ngpus}, adv_batch_size: {adv_batch_size}')
    print(f'adv_batch_size: {adv_batch_size}')

    # load model
    # print('starting the model and loader...')
    # model = SDE_Adv_Model(args, config)
    # if ngpus > 1:
    #     model = torch.nn.DataParallel(model)
    # model = model.eval().to(config.device)

    # load data
    test_loader = load_data(args, adv_batch_size)

    # eval classifier and sde_adv against attacks
    if args.attack_version in ['standard', 'rand', 'custom']:
        eval_autoattack(args, config, test_loader, adv_batch_size, log_dir)
    else:
        raise NotImplementedError(f'unknown attack_version: {args.attack_version}')

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
    parser.add_argument('--domain', type=str, default='opensar', help='[acd, mstar, opensar]')
    parser.add_argument('--classifier_name', type=str, default='resnet', help='which classifier to use')
    parser.add_argument('--partition', type=str, default='val')
    parser.add_argument('--adv_batch_size', type=int, default=1)
    parser.add_argument('--attack_type', type=str, default='square')
    parser.add_argument('--lp_norm', type=str, default='Linf', choices=['Linf', 'L2'])
    parser.add_argument('--attack_version', type=str, default='standard')

    parser.add_argument('--num_sub', type=int, default=64, help='imagenet subset')
    parser.add_argument('--adv_eps', type=float, default=0.05)
    parser.add_argument('--moe', type=bool, default=True)
    # parser.add_argument('--gpu_ids', type=str, default='0')

    args = parser.parse_args()

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
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    robustness_eval(args, config)
