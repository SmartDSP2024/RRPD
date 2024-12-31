# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for DiffPure. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------


from torchvision.models import resnet18, vgg13_bn, densenet121, wide_resnet50_2, alexnet, efficientnet_b0, \
    shufflenet_v2_x2_0, mobilenet_v3_large, resnet50, resnet101
import sys
import argparse
from typing import Any
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from classifiers.models import get_model
import torchattacks
from thop import profile, clever_format


def compute_n_params(model, return_str=True):
    tot = 0
    for p in model.parameters():
        w = 1
        for x in p.shape:
            w *= x
        tot += w
    if return_str:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


class Logger(object):
    """
    Redirect stderr to stdout, optionally print stdout to a file,
    and optionally force flushing on both stdout and the file.
    """

    def __init__(self, file_name: str = None, file_mode: str = "w", should_flush: bool = True):
        self.file = None

        if file_name is not None:
            self.file = open(file_name, file_mode)

        self.should_flush = should_flush
        self.stdout = sys.stdout
        self.stderr = sys.stderr

        sys.stdout = self
        sys.stderr = self

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def write(self, text: str) -> None:
        """Write text to stdout (and a file) and optionally flush."""
        if len(text) == 0: # workaround for a bug in VSCode debugger: sys.stdout.write(''); sys.stdout.flush() => crash
            return

        if self.file is not None:
            self.file.write(text)

        self.stdout.write(text)

        if self.should_flush:
            self.flush()

    def flush(self) -> None:
        """Flush written text to both stdout and a file, if open."""
        if self.file is not None:
            self.file.flush()

        self.stdout.flush()

    def close(self) -> None:
        """Flush, close possible files, and remove stdout/stderr mirroring."""
        self.flush()

        # if using multiple loggers, prevent closing in wrong order
        if sys.stdout is self:
            sys.stdout = self.stdout
        if sys.stderr is self:
            sys.stderr = self.stderr

        if self.file is not None:
            self.file.close()


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def update_state_dict(state_dict, idx_start=9):

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[idx_start:]  # remove 'module.0.' of dataparallel
        new_state_dict[name]=v

    return new_state_dict


# ------------------------------------------------------------------------
def get_accuracy(model, x_orig, y_orig, bs=64, device=torch.device('cuda:0')):
    n_batches = x_orig.shape[0] // bs
    acc = 0.
    for counter in range(n_batches):
        x = x_orig[counter * bs:min((counter + 1) * bs, x_orig.shape[0])].clone().to(device)
        y = y_orig[counter * bs:min((counter + 1) * bs, x_orig.shape[0])].clone().to(device)
        output = model(x)
        acc += (output.max(1)[1] == y).float().sum()

    return (acc / x_orig.shape[0]).item()


def get_image_classifier(args, device):
    if args.domain == 'mstar':
        num_class = 10
    elif args.domain == 'acd':
        num_class = 6
    elif args.domain == 'opensar':
        num_class = 6
    outputs = []
    model_name = args.classifier_name
    model = get_model(model_name,num_class)
    model.load_state_dict(torch.load('./pretrained/clean/%s/%s.pt'%(args.domain,model_name)))
    model.eval()
    
    # inputs = torch.randn(1, 1, 128, 128)
    # flops, params = profile(model, inputs=(inputs, ), verbose=False)
    # flops, params = clever_format([flops, params], "%.3f")

    model.to(device)
    outputs.append(model)
    if args.moe:
        model_adv = get_model(model_name,num_class)
        model_adv.load_state_dict(torch.load('./pretrained/adv/%s/%s.pt'%(args.domain, model_name)))
        model_adv.eval()
        model_adv.to(device)
        outputs.append(model_adv)
        model_purify = get_model(model_name,num_class)
        model_purify.load_state_dict(torch.load('./pretrained/purify/%s/%s.pt'%(args.domain, model_name)))
        model_purify.eval()
        model_purify.to(device)
        outputs.append(model_purify)
    return outputs


def load_data(args, adv_batch_size):
    resize_shape = 128
    _, test_loader = get_dataloader(args.domain, adv_batch_size, resize_shape)
    # x_val, y_val = next(iter(test_loader))
    # print(f'x_val shape: {x_val.shape}')
    # x_val, y_val = x_val.contiguous().requires_grad_(True), y_val.contiguous()
    # print(f'x (min, max): ({x_val.min()}, {x_val.max()})')

    return test_loader

def get_dataloader(dataset, bs, size, train_path, test_path, shuffle=True):
    dataset = dataset
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    train_dataset = ImageFolder(train_path, transform=train_transform)
    test_datset = ImageFolder(test_path, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=shuffle, num_workers=0)
    test_loader = DataLoader(test_datset, batch_size=bs, shuffle=shuffle, num_workers=0)
    
    return train_loader, test_loader


class AConvNet(nn.Module):
    def __init__(self, num_class=10, dropout_rate=0.5, ):
        super(AConvNet, self).__init__()
        self.dropout_rate = dropout_rate
        self.num_class = num_class
        self._layer = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(5, 5), stride=(1, 1), padding="valid"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1), padding="valid"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(32, 64, kernel_size=(6, 6), stride=(1, 1), padding="valid"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=(5, 5), stride=(1, 1), padding="valid"),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_rate),
            nn.Conv2d(128, self.num_class, kernel_size=(3, 3), stride=(1, 1), padding="valid"),
            nn.Flatten(),
            nn.Linear(6 * 6 * self.num_class, self.num_class),
        )

    def forward(self, x):
        return self._layer(x)


class CNN(nn.Module):
    def __init__(self, num_class=10):
        super(CNN, self).__init__()
        self.num_class = num_class
        self._conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding="valid"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1), padding="valid"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(32, 64, kernel_size=(6, 6), stride=(1, 1), padding="valid"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 32, kernel_size=(5, 5), stride=(1, 1), padding="valid"),
            nn.ReLU(inplace=True),

        )
        self._liner = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.Linear(1024, 128),
            nn.Linear(128, self.num_class),
        )

    def forward(self, x):
        x = self._conv(x)
        x = self._liner(x)
        return x


def _aconvnet(num_class):
    return AConvNet(num_class=num_class)


def _cnn(num_class):
    return CNN(num_class=num_class)


def _alexnet(num_class):
    model = alexnet(num_classes=num_class)
    model.features[0] = torch.nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)
    return model


def _vgg(num_class):
    model = vgg13_bn(num_classes=num_class)
    model.features[0] = torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
    return model


def _resnet(num_class):
    model = resnet18(num_classes=num_class)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model


def _resnet50(num_class):
    model = resnet50(num_classes=num_class)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model


def _resnet101(num_class):
    model = resnet101(num_classes=num_class)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model


def _wideresnet(num_class):
    model = wide_resnet50_2(num_classes=num_class)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model


def _densenet(num_class):
    model = densenet121(num_classes=num_class)
    model.features.conv0 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model


# def _vit(num_class):
#     model = vit_l_32(num_classes=num_class, image_size=128)
#     model.conv_proj = torch.nn.Conv2d(1, 1024, kernel_size=(32, 32), stride=(32, 32))
#     return model


def _efficientnet(num_class):
    model = efficientnet_b0(num_classes=num_class)
    model.features[0][0] = torch.nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
    return model


def _mobilenet(num_class):
    model = mobilenet_v3_large(num_classes=num_class)
    model.features[0][0] = torch.nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
    return model


def _shufflenet(num_class):
    model = shufflenet_v2_x2_0(num_classes=num_class)
    model.conv1[0] = torch.nn.Conv2d(1, 24, kernel_size=3, stride=2, padding=1)
    return model


def get_model(name, num_class):
    name = '_' + name
    return globals()[name](num_class=num_class)


def get_attack(atk,model,num_class):
    attack = None
    '''
    attacks = ["fgsm","pgd","cw","deepfool","sqa","op","vmi","hsj"]
    L0范数
    表示非0元素的个数,对抗样本中表示扰动的非0元素的个数

    L2范数
    表示各元素的平方和再开方,对抗样本中表示扰动的各元素的平方和再开平方根；针对图像数据,L2范数越小表示对抗样本人眼越难识别

    L∞范数
    表示各元素的绝对值的最大值,对抗样本中表示扰动的各元素的最大值

    '''
    if atk == 'jsma':
        attack =  torchattacks.JSMA(model,gamma=0.01,num_class=num_class) #L0
    elif atk == 'cw':
        attack = torchattacks.CW(model,steps=20) #L2
    elif  atk == 'deepfool':
        attack = torchattacks.DeepFool(model,steps=20,overshoot=0.08) #L2
    elif atk == 'fgsm':
        attack = torchattacks.FGSM(model,eps=8/255) #Linf
    elif atk == 'pgd':
        attack = torchattacks.PGD(model,steps=10,eps=4/255) #Linf
    elif atk == 'bim':
        attack = torchattacks.BIM(model,eps=4/255,steps=2) #Linf
    elif atk == 'eot':
        attack = torchattacks.EADEN(model, lr=0.01, binary_search_steps=4, max_iterations=20) #L2
    elif atk == 'apgd':
        attack = torchattacks.APGD(model, norm='Linf', eps=4/255, steps=2) #Linf
    elif atk == 'sf':
        attack = torchattacks.SparseFool(model, steps=10, lam=3, overshoot=0.01)
    elif atk == 'aa':
        attack = torchattacks.AutoAttack(model, norm='Linf', eps=8/255, n_classes=num_class)

    elif atk == 'op':
        attack = torchattacks.OnePixel(model,pixels=16,steps=50) #L0
    elif atk == 'sqa':
        attack = torchattacks.Square(model,eps= 8/255, n_queries=500) #Linf
    elif atk == 'sqal2':
        attack = torchattacks.Square(model,norm="L2",eps=2, n_queries=500) #L2
    else:
        raise AssertionError("attack error")
    return attack