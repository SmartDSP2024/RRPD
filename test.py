import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from utils import get_dataloader, get_model
import argparse
import torch
import numpy as np
import argparse
from tqdm import tqdm


def test(model_name, dataset, bs, size):
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = 'cpu'
    print(f'using device is {device}')
    train_path = './attack_datasets/%s/train' % (args.dataset)
    test_path = './attack_datasets/%s/test' % (args.dataset)
    _, test_loader = get_dataloader(dataset, bs, size, train_path=train_path, test_path=test_path)
    if dataset == 'mstar':
        num_class = 10
    elif dataset == 'acd':
        num_class = 6
    elif dataset == 'opensar':
        num_class = 6
    model = get_model(model_name, num_class)
    model = model.to(device)
    model.load_state_dict(torch.load(os.path.join(f'./pretrained/clean/{dataset}/{model_name}.pt')))
    model.eval()

    correct1 = 0
    nums = 0

    for inputs, labels in tqdm(test_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        pred = torch.max(outputs, 1)[1]
        nums += len(pred)
        test = inputs.shape[0] == len(pred)
        print(test)
        correct1 += (pred == labels).sum()
    print('correct: ', correct1.item() / nums)


# main
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='resnet', type=str)
    parser.add_argument('--dataset', default='mstar', type=str)
    parser.add_argument('--bs', default=1024, type=int)
    parser.add_argument('--size', default=128, type=int)
    args = parser.parse_args()
    test(args.model, args.dataset, args.bs, args.size)
