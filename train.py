import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from utils import get_dataloader, get_model
import argparse
import torch
import numpy as np
import argparse
from tqdm import tqdm


def train(args):
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = 'cpu'
    if args.expert == 'attack':
        train_path = './attack_datasets/%s/train' % (args.dataset)
        test_path = './attack_datasets/%s/test' % (args.dataset)
    elif args.expert == 'purify':
        train_path = './purified_datasets/%s/test' % (args.dataset)
        test_path = './purified_datasets/%s/test' % (args.dataset)
    train_loader, test_loader = get_dataloader(args.dataset, args.bs, args.size, train_path=train_path, test_path=test_path)
    if args.dataset == 'mstar':
        num_class = 10
    elif args.dataset == 'acd':
        num_class = 6
    elif args.dataset == 'opensar':
        num_class = 6
    model = get_model(args.model, num_class)
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = None
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    
    best_acc = 0
    last_epoch = 0

    for epoch in tqdm(range(1, args.epochs + 1)):
        model.train(True)
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        acc = test(test_loader, model, device)
        if acc > best_acc:
            last_epoch = epoch
            best_acc = acc
            torch.save(model.state_dict(),"./models/%s/%s_%s-%.2f.pt"%(args.dataset, args.expert, model.__class__.__name__, best_acc))
        if epoch - last_epoch == args.es_epoch:
            print('Early stop!')
            break
        print("Epoch:%d Loss:%.2f Acc:%.2f Best:%.2f"%(epoch, loss, acc, best_acc))


def test(test_loader, model, device):
    model.eval()
    correct = 0
    nums = 0
    for inputs, labels in tqdm(test_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        pred = torch.max(outputs, 1)[1]
        nums += len(pred)
        correct += (pred == labels).sum()
    return correct / nums * 100


def newtrain(args):
    os.makedirs(f'./models/{args.dataset}', exist_ok=True)
    train(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='resnet', type=str)
    parser.add_argument('--dataset', default='mstar', type=str)
    parser.add_argument('--at', default=True)
    parser.add_argument('--opt', default='adam', type=str)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--bs', default=512, type=int)
    parser.add_argument('--size', default=128, type=int)
    parser.add_argument('--es-epoch', default=10, type=int)
    parser.add_argument('--expert', default='purify', type=str, help='attack or purify')
    args = parser.parse_args()
    newtrain(args)
