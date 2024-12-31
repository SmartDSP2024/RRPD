import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from utils import get_attack, get_dataloader, get_model
import argparse
import torch
from tqdm import tqdm
from PIL import Image

# dict: [key: label, val: name]
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

def atk(args):
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = 'cpu'
    
    print(f'stage: {args.stage}, start attack...')
    
    train_path = './datasets/%s/train' % (args.dataset)
    test_path = './datasets/%s/test' % (args.dataset)
    train_loader, test_loader = get_dataloader(dataset=args.dataset, bs=args.bs, size=args.size, train_path=train_path, test_path=test_path, shuffle=False)
    data_loader = train_loader if args.stage == 'train' else test_loader
    if args.dataset == 'mstar':
        num_class = 10
    elif args.dataset == 'acd':
        num_class = 6
    elif args.dataset == 'opensar':
        num_class = 6
    model = get_model(args.model, num_class)
    model = model.to(device)
    model.load_state_dict(torch.load(os.path.join(f'./pretrained/clean/{args.dataset}/{args.model}.pt')))
    model.eval()

    cnt = 0
    flag = 0
    for inputs, labels in tqdm(data_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        atk = get_attack('aa', model, num_class)
        adv_inputs = atk(inputs, labels)
        adv_inputs = adv_inputs.squeeze(1)

        for i in range(adv_inputs.size(0)):
            img_tensor = adv_inputs[i]
            label = labels[i].item()
            if label != flag:
                print(f"Category {mstar_label_to_class[flag]} has all been saved, for a total of {cnt}.")
                flag = label
                cnt = 0

            label_dir = os.path.join(args.output_dir,  mstar_label_to_class[label])
            os.makedirs(label_dir, exist_ok=True)
            
            img = Image.fromarray((img_tensor.detach().cpu().numpy() * 255).astype('uint8'))
            img.save(os.path.join(label_dir, f"image_{cnt}.jpeg"))
            cnt += 1

    print(f"Category {mstar_label_to_class[flag]} has all been saved, for a total of {cnt}.")


# main
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='resnet', type=str)
    parser.add_argument('--dataset', default='mstar', type=str)
    parser.add_argument('--bs', default=512, type=int)
    parser.add_argument('--size', default=128, type=int)
    parser.add_argument('--stage', default='test', type=str, help='attack target, tarin or test')
    parser.add_argument('--output_dir', default='./attack_datasets', type=str)
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.dataset, args.stage)
    atk(args)