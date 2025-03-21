import os
import sys
import copy
import time
import yaml
import argparse
import torch
import torchvision

from torchvision import transforms
from dotmap import DotMap
from utils.interaction import interaction_process, return_client_model_list, return_model_list


def hook(self, input, output):
    features.value = input[0].clone()


def main(args):
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    model = load_your_model(args.model)

    transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
    ])
    dataset = torchvision.datasets.CIFAR10('./data/cifar10', train=False, transform=transform, download=True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    model.fc.register_forward_hook(hook)
    model.eval()

    for epoch in args.epoch:
        model_list = return_model_list(epoch, args.num_model, args.saved_dir)
        interaction_process(model, model_list, dataloader, output_dir, args.interaction, epoch)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args, _ = parser.parse_known_args()
    config_path = args.config

    with open(config_path) as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    args = DotMap(args)

    main(args)

