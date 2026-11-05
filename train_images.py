#!/usr/bin/env python3
"""
train_images.py - Entrena las redes de MNIST/CIFAR-10 y guarda checkpoints.

Uso:
    python train_images.py --dataset mnist --model lenet5 --epochs 8
    python train_images.py --dataset mnist --model medium_cnn --epochs 8
    python train_images.py --dataset cifar10 --model cifar_net --epochs 30

Output: checkpoints/{dataset}_{model}.pt
        checkpoints/{dataset}_{model}_meta.json
"""
import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from models_images import LeNet5, MediumCNN, CIFARNet
from data_images import get_mnist_loaders, get_cifar10_loaders


MODEL_MAP = {'lenet5': LeNet5, 'medium_cnn': MediumCNN, 'cifar_net': CIFARNet}


def train_one_epoch(model, loader, opt, device):
    model.train()
    tot, correct, tot_loss = 0, 0, 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        out = model(xb)
        loss = F.cross_entropy(out, yb)
        loss.backward()
        opt.step()
        tot += xb.size(0)
        correct += (out.argmax(1) == yb).sum().item()
        tot_loss += loss.item() * xb.size(0)
    return correct / tot, tot_loss / tot


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tot, correct = 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        tot += xb.size(0)
        correct += (out.argmax(1) == yb).sum().item()
    return correct / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['mnist', 'cifar10'], required=True)
    ap.add_argument('--model', choices=list(MODEL_MAP.keys()), required=True)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    print(f'device: {device}, dataset: {args.dataset}, model: {args.model}')

    if args.dataset == 'mnist':
        train_loader, test_loader = get_mnist_loaders()
    else:
        train_loader, test_loader = get_cifar10_loaders()

    model = MODEL_MAP[args.model]().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'parametros: {n_params:,}')

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    history = []
    t0 = time.time()
    for ep in range(args.epochs):
        train_acc, train_loss = train_one_epoch(model, train_loader, opt, device)
        test_acc = evaluate(model, test_loader, device)
        scheduler.step()
        history.append({'epoch': ep + 1, 'train_acc': train_acc,
                        'train_loss': train_loss, 'test_acc': test_acc})
        print(f'  epoch {ep+1}/{args.epochs}: train_acc={train_acc:.4f} '
             f'loss={train_loss:.4f}  test_acc={test_acc:.4f}')

    os.makedirs('checkpoints', exist_ok=True)
    tag = f'{args.dataset}_{args.model}'
    ckpt_path = f'checkpoints/{tag}.pt'
    torch.save(model.state_dict(), ckpt_path)

    meta = {'dataset': args.dataset, 'model': args.model, 'n_params': n_params,
           'epochs': args.epochs, 'lr': args.lr, 'seed': args.seed,
           'final_test_acc': history[-1]['test_acc'], 'history': history,
           'elapsed_s': time.time() - t0}
    with open(f'checkpoints/{tag}_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'\nGuardado: {ckpt_path}')
    print(f'Test acc final: {meta["final_test_acc"]:.4f}')


if __name__ == '__main__':
    main()
