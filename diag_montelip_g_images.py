#!/usr/bin/env python3
"""
diag_montelip_g_images.py - Mismo diagnostico que sweep_montelip_g_eta_diag.py
(N=0 vs N=1 vs N grande, con accepted_up/down y final_dist) pero sobre
checkpoints de redes de imagen reales (MNIST/CIFAR-10) en vez de las
gaussianas sinteticas -- para ver si "un solo paso de gradiente basta"
se sostiene con arquitecturas convolucionales y datos reales, o si es un
artefacto especifico del regimen sintetico (MLP, clasificacion binaria
gaussiana) que hemos probado hasta ahora.

Uso:
    python -u diag_montelip_g_images.py --dataset cifar10 --model cifar_net
    python -u diag_montelip_g_images.py --dataset mnist --model lenet5
"""
import argparse
import json
import os

import numpy as np
import torch

from models_images import LeNet5, MediumCNN, CIFARNet
from data_images import get_mnist_test, get_cifar10_test, select_base_points_from_dataset
from montelip import montelip_g
from montelip_ball_sampling_ADDITION import jacobian_ball_sampled


MODEL_MAP = {'lenet5': LeNet5, 'medium_cnn': MediumCNN, 'cifar_net': CIFARNet}


def to_serializable(o):
    if isinstance(o, dict):
        return {k: to_serializable(v) for k, v in o.items()}
    if isinstance(o, list):
        return [to_serializable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['mnist', 'cifar10'], required=True)
    ap.add_argument('--model', choices=list(MODEL_MAP.keys()), required=True)
    ap.add_argument('--r', type=float, default=8/255)
    ap.add_argument('--eta-fracs', type=str, default='0.1,0.5,2.0')
    ap.add_argument('--iters', type=str, default='0,1,50,3000')
    ap.add_argument('--n-points', type=int, default=5)
    ap.add_argument('--jac-samples', type=int, default=50)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out-dir', type=str, default='results_images_diag')
    args = ap.parse_args()

    eta_fracs = [float(e) for e in args.eta_fracs.split(',')]
    iters_list = [int(i) for i in args.iters.split(',')]
    device = torch.device(args.device)
    tag = f'{args.dataset}_{args.model}'
    print(f'device: {device}, {tag}, r={args.r}')

    ckpt_path = f'checkpoints/{tag}.pt'
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'{ckpt_path} no existe -- corre train_images.py primero')

    model = MODEL_MAP[args.model]().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    test_ds = get_mnist_test() if args.dataset == 'mnist' else get_cifar10_test()
    x_bases, base_idx, labels = select_base_points_from_dataset(
        model, test_ds, n_points=args.n_points, seed=args.seed, device=device)
    print(f'Puntos base: {list(base_idx)}, labels {labels}')

    out = {'dataset': args.dataset, 'model': args.model, 'r': args.r, 'points': []}

    for i, x0 in enumerate(x_bases):
        print(f'\npunto {i+1}/{len(x_bases)} (label={labels[i]})')

        jb = jacobian_ball_sampled(model, x0, r_domain=args.r, n_samples=args.jac_samples,
                                   seed=args.seed)
        ref = jb['L_estimate']
        print(f'  referencia (jac_ball): {ref:.4f}')

        point_result = {'x_idx': int(base_idx[i]), 'label': labels[i],
                        'reference': ref, 'runs': []}

        for eta_frac in eta_fracs:
            eta = eta_frac * args.r
            for n_iters in iters_list:
                torch.manual_seed(500 + i); np.random.seed(500 + i)
                mg = montelip_g(model, x0, r_domain=args.r, lr=eta, T0=1.0, rho=1.003,
                                max_iters=n_iters)
                recovery = mg.L_estimate / ref if ref > 0 else None
                run_data = {
                    'eta_frac': eta_frac, 'n_iters': n_iters,
                    'L_estimate': mg.L_estimate, 'elapsed_s': mg.elapsed_s,
                    'recovery': recovery, 'accepted_up': mg.accepted_up,
                    'accepted_down': mg.accepted_down, 'final_dist': mg.final_dist,
                }
                point_result['runs'].append(run_data)
                rec_str = f'{recovery:.1%}' if recovery is not None else 'n/a'
                print(f'  eta={eta_frac:.2f}r  N={n_iters:5d}: L={mg.L_estimate:.4f}  '
                     f'recovery={rec_str}  up={mg.accepted_up}  down={mg.accepted_down}  '
                     f'dist={mg.final_dist:.5f}')

        out['points'].append(point_result)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = f'{args.out_dir}/diag_{tag}.json'
    with open(out_path, 'w') as f:
        json.dump(to_serializable(out), f, indent=2)
    print(f'\nGuardado: {out_path}')


if __name__ == '__main__':
    main()
