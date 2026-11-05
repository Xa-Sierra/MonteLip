#!/usr/bin/env python3
"""
compare_montelip_pgd_images.py - Comparacion directa, en los mismos
puntos base, de MonteLip-G (con eta ya calibrado via
diag_montelip_g_images.py) contra PGD-multistart, sobre una red de
imagen real (CIFAR-10/MNIST).

Objetivo: verificar si el mecanismo de escape de MonteLip-G (Metropolis +
temperatura) aporta algo que PGD-multistart (reinicios + ascenso puro,
sin mecanismo de escape) no logra igualar, en el regimen donde ya
confirmamos que el paisaje de R(a,b) es genuinamente multimodal (CIFAR-10
real, a diferencia de las gaussianas sinteticas donde ambos eran
triviales).

Se reporta PGD-multistart en dos configuraciones:
  - "default": la usada en el resto del proyecto (5 reinicios x 200
    pasos = 1000 pasos totales)
  - "matched": presupuesto de pasos igualado a N=3000 de MonteLip-G
    (15 reinicios x 200 pasos = 3000), para una comparacion justa de
    "mismo costo computacional aproximado"

Uso:
    python -u compare_montelip_pgd_images.py --dataset cifar10 --model cifar_net \
        --eta-fracs 0.02,0.05 --mg-iters 3000
"""
import argparse
import json
import os

import numpy as np
import torch

from models_images import LeNet5, MediumCNN, CIFARNet
from data_images import get_mnist_test, get_cifar10_test, select_base_points_from_dataset
from montelip import montelip_g, gradient_ascent_local
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
    ap.add_argument('--eta-fracs', type=str, default='0.02,0.05',
                    help='Valores de eta (x r) para MonteLip-G a comparar, '
                         'basados en el barrido previo')
    ap.add_argument('--mg-iters', type=int, default=3000)
    ap.add_argument('--n-points', type=int, default=5)
    ap.add_argument('--jac-samples', type=int, default=50)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out-dir', type=str, default='results_compare_pgd')
    args = ap.parse_args()

    eta_fracs = [float(e) for e in args.eta_fracs.split(',')]
    device = torch.device(args.device)
    tag = f'{args.dataset}_{args.model}'
    print(f'device: {device}, {tag}, r={args.r}')

    ckpt_path = f'checkpoints/{tag}.pt'
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
                        'reference': ref, 'montelip_g': [], 'pgd_multistart': []}

        for eta_frac in eta_fracs:
            eta = eta_frac * args.r
            torch.manual_seed(500 + i); np.random.seed(500 + i)
            mg = montelip_g(model, x0, r_domain=args.r, lr=eta, T0=1.0, rho=1.003,
                            max_iters=args.mg_iters)
            recovery = mg.L_estimate / ref if ref > 0 else None
            point_result['montelip_g'].append({
                'eta_frac': eta_frac, 'n_iters': args.mg_iters,
                'L_estimate': mg.L_estimate, 'elapsed_s': mg.elapsed_s,
                'recovery': recovery,
            })
            print(f'  MonteLip-G  eta={eta_frac:.2f}r  N={args.mg_iters}: '
                 f'L={mg.L_estimate:.4f}  recovery={recovery:.1%}  t={mg.elapsed_s:.2f}s')

        torch.manual_seed(600 + i); np.random.seed(600 + i)
        pgd_default = gradient_ascent_local(model, x0, r_domain=args.r,
                                            n_restarts=5, n_steps=200)
        recovery_d = pgd_default['L_estimate'] / ref if ref > 0 else None
        point_result['pgd_multistart'].append({
            'config': 'default_5x200', 'n_restarts': 5, 'n_steps': 200,
            'total_steps': 1000, 'L_estimate': pgd_default['L_estimate'],
            'elapsed_s': pgd_default['elapsed_s'], 'recovery': recovery_d,
        })
        print(f'  PGD-multistart (default 5x200): L={pgd_default["L_estimate"]:.4f}  '
             f'recovery={recovery_d:.1%}  t={pgd_default["elapsed_s"]:.2f}s')

        n_restarts_matched = max(1, args.mg_iters // 200)
        torch.manual_seed(700 + i); np.random.seed(700 + i)
        pgd_matched = gradient_ascent_local(model, x0, r_domain=args.r,
                                            n_restarts=n_restarts_matched, n_steps=200)
        recovery_m = pgd_matched['L_estimate'] / ref if ref > 0 else None
        point_result['pgd_multistart'].append({
            'config': f'matched_{n_restarts_matched}x200',
            'n_restarts': n_restarts_matched, 'n_steps': 200,
            'total_steps': n_restarts_matched * 200,
            'L_estimate': pgd_matched['L_estimate'],
            'elapsed_s': pgd_matched['elapsed_s'], 'recovery': recovery_m,
        })
        print(f'  PGD-multistart (matched {n_restarts_matched}x200): '
             f'L={pgd_matched["L_estimate"]:.4f}  recovery={recovery_m:.1%}  '
             f't={pgd_matched["elapsed_s"]:.2f}s')

        out['points'].append(point_result)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = f'{args.out_dir}/compare_{tag}.json'
    with open(out_path, 'w') as f:
        json.dump(to_serializable(out), f, indent=2)
    print(f'\nGuardado: {out_path}')

    print('\n' + '=' * 60)
    print('Resumen (recuperacion media sobre los puntos):')
    for idx, eta_frac in enumerate(eta_fracs):
        vals = [pt['montelip_g'][idx]['recovery'] for pt in out['points']]
        print(f'  MonteLip-G eta={eta_frac:.2f}r: {np.mean(vals):.1%} +- {np.std(vals):.1%}')
    for cfg_idx, label in [(0, 'default_5x200'), (1, 'matched')]:
        vals = [pt['pgd_multistart'][cfg_idx]['recovery'] for pt in out['points']]
        print(f'  PGD-multistart ({label}): {np.mean(vals):.1%} +- {np.std(vals):.1%}')


if __name__ == '__main__':
    main()
