#!/usr/bin/env python3
"""
diagnose_mh_collapse.py
------------------------
Diagnostico dirigido para entender por que la correccion de Hastings
(accept_rule='mh') colapsa la estimacion en puntos base especificos
(x_idx=8023 y x_idx=7417, dim=784, seed=1), mientras que en el resto de
los puntos de esa misma corrida (x_idx=546, 11839, 14902) el resultado es
identico a la regla Metropolis simple.

Reproduce EXACTAMENTE el mismo modelo y los mismos datos que results_seed1
(mismo seed=1, mismo dim=784, mismo n_layers=5), y luego corre montelip_g
con record_history=True y verbose=True para ambos puntos y ambas reglas de
aceptacion, imprimiendo el historial completo para ver en que iteracion
diverge la trayectoria 'mh' respecto a 'metropolis'.

Uso:
    python diagnose_mh_collapse.py
    python diagnose_mh_collapse.py --x-idx 8023,7417 --mg-iters 3000
    python diagnose_mh_collapse.py --x-idx 546 --mg-iters 3000   # control:
        un punto que SI coincidio entre metropolis y mh, para contraste
"""
import argparse

import numpy as np
import torch

from models import VarMLP
from functions_gauss import gaussian_binary_data, to_torch_xy
from montelip import montelip_g
from run_experiment import train_classifier_with_retry


def rebuild_model_and_data(dim, n_layers, seed, device):
    """Reproduce EXACTAMENTE el mismo modelo/datos que run_one_dim(seed=seed)
    en run_experiment.py, para poder extraer un x_base por indice y que sea
    el mismo punto que aparece en dim_{dim}.json con ese x_idx."""
    torch.manual_seed(seed); np.random.seed(seed)
    X, y = gaussian_binary_data(dim, n_samples=max(2000, 20 * dim), seed=seed)
    Xt, yt = to_torch_xy(X, y)
    model, info = train_classifier_with_retry(
        lambda: VarMLP(dim, n_layers, out_size=1), Xt, yt, device, seed=seed)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f'  modelo reconstruido: seed_used={info["seed_used"]}  '
         f'train_acc={info["final_train_acc"]:.4f}  '
         f'spectral_check={info["spectral_check"]:.4f}')
    return model, X


def diagnose_point(model, X, x_idx, r, mg_iters, device, lr=None, noise=None, rho=1.003, T0=1.0, run_seed=200,
                   verbose=True):
    x0 = torch.tensor(X[x_idx], dtype=torch.float32, device=device)
    lr_show = lr if lr is not None else 2.0 * r
    noise_show = noise if noise is not None else 0.5 * r
    print(f'\n{"="*70}\n  x_idx = {x_idx}   (r={r}, lr={lr_show:.5f}, noise={noise_show:.5f}, '
         f'lr/noise={lr_show/noise_show:.2f}, rho={rho}, T0={T0}, mg_iters={mg_iters}, run_seed={run_seed})\n{"="*70}')

    results = {}
    for rule in ('metropolis', 'mh'):
        print(f'\n  --- accept_rule = {rule!r} ---')
        # run_seed controla el punto de partida y las muestras gaussianas
        # subyacentes en cada iteracion; misma run_seed para ambas reglas
        # aisla el efecto de la regla de aceptacion. Variar run_seed entre
        # llamadas permite distinguir un sesgo estructural (persiste con
        # cualquier semilla) de una semilla especifica desafortunada.
        torch.manual_seed(run_seed); np.random.seed(run_seed)
        res = montelip_g(model, x0, r_domain=r, lr=lr, noise=noise,
                         T0=T0, rho=rho, max_iters=mg_iters, accept_rule=rule,
                         record_history=True, verbose=verbose)
        results[rule] = res
        print(f'  L_estimate={res.L_estimate:.6f}  iters={res.iters}  '
             f'accepted_up={res.accepted_up}  accepted_down={res.accepted_down}  '
             f'final_dist={res.final_dist:.6f}  final_T={res.final_T:.2e}')

    # Punto donde ambas trayectorias de 'best-so-far' divergen por primera vez
    h_m = results['metropolis'].history
    h_h = results['mh'].history
    n = min(len(h_m), len(h_h))
    diverge_at = None
    for k in range(n):
        if abs(h_m[k] - h_h[k]) > 1e-9:
            diverge_at = k
            break
    print(f'\n  Primera divergencia en la historia de "best": '
         f'{"iteracion " + str(diverge_at) if diverge_at is not None else "NINGUNA (identicas)"}')
    if diverge_at is not None:
        lo = max(0, diverge_at - 2)
        hi = min(n, diverge_at + 5)
        print(f'  Contexto (iters {lo}..{hi-1}):')
        print(f'    metropolis: {[round(v,4) for v in h_m[lo:hi]]}')
        print(f'    mh:         {[round(v,4) for v in h_h[lo:hi]]}')

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dim', type=int, default=784)
    ap.add_argument('--n-layers', type=int, default=5)
    ap.add_argument('--seed', type=int, default=1,
                    help='seed de entrenamiento/datos; usar 1 para reproducir results_seed1')
    ap.add_argument('--r', type=float, default=0.05)
    ap.add_argument('--x-idx', type=str, default='8023,7417',
                    help='indices de punto base a diagnosticar, separados por coma. '
                         'Default: los dos puntos que colapsaron en results_seed1/dim_784.json. '
                         'Incluye 546 como control (ese punto SI coincidio entre reglas).')
    ap.add_argument('--mg-iters', type=int, default=3000)
    ap.add_argument('--lr', type=float, default=None,
                    help='paso de gradiente ABSOLUTO, desacoplado de r (default: 2*r, el original). '
                         'Usa esto para probar la hipotesis de que lr/noise, no r, es la causa del colapso.')
    ap.add_argument('--noise', type=float, default=None,
                    help='escala de ruido inicial ABSOLUTA, desacoplada de r (default: 0.5*r).')
    ap.add_argument('--rho', type=float, default=1.003,
                    help='razon de enfriamiento geometrico T <- T/rho (default 1.003, el original). '
                         'Un rho mas cercano a 1 alarga la ventana de T alta y el numero de '
                         'iteraciones hasta T_min, dando mas presupuesto de exploracion.')
    ap.add_argument('--T0', type=float, default=1.0, help='temperatura inicial (default 1.0).')
    ap.add_argument('--n-seeds', type=int, default=1,
                    help='si es >1, repite el diagnostico con n_seeds semillas internas '
                         'distintas (run_seed=200, 201, 202, ...) y reporta la distribucion '
                         'de L_estimate para metropolis vs mh, en silencio (sin el log '
                         'verbose por iteracion). Sirve para distinguir un sesgo estructural '
                         '(mh queda sistematicamente por debajo, con cualquier semilla) de '
                         'una semilla especifica desafortunada.')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    device = torch.device(args.device)
    x_idxs = [int(v) for v in args.x_idx.split(',')]

    print(f'Reconstruyendo modelo: dim={args.dim}  n_layers={args.n_layers}  '
         f'seed={args.seed}  device={device}')
    model, X = rebuild_model_and_data(args.dim, args.n_layers, args.seed, device)

    if args.n_seeds <= 1:
        for x_idx in x_idxs:
            diagnose_point(model, X, x_idx, args.r, args.mg_iters, device,
                           lr=args.lr, noise=args.noise, rho=args.rho, T0=args.T0)
    else:
        for x_idx in x_idxs:
            print(f'\n{"#"*70}\n  BARRIDO DE SEMILLAS -- x_idx={x_idx}  '
                 f'({args.n_seeds} semillas: 200..{200+args.n_seeds-1})\n{"#"*70}')
            met_vals, mh_vals = [], []
            for k in range(args.n_seeds):
                run_seed = 200 + k
                res = diagnose_point(model, X, x_idx, args.r, args.mg_iters, device,
                                     lr=args.lr, noise=args.noise, rho=args.rho, T0=args.T0,
                                     run_seed=run_seed, verbose=False)
                met_vals.append(res['metropolis'].L_estimate)
                mh_vals.append(res['mh'].L_estimate)
            met_vals = np.array(met_vals); mh_vals = np.array(mh_vals)
            print(f'\n  --- Resumen x_idx={x_idx} sobre {args.n_seeds} semillas ---')
            print(f'  metropolis: media={met_vals.mean():.4f}  std={met_vals.std():.4f}  '
                 f'min={met_vals.min():.4f}  max={met_vals.max():.4f}')
            print(f'  mh:         media={mh_vals.mean():.4f}  std={mh_vals.std():.4f}  '
                 f'min={mh_vals.min():.4f}  max={mh_vals.max():.4f}')
            n_mh_below = int((mh_vals < met_vals).sum())
            print(f'  mh < metropolis en {n_mh_below}/{args.n_seeds} semillas '
                 f'({"sesgo consistente -> estructural" if n_mh_below == args.n_seeds else "mixto -> revisar"})')


if __name__ == '__main__':
    main()
