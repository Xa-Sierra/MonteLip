#!/usr/bin/env python3
"""
run_experiment.py - Barrido de dimension creciente para demostrar el
colapso dimensional de MonteLip isotropico y validar MonteLip-G.

Tarea: clasificacion binaria de dos gaussianas separables (functions_gauss.py).
Dimensiones: 2, 5, 10, 20, 50, 100, 200, 784 (conecta directo con el
input_dim de MNIST usado en el codigo de 096_MBA).

Requiere en el mismo directorio (o PYTHONPATH):
    models.py, functions_gauss.py, analytical_adapter.py, lipbab_wrapper.py
    montelip.py, montelip_ball_sampling_ADDITION.py  (los que ya tienes)
    network_bound.py, utils.py, my_config.py          (Avant & Morgansen)

Uso:
    python run_experiment.py [--dims 2,5,10,20,50,100,200,784]
                              [--r 0.05] [--n-points 5] [--n-layers 5]
                              [--lipbab-max-dim 20] [--skip-lipbab]
                              [--skip-analytical]

Output: results/dim_{d}.json  (uno por dimension)
        results/summary.json  (recuperacion agregada, para la figura)
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from models import VarMLP
from functions_gauss import (gaussian_binary_data, to_torch_xy,
                             select_correctly_classified_base_points)
from montelip import (montelip_g, montelip, random_search_local,
                      gradient_ascent_local, spectral_bound)
from montelip_ball_sampling_ADDITION import jacobian_ball_sampled

try:
    from analytical_adapter import analytical_bounds
    HAS_ANALYTICAL = True
except Exception as e:
    print(f'AVISO: analytical_adapter no disponible ({e}); se omite Analitico.')
    HAS_ANALYTICAL = False

try:
    from lipbab_wrapper import run_lipbab
    HAS_LIPBAB = True
except Exception as e:
    print(f'AVISO: lipbab_wrapper no disponible ({e}); se omite LipBaB.')
    HAS_LIPBAB = False

try:
    from lipsdp import lipsdp_bound, sanity_check as lipsdp_sanity_check
    HAS_LIPSDP = True
except Exception as e:
    print(f'AVISO: lipsdp no disponible ({e}); se omite LipSDP.')
    HAS_LIPSDP = False

try:
    from clever import clever_score
    HAS_CLEVER = True
except Exception as e:
    print(f'AVISO: clever no disponible ({e}); se omite CLEVER.')
    HAS_CLEVER = False


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


def train_classifier(model, Xt, yt, device, epochs=2000, lr=1e-2):
    model = model.to(device)
    Xt, yt = Xt.to(device), yt.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    t0 = time.time()
    final_loss = None
    for ep in range(epochs):
        opt.zero_grad()
        out = model(Xt)
        loss = loss_fn(out, yt)
        loss.backward()
        opt.step()
        final_loss = loss.item()
        if ep % 500 == 0:
            with torch.no_grad():
                acc = ((out > 0).float() == yt).float().mean().item()
            print(f'    epoch {ep}: loss={loss.item():.4f}  acc={acc:.3f}')
    with torch.no_grad():
        acc = ((model(Xt) > 0).float() == yt).float().mean().item()
    return model, {'final_loss': final_loss, 'final_train_acc': acc,
                   'elapsed_s': time.time() - t0}


def train_classifier_with_retry(model_factory, Xt, yt, device, epochs=2000,
                                lr=1e-2, min_spectral=1e-6, max_retries=5, seed=0):
    """
    Entrena y verifica que la red NO quedo muerta (ReLUs colapsadas ->
    salida constante -> Lipschitz identicamente cero en todos lados, nada
    que medir). El chequeo es directo sobre la cantidad que importa (cota
    espectral > min_spectral), no un proxy via accuracy de clasificacion
    -- no nos interesa que la red clasifique bien, solo que no sea
    degenerada. Un accuracy mediocre (70-80%) en la zona de traslape de
    las gaussianas es perfectamente aceptable y no debe disparar reintento.
    """
    for attempt in range(max_retries):
        torch.manual_seed(seed + attempt * 1000)
        model = model_factory()
        model, info = train_classifier(model, Xt, yt, device, epochs=epochs, lr=lr)
        spec = spectral_bound(model)['L_estimate']
        info['spectral_check'] = spec
        if spec > min_spectral:
            info['seed_used'] = seed + attempt * 1000
            info['attempts'] = attempt + 1
            return model, info
        print(f'    *** red degenerada (cota espectral={spec:.2e}, acc={info["final_train_acc"]:.3f}, '
             f'probable dying-ReLU) -- reintentando con otra semilla '
             f'({attempt+1}/{max_retries}) ***')
    info['seed_used'] = seed + attempt * 1000
    info['attempts'] = max_retries
    info['FAILED_TO_CONVERGE'] = True
    print(f'    *** ADVERTENCIA: red sigue degenerada tras {max_retries} intentos -- '
         f'revisar manualmente, los resultados de esta dimension no son confiables ***')
    return model, info


def run_one_dim(dim, r, n_points, n_layers, device, lipbab_max_dim,
                skip_lipbab, skip_analytical, mg_iters, iso_iters,
                jac_samples, skip_lipsdp=False, lipsdp_max_dim=100, seed=0):
    print(f'\n=== dim = {dim} ===')
    torch.manual_seed(seed); np.random.seed(seed)

    X, y = gaussian_binary_data(dim, n_samples=max(2000, 20 * dim), seed=seed)
    Xt, yt = to_torch_xy(X, y)

    model, train_info = train_classifier_with_retry(
        lambda: VarMLP(dim, n_layers, out_size=1), Xt, yt, device, seed=seed)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    x_bases, base_idx = select_correctly_classified_base_points(
        model, X, y, n_points=n_points, seed=seed, device=device)

    spec = spectral_bound(model)
    print(f'  cota espectral global: {spec["L_estimate"]:.4f}')

    out = {'dim': dim, 'n_layers': n_layers, 'r': r, 'train_info': train_info,
          'spectral': spec, 'records': []}

    if HAS_LIPSDP and not skip_lipsdp and dim <= lipsdp_max_dim:
        try:
            sdp = lipsdp_bound(model.get_weight_bias_list())
            out['lipsdp'] = sdp
            print(f'  LipSDP (global):     {sdp["L_estimate"]}  ({sdp["elapsed_s"]:.2f}s)  status={sdp["status"]}')
            # verificacion de consistencia con el primer punto base
            x0_check = torch.tensor(x_bases[0], dtype=torch.float32, device=device)
            chk = lipsdp_sanity_check(model.get_weight_bias_list(), x0_check, r,
                                      jacobian_ball_sampled, model, spectral_bound,
                                      verbose=True)
            out['lipsdp_sanity_check'] = chk['ok']
            if not chk['ok']:
                print('  *** ADVERTENCIA: sanity check de LipSDP FALLO -- no confiar en estos numeros ***')
        except Exception as e:
            out['lipsdp'] = {'error': str(e)}
            print(f'  LipSDP: ERROR: {e}')
    else:
        if not HAS_LIPSDP:
            reason = 'modulo lipsdp no se pudo importar (revisar cvxpy instalado)'
        elif skip_lipsdp:
            reason = '--skip-lipsdp activo'
        else:
            reason = f'dim={dim} > lipsdp_max_dim={lipsdp_max_dim}'
        out['lipsdp'] = {'skipped': True, 'reason': reason}

    for i, x0_np in enumerate(x_bases):
        x0 = torch.tensor(x0_np, dtype=torch.float32, device=device)
        row = {'x_idx': int(base_idx[i])}
        print(f'  punto base {i+1}/{len(x_bases)}')

        jb = jacobian_ball_sampled(model, x0, r_domain=r, n_samples=jac_samples, seed=seed)
        row['jac_ball'] = {'L_estimate': jb['L_estimate'], 'elapsed_s': jb['elapsed_s']}
        print(f'    L_Jac (referencia):  {jb["L_estimate"]:.4f}  ({jb["elapsed_s"]:.2f}s)')

        torch.manual_seed(200 + i); np.random.seed(200 + i)
        mg = montelip_g(model, x0, r_domain=r, T0=1.0, rho=1.003, max_iters=mg_iters)
        row['montelip_g'] = {'L_estimate': mg.L_estimate, 'elapsed_s': mg.elapsed_s}
        print(f'    MonteLip-G:          {mg.L_estimate:.4f}  ({mg.elapsed_s:.2f}s)')

        torch.manual_seed(200 + i); np.random.seed(200 + i)
        ml = montelip(model, x0, r_domain=r, eps=r, T0=3.0, rho=1.0001, max_iters=iso_iters)
        row['montelip_iso'] = {'L_estimate': ml.L_estimate, 'elapsed_s': ml.elapsed_s}
        print(f'    MonteLip-iso:        {ml.L_estimate:.4f}  ({ml.elapsed_s:.2f}s)')

        torch.manual_seed(300 + i); np.random.seed(300 + i)
        rs = random_search_local(model, x0, r_domain=r, eps=r, n_iters=ml.iters)
        row['random'] = rs
        print(f'    Random search:       {rs["L_estimate"]:.4f}  ({rs["elapsed_s"]:.2f}s)')

        ga = gradient_ascent_local(model, x0, r_domain=r, n_restarts=5, n_steps=200)
        row['pgd_multistart'] = ga
        print(f'    PGD-multistart:      {ga["L_estimate"]:.4f}  ({ga["elapsed_s"]:.2f}s)')

        if HAS_CLEVER:
            try:
                cl = clever_score(model, x0, r_domain=r, n_batches=25, batch_size=50, seed=i)
                row['clever'] = {k: v for k, v in cl.items() if k != 'block_maxima'}
                print(f'    CLEVER:              {cl["L_estimate"]:.4f}  ({cl["elapsed_s"]:.2f}s)  '
                     f'ks_pvalue={cl["ks_pvalue"]}')
            except Exception as e:
                row['clever'] = {'error': str(e)}
                print(f'    CLEVER:              ERROR: {e}')

        if HAS_ANALYTICAL and not skip_analytical:
            try:
                an = analytical_bounds(model, x0, eps=r)
                row['analytical'] = an
                print(f'    Analitico (local):   {an["L_local"]:.4f}  ({an["elapsed_local_s"]:.2f}s)')
            except Exception as e:
                row['analytical'] = {'error': str(e)}
                print(f'    Analitico:           ERROR: {e}')

        if HAS_LIPBAB and not skip_lipbab and dim <= lipbab_max_dim:
            try:
                lb = run_lipbab(model.get_weight_bias_list(), x0_np, r, max_iters=2000)
                row['lipbab'] = lb
                print(f'    LipBaB:              {lb["L_estimate"]:.4f}  ({lb["elapsed_s"]:.2f}s)  status={lb["status"]}')
            except Exception as e:
                row['lipbab'] = {'error': str(e)}
                print(f'    LipBaB:              ERROR: {e}')
        else:
            if not HAS_LIPBAB:
                reason = 'lipbab_wrapper no se pudo importar (revisar cvxopt instalado)'
            elif skip_lipbab:
                reason = '--skip-lipbab activo'
            else:
                reason = f'dim={dim} > lipbab_max_dim={lipbab_max_dim}'
            row['lipbab'] = {'skipped': True, 'reason': reason}

        out['records'].append(row)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dims', type=str, default='2,5,10,20,50,100,200,784')
    ap.add_argument('--r', type=float, default=0.05)
    ap.add_argument('--n-points', type=int, default=5)
    ap.add_argument('--n-layers', type=int, default=5)
    ap.add_argument('--mg-iters', type=int, default=3000)
    ap.add_argument('--iso-iters', type=int, default=30000)
    ap.add_argument('--jac-samples', type=int, default=50)
    ap.add_argument('--lipbab-max-dim', type=int, default=20,
                    help='LipBaB se salta para dim > este valor (escala mal)')
    ap.add_argument('--lipsdp-max-dim', type=int, default=100,
                    help='LipSDP se salta para dim > este valor (SDP escala mal: '
                         'ancho de capa = dim en VarMLP, la SDP crece como O(dim*n_layers))')
    ap.add_argument('--skip-lipbab', action='store_true')
    ap.add_argument('--skip-analytical', action='store_true')
    ap.add_argument('--skip-lipsdp', action='store_true')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out-dir', type=str, default='results',
                    help='Carpeta de salida -- usa una distinta por --seed '
                         'si vas a correr varias semillas para promediar '
                         '(ej. results_seed0, results_seed1, ...)')
    args = ap.parse_args()

    dims = [int(d) for d in args.dims.split(',')]
    device = torch.device(args.device)
    print(f'device: {device}')
    print(f'dims: {dims}')

    os.makedirs(args.out_dir, exist_ok=True)
    all_results = {}
    for dim in dims:
        try:
            res = run_one_dim(dim, args.r, args.n_points, args.n_layers, device,
                              args.lipbab_max_dim, args.skip_lipbab, args.skip_analytical,
                              args.mg_iters, args.iso_iters, args.jac_samples,
                              skip_lipsdp=args.skip_lipsdp,
                              lipsdp_max_dim=args.lipsdp_max_dim, seed=args.seed)
        except Exception as e:
            print(f'\n*** FALLO en dim={dim}: {e} -- se continua con el resto del barrido ***\n')
            with open(f'{args.out_dir}/dim_{dim}_FAILED.json', 'w') as f:
                json.dump({'dim': dim, 'error': str(e)}, f, indent=2)
            continue
        with open(f'{args.out_dir}/dim_{dim}.json', 'w') as f:
            json.dump(to_serializable(res), f, indent=2)
        all_results[dim] = res
        print(f'  guardado: {args.out_dir}/dim_{dim}.json')

    # resumen: media +/- std de cada metodo (sobre los n_points puntos base)
    # vs jac_ball (o vs spectral para LipSDP, que es cota GLOBAL, no local)
    summary = {}
    for dim, res in all_results.items():
        recov = {'montelip_g': [], 'montelip_iso': [], 'random': [],
                 'pgd_multistart': [], 'clever': [], 'analytical': [], 'lipbab': []}
        for row in res['records']:
            ref = row['jac_ball']['L_estimate']
            if ref <= 0:
                continue
            recov['montelip_g'].append(row['montelip_g']['L_estimate'] / ref)
            recov['montelip_iso'].append(row['montelip_iso']['L_estimate'] / ref)
            recov['random'].append(row['random']['L_estimate'] / ref)
            recov['pgd_multistart'].append(row['pgd_multistart']['L_estimate'] / ref)
            cl = row.get('clever', {})
            if cl.get('L_estimate') is not None:
                recov['clever'].append(cl['L_estimate'] / ref)
            an = row.get('analytical', {})
            if an.get('L_local') is not None:
                recov['analytical'].append(an['L_local'] / ref)
            lb = row.get('lipbab', {})
            if lb.get('L_estimate') is not None:
                recov['lipbab'].append(lb['L_estimate'] / ref)

        entry = {}
        for k, v in recov.items():
            if v:
                entry[k] = float(np.mean(v))
                entry[f'{k}_std'] = float(np.std(v))
                entry[f'{k}_n'] = len(v)
            else:
                entry[k] = None
                entry[f'{k}_std'] = None
                entry[f'{k}_n'] = 0
        # LipSDP es cota GLOBAL: se reporta aparte, no como "recuperacion"
        # local, sino como razon respecto a la cota espectral naive (que
        # tambien es global) -- entre mas chico, mas ajustada la cota.
        sdp = res.get('lipsdp', {})
        spec_L = res['spectral']['L_estimate']
        if sdp.get('L_estimate') is not None and spec_L > 0:
            entry['lipsdp_vs_spectral'] = float(sdp['L_estimate'] / spec_L)
            entry['lipsdp_sanity_ok'] = res.get('lipsdp_sanity_check')
        else:
            entry['lipsdp_vs_spectral'] = None
        entry['train_acc'] = res['train_info'].get('final_train_acc')
        entry['train_attempts'] = res['train_info'].get('attempts')
        summary[dim] = entry

    with open(f'{args.out_dir}/summary.json', 'w') as f:
        json.dump(to_serializable(summary), f, indent=2)

    print('\n' + '=' * 60)
    print(f'Resumen (media +/- std sobre {args.n_points} puntos base; '
         f'n < {args.n_points} indica que algun estimador fallo en algun punto):')
    for dim, s in summary.items():
        def pct(k):
            if s.get(k) is None:
                return 'n/a'
            std = s.get(f'{k}_std', 0) or 0
            return f'{s[k]:.1%}+-{std:.1%}(n={s.get(f"{k}_n","?")})'
        print(f'  dim={dim:4d}  [acc={s["train_acc"]:.3f}, intentos={s["train_attempts"]}]')
        print(f'    G={pct("montelip_g")}  iso={pct("montelip_iso")}  '
              f'random={pct("random")}  pgd={pct("pgd_multistart")}  clever={pct("clever")}')
        print(f'    analitico={pct("analytical")}  lipbab={pct("lipbab")}  '
              f'lipsdp/spectral={s.get("lipsdp_vs_spectral")}')


if __name__ == '__main__':
    main()
