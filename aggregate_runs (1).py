#!/usr/bin/env python3
"""
aggregate_runs.py - Combina varias corridas de run_experiment.py (cada una
con un --seed distinto, cada una en su propio --out-dir) en una sola
media +/- std por dimension, que ahora captura tanto la variabilidad de
los puntos base COMO la variabilidad del entrenamiento de la red (semilla
distinta = red distinta).

CORRECCION (tras revisar seed0 y seed1): la referencia usada para calcular
"% de recuperacion" ahora es LipBaB (exacto) cuando esta disponible
(dim<=lipbab_max_dim), en vez de L_Jac muestreada. Se confirmo en dim=2
que N=50 muestras no basta para cubrir bien la bola en dimension muy baja
-- L_Jac quedaba ~7% por debajo del valor real (verificado porque LipBaB,
que es exacto, daba consistentemente ~107% de "recuperacion" contra esa
referencia subestimada). Donde LipBaB no corre (dim>lipbab_max_dim), se
sigue usando L_Jac muestreada como antes. El campo 'reference_used' en la
salida documenta, por dimension, cual referencia se uso.

Uso (despues de correr, por ejemplo):
    python run_experiment.py --seed 0 --out-dir results_seed0
    python run_experiment.py --seed 1 --out-dir results_seed1
    python run_experiment.py --seed 2 --out-dir results_seed2

    python aggregate_runs.py --dirs results_seed0,results_seed1,results_seed2
                              --out-dir results_combined

Output: results_combined/summary_combined.json
"""
import argparse
import json
import os

import numpy as np


METHODS = ['montelip_g', 'montelip_iso', 'random', 'pgd_multistart',
          'clever', 'analytical', 'lipbab']


def load_dim_json(out_dir, dim):
    path = f'{out_dir}/dim_{dim}.json'
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


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
    ap.add_argument('--dirs', type=str, required=True,
                    help='Carpetas de resultados separadas por coma, ej. '
                         'results_seed0,results_seed1,results_seed2')
    ap.add_argument('--dims', type=str, default='2,5,10,20,50,100,200,784')
    ap.add_argument('--out-dir', type=str, default='results_combined')
    args = ap.parse_args()

    dirs = args.dirs.split(',')
    dims = [int(d) for d in args.dims.split(',')]
    os.makedirs(args.out_dir, exist_ok=True)

    combined = {}
    for dim in dims:
        pooled = {m: [] for m in METHODS}
        train_accs = []
        n_runs_found = 0
        ref_sources = []

        for d in dirs:
            res = load_dim_json(d, dim)
            if res is None:
                print(f'AVISO: {d}/dim_{dim}.json no encontrado, se omite '
                     f'esa corrida para dim={dim}')
                continue
            n_runs_found += 1
            train_accs.append(res['train_info'].get('final_train_acc'))
            for row in res['records']:
                # Referencia: LipBaB (exacto) cuando esta disponible, ya que
                # es mas confiable que L_Jac muestreada (N=50) -- en
                # dimension baja (dim=2) confirmamos que N=50 no basta para
                # cubrir bien la bola y L_Jac queda por debajo del valor
                # real, distorsionando los % de recuperacion de TODOS los
                # metodos. LipBaB solo corre para dim<=lipbab_max_dim, asi
                # que en dimensiones altas se sigue usando jac_ball.
                lb_ref = row.get('lipbab', {})
                if lb_ref.get('L_estimate') is not None:
                    ref = lb_ref['L_estimate']
                    ref_source = 'lipbab'
                else:
                    ref = row['jac_ball']['L_estimate']
                    ref_source = 'jac_ball'
                if ref <= 0:
                    continue
                pooled['montelip_g'].append(row['montelip_g']['L_estimate'] / ref)
                pooled['montelip_iso'].append(row['montelip_iso']['L_estimate'] / ref)
                pooled['random'].append(row['random']['L_estimate'] / ref)
                pooled['pgd_multistart'].append(row['pgd_multistart']['L_estimate'] / ref)
                cl = row.get('clever', {})
                if cl.get('L_estimate') is not None:
                    pooled['clever'].append(cl['L_estimate'] / ref)
                an = row.get('analytical', {})
                if an.get('L_local') is not None:
                    pooled['analytical'].append(an['L_local'] / ref)
                # lipbab ya no se agrega a 'pooled' contra si mismo cuando
                # es la referencia (daria trivialmente 100%); se mantiene
                # SOLO como diagnostico de que tan cerca estaba jac_ball,
                # ver 'lipbab_vs_jacball' mas abajo
                if ref_source == 'jac_ball':
                    lb = row.get('lipbab', {})
                    if lb.get('L_estimate') is not None:
                        pooled['lipbab'].append(lb['L_estimate'] / ref)
                ref_sources.append(ref_source)

        entry = {'n_runs_pooled': n_runs_found,
                'train_acc_mean': float(np.mean(train_accs)) if train_accs else None,
                'train_acc_std': float(np.std(train_accs)) if train_accs else None,
                'reference_used': ('lipbab' if ref_sources and all(r == 'lipbab' for r in ref_sources)
                                   else 'jac_ball' if ref_sources and all(r == 'jac_ball' for r in ref_sources)
                                   else 'mixed' if ref_sources else None)}
        for k, v in pooled.items():
            if v:
                entry[k] = float(np.mean(v))
                entry[f'{k}_std'] = float(np.std(v))
                entry[f'{k}_n'] = len(v)  # total de observaciones (puntos x corridas)
            else:
                entry[k] = None
                entry[f'{k}_std'] = None
                entry[f'{k}_n'] = 0
        combined[dim] = entry

    with open(f'{args.out_dir}/summary_combined.json', 'w') as f:
        json.dump(to_serializable(combined), f, indent=2)

    print(f'\nGuardado: {args.out_dir}/summary_combined.json')
    print('\nResumen combinado (media +/- std, pooled sobre corridas x puntos base):')
    for dim, s in combined.items():
        if s['n_runs_pooled'] == 0:
            print(f'  dim={dim:4d}: SIN DATOS en ninguna corrida')
            continue
        print(f'  dim={dim:4d}  [{s["n_runs_pooled"]} corridas, '
             f'train_acc={s["train_acc_mean"]:.3f}+-{s["train_acc_std"]:.3f}, '
             f'referencia={s["reference_used"]}]')
        for m in METHODS:
            if s.get(m) is not None:
                print(f'    {m:16s}: {s[m]:.1%} +- {s[f"{m}_std"]:.1%}  (n={s[f"{m}_n"]})')


if __name__ == '__main__':
    main()
