"""
clever.py
---------
CLEVER (Weng, Zhang, Chen, Yi, Su, Gao, Hsieh & Daniel, "Evaluating the
Robustness of Neural Networks: An Extreme Value Theory Approach", ICLR
2018) -- estimador de cota inferior de la constante de Lipschitz local
via muestreo de normas de gradiente + ajuste de distribucion Weibull
inversa (Extreme Value Theory).

ADAPTACION respecto al paper original (documentar en el paper):
El CLEVER original estima el "cross-Lipschitz" de g(x) = f_c(x) - f_j(x)
(margen entre dos clases), pensado para certificar distorsion adversarial
minima. Aqui adaptamos la MISMA maquinaria estadistica (Algoritmo 1 del
paper: muestreo por bloques, maximo por bloque, ajuste MLE de Weibull
inversa sobre los maximos de bloque) para estimar directamente
L(x0,chi) = sup_x ||J_f(x)||_2 de la red -- la misma cantidad que
estiman los demas metodos de este proyecto (MonteLip-G, jac_ball, etc.),
para una comparacion directa y justa. El dominio de muestreo es nuestra
bola L_inf de radio r (consistente con el resto del proyecto), no la
bola L_p generica del paper original.

Referencia de la distribucion: scipy.stats.weibull_max es exactamente la
"reverse Weibull" del paper (CDF con soporte superior finito en `loc`);
el parametro `loc` estimado por MLE es el analogo directo de a_W en el
paper -- la estimacion de CLEVER de la constante de Lipschitz local.
"""
import time

import numpy as np
import torch
from scipy import stats


def _jacobian_top_sv(model, x):
    """||J_f(x)||_2 via autodiff (identico a jacobian_top_sv en
    montelip.py, reproducido aqui para no crear dependencia circular)."""
    x = x.clone().detach().requires_grad_(True)
    out = model(x).reshape(-1)
    rows = []
    for i in range(out.numel()):
        g = torch.autograd.grad(out[i], x, retain_graph=(i < out.numel() - 1))[0]
        rows.append(g.reshape(-1))
    J = torch.stack(rows)
    return torch.linalg.svdvals(J)[0].item()


def clever_score(model, x_base, r_domain=0.05, n_batches=25, batch_size=50,
                 seed=None, verbose=False):
    """
    Estima L(x_base, chi) via CLEVER adaptado (ver docstring del modulo).

    n_batches (Nb) x batch_size (Ns) = total de evaluaciones de gradiente.
    El paper original usa Nb=500, Ns=1024 (demasiado costoso aqui); estos
    defaults (25x50=1250 evaluaciones) siguen su propia recomendacion de
    que Nb=50-100 ya es suficiente en la practica (Seccion 5.4 del paper,
    Figura 7): las estimaciones cambian poco al aumentar Nb mas alla de
    eso.

    Devuelve dict {L_estimate, elapsed_s, ks_pvalue, weibull_params,
    block_maxima} -- ks_pvalue es el p-valor del test de Kolmogorov-Smirnov
    de bondad de ajuste (el paper recomienda reportarlo: p>0.05 sugiere
    que el ajuste Weibull es razonable; p<=0.05 es una senal de alerta).
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    device = next(model.parameters()).device
    x_base = x_base.to(device).clone()
    lo, hi = x_base - r_domain, x_base + r_domain

    t0 = time.time()
    block_maxima = []
    for b in range(n_batches):
        batch_vals = []
        for s in range(batch_size):
            u = torch.rand_like(x_base) * 2 - 1
            x = (x_base + u * r_domain).clamp(lo, hi)
            sv = _jacobian_top_sv(model, x)
            batch_vals.append(sv)
        block_maxima.append(max(batch_vals))
        if verbose and (b + 1) % 5 == 0:
            print(f'  batch {b+1}/{n_batches}: max so far = {max(block_maxima):.4f}')

    block_maxima = np.array(block_maxima)

    # ajuste MLE de Weibull inversa (scipy.stats.weibull_max) sobre los
    # maximos de bloque -- loc = estimacion de a_W = CLEVER score
    try:
        c, loc, scale = stats.weibull_max.fit(block_maxima)
        ks_stat, ks_pvalue = stats.kstest(block_maxima, 'weibull_max',
                                          args=(c, loc, scale))
        L_estimate = float(loc)
        fit_status = 'OK'
    except Exception as e:
        # si el ajuste MLE falla (poco comun, pero puede pasar con pocos
        # datos o datos degenerados), usar el maximo empirico como
        # respaldo -- es una subestimacion conservadora, se documenta
        # explicitamente en el resultado
        L_estimate = float(block_maxima.max())
        c, loc, scale = None, None, None
        ks_stat, ks_pvalue = None, None
        fit_status = f'MLE_FAILED_FALLBACK_TO_MAX: {e}'

    elapsed = time.time() - t0
    return {
        'L_estimate': L_estimate,
        'elapsed_s': elapsed,
        'n_batches': n_batches,
        'batch_size': batch_size,
        'fit_status': fit_status,
        'weibull_params': {'c': c, 'loc': loc, 'scale': scale},
        'ks_statistic': ks_stat,
        'ks_pvalue': ks_pvalue,
        'block_maxima': block_maxima.tolist(),
    }
