"""
analytical_adapter.py
----------------------
Adaptador delgado para usar la libreria REAL de Avant & Morgansen
(network_bound.py, utils.py, tal como los subiste) con VarMLP, en vez de
reimplementar el bound a mano. Requiere que network_bound.py, utils.py y
my_config.py esten en el mismo directorio o en el PYTHONPATH.

No modifica network_bound.py ni utils.py -- la correccion vive enteramente
en models.py (VarMLP ahora expone self.layers como nn.Sequential con
ReLU() como modulos explicitos, que es lo que network_bound.py necesita
para detectar el patron afin->ReLU).
"""
import time
import numpy as np
import torch


def analytical_bounds(model, x0, eps, batch_size=32):
    """
    model: VarMLP (o cualquier nn.Module cuyo .layers sea una lista plana
        Linear/ReLU compatible con network_bound.py).
    x0: torch.Tensor, shape (1, input_dim) o (input_dim,) -- se ajusta
        automaticamente a batch de tamano 1, como espera network_bound.py.
    eps: radio local (mismo sentido que r_domain en montelip.py: bola
        L_inf, o L2 segun la convencion interna de utils.get_RAD -- ver
        docstring de network_bound.local_bound).

    Devuelve dict {L_global, L_local, elapsed_global_s, elapsed_local_s}.
    """
    import network_bound  # requiere network_bound.py/utils.py/my_config.py en el path

    model.eval()
    if x0.dim() == 1:
        x0 = x0.unsqueeze(0)

    t0 = time.time()
    layer_bounds_global = network_bound.global_bound(model, x0)
    L_global = float(np.prod(layer_bounds_global))
    t_global = time.time() - t0

    t0 = time.time()
    L_local = float(network_bound.local_bound(model, x0, eps, batch_size=batch_size))
    t_local = time.time() - t0

    return {
        'L_global': L_global,
        'L_local': L_local,
        'elapsed_global_s': t_global,
        'elapsed_local_s': t_local,
        'per_layer_global': [float(v) for v in layer_bounds_global],
    }
