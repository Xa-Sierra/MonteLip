"""
functions_gauss.py
-------------------
Tarea de clasificacion sintetica (dos gaussianas separables) para las
redes pequenas del experimento de colapso dimensional. Reemplaza XOR/
sinusoidal -- no se necesita ninguna propiedad Lipschitz conocida de la
tarea, solo una red no trivial ya entrenada (ver README.md de este
directorio para el porque).

Clase 0 ~ N(-mu, sigma^2 I), Clase 1 ~ N(+mu, sigma^2 I), mu = (sep/2)*1.
Con sep=2.0 y sigma=1.0 las clases se solapan un poco (tarea no trivial,
frontera de decision no degenerada) pero siguen siendo separables con
buena precision por un MLP pequeno.
"""
import numpy as np
import torch


def gaussian_binary_data(dim, n_samples=2000, sep=2.0, sigma=1.0, seed=0):
    rng = np.random.default_rng(seed)
    n0 = n_samples // 2
    n1 = n_samples - n0
    mu = np.full(dim, sep / 2.0, dtype=np.float32)
    X0 = rng.normal(-mu, sigma, size=(n0, dim)).astype(np.float32)
    X1 = rng.normal(mu, sigma, size=(n1, dim)).astype(np.float32)
    X = np.concatenate([X0, X1], axis=0)
    y = np.concatenate([np.zeros(n0), np.ones(n1)]).astype(np.float32)
    perm = rng.permutation(n_samples)
    return X[perm], y[perm]


def to_torch_xy(X, y, device='cpu'):
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(-1)
    return Xt, yt


def select_correctly_classified_base_points(model, X, y, n_points=5, seed=0,
                                             device='cpu'):
    """Elige n_points ejemplos que el modelo clasifica correctamente
    (logit del signo correcto), para que el punto base sea representativo
    del comportamiento 'tipico' de la red, no un caso mal aprendido."""
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(Xt).squeeze(-1).cpu().numpy()
    pred = (logits > 0).astype(np.float32)
    correct_idx = np.where(pred == y)[0]
    rng = np.random.default_rng(seed)
    idx = rng.choice(correct_idx, min(n_points, len(correct_idx)), replace=False)
    return X[idx], idx
