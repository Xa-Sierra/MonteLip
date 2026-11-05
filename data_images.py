"""
data_images.py
---------------
Loaders de MNIST y CIFAR-10, y seleccion de puntos base para la
comparacion de estimadores de Lipschitz. MNIST reusa la misma
normalizacion que 01_train.py/utils.py de 096_MBA (media/std fijas);
CIFAR-10 usa las estadisticas estandar del dataset.
"""
import numpy as np
import torch
import torchvision
import torchvision.transforms as T


MNIST_MEAN, MNIST_STD = 0.1307, 0.3081
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def get_mnist_loaders(data_root='./data', batch_size_train=128, batch_size_test=512):
    transform = T.Compose([T.ToTensor(), T.Normalize((MNIST_MEAN,), (MNIST_STD,))])
    train_ds = torchvision.datasets.MNIST(data_root, train=True, download=True, transform=transform)
    test_ds = torchvision.datasets.MNIST(data_root, train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size_train, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size_test, shuffle=False)
    return train_loader, test_loader


def get_mnist_test(data_root='./data'):
    transform = T.Compose([T.ToTensor(), T.Normalize((MNIST_MEAN,), (MNIST_STD,))])
    return torchvision.datasets.MNIST(data_root, train=False, download=True, transform=transform)


def get_cifar10_loaders(data_root='./data', batch_size_train=128, batch_size_test=512,
                        augment=True):
    """augment=True aplica flip/crop aleatorio en train (estandar para CIFAR-10 desde
    cero); test siempre sin augmentacion."""
    train_tf_list = []
    if augment:
        train_tf_list += [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()]
    train_tf_list += [T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)]
    train_transform = T.Compose(train_tf_list)
    test_transform = T.Compose([T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])

    train_ds = torchvision.datasets.CIFAR10(data_root, train=True, download=True,
                                            transform=train_transform)
    test_ds = torchvision.datasets.CIFAR10(data_root, train=False, download=True,
                                           transform=test_transform)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size_train, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size_test, shuffle=False)
    return train_loader, test_loader


def get_cifar10_test(data_root='./data'):
    """Sin augmentacion -- para seleccionar puntos base de evaluacion."""
    transform = T.Compose([T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
    return torchvision.datasets.CIFAR10(data_root, train=False, download=True, transform=transform)


def select_base_points_from_dataset(model, test_dataset, n_points=5, seed=0, device='cpu',
                                    require_correct=True):
    """Selecciona n_points ejemplos del test set. Si require_correct=True (default),
    solo entre los que el modelo clasifica correctamente -- mismo criterio que
    select_correctly_classified_base_points en functions_gauss.py, adaptado a
    datasets de imagenes (indexados, no arrays en memoria)."""
    model.eval()
    rng = np.random.default_rng(seed)

    if not require_correct:
        idx = rng.choice(len(test_dataset), n_points, replace=False)
        x_bases = [test_dataset[i][0].unsqueeze(0).to(device) for i in idx]
        labels = [int(test_dataset[i][1]) for i in idx]
        return x_bases, list(idx), labels

    # evalua un subconjunto candidato (mas grande que n_points) y filtra correctos
    n_candidates = min(len(test_dataset), max(500, n_points * 20))
    cand_idx = rng.choice(len(test_dataset), n_candidates, replace=False)
    correct_idx = []
    with torch.no_grad():
        for i in cand_idx:
            x, y = test_dataset[i]
            x = x.unsqueeze(0).to(device)
            pred = model(x).argmax(dim=1).item()
            if pred == y:
                correct_idx.append(i)
            if len(correct_idx) >= n_points:
                break
    if len(correct_idx) < n_points:
        raise RuntimeError(f'Solo se encontraron {len(correct_idx)} ejemplos correctos '
                           f'de {n_candidates} candidatos -- revisa que el modelo este '
                           f'bien entrenado, o aumenta n_candidates.')

    x_bases = [test_dataset[i][0].unsqueeze(0).to(device) for i in correct_idx]
    labels = [int(test_dataset[i][1]) for i in correct_idx]
    return x_bases, correct_idx, labels
