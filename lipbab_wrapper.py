"""
lipbab_wrapper.py
------------------
Algoritmo LipBaB (Bhowmick, D'Souza, Raghavan, ICANN 2021) -- las funciones
centrales de abajo (rdn, rup, ii_*, mm_mul, mv_spmul, mim_mul, ivim_spmul,
e_bounds, symprop, LipNet, lip_bound, linprop, ffilter, branch) son las
del script original de los autores (LipBaB__TEST_v3.py), SIN MODIFICAR.

Lo unico que se reescribe es la seccion de integracion ("custom network" en
el script original), que ahora es la funcion run_lipbab() al final:
generica para cualquier VarMLP, con el dominio correcto (bola L_inf de
radio r alrededor de x0, consistente con montelip.py y exact_methods.py) en
vez del hipercubo unitario [0,1]^n que tenia hardcodeado el script
original.

Requiere: cvxopt (con solver glpk, ambos libres/open-source -- no necesita
Gurobi, consistente con la decision de omitir LipMIP por licencia).
"""
from collections import deque
from queue import PriorityQueue
from copy import deepcopy

import numpy as np
from cvxopt import matrix, solvers


# ===========================================================================
# ALGORITMO ORIGINAL (Bhowmick et al.) -- sin modificar
# ===========================================================================

def rdn(a):
    return round(a - (5e-16), 15)


def rup(a):
    return round(a + (5e-16), 15)


def ii_add(a, b):
    return [rdn(a[0] + b[0]), rup(a[1] + b[1])]


def ii_mul(a, b):
    mn = min(rdn(a[0] * b[0]), rdn(a[0] * b[1]), rdn(a[1] * b[0]), rdn(a[1] * b[1]))
    mx = max(rup(a[0] * b[0]), rup(a[0] * b[1]), rup(a[1] * b[0]), rup(a[1] * b[1]))
    return [mn, mx]


def ic_mul(a, b):
    return [min(rdn(b * a[0]), rdn(b * a[1])), max(rup(b * a[0]), rup(b * a[1]))]


def mm_mul(m1, m2):
    m = [[0 for j in range(len(m2[0]))] for i in range(len(m1))]
    for i in range(len(m1)):
        for j in range(len(m2[0])):
            for k in range(len(m2)):
                m[i][j] += m1[i][k] * m2[k][j]
    return m


def mv_spmul(m1, m2):
    m = [[0 for j in range(len(m2))] for i in range(len(m1))]
    for i in range(len(m2)):
        for j in range(len(m1)):
            m[j][i] = m1[j][i] * m2[i][0]
    return m


def mim_mul(m1, m2):
    m = [[[0, 0] for j in range(len(m2[0]))] for i in range(len(m1))]
    for i in range(len(m1)):
        for j in range(len(m2[0])):
            for k in range(len(m2)):
                m[i][j] = ii_add(m[i][j], ic_mul(m2[k][j], m1[i][k]))
    return m


def ivim_spmul(m1, m2):
    m = [[[0, 0] for j in range(len(m2[0]))] for i in range(len(m1))]
    for i in range(len(m1)):
        for j in range(len(m2[0])):
            m[i][j] = ii_mul(m1[i], m2[i][j])
    return m


# Estado del problema (globals, igual que el script original -- se
# reinicializan en cada llamada a run_lipbab())
var = []
act_pat = []
weights = [None]
biases = [None]
L = 0
X = []
pnorm = 2


def e_bounds(a, b):
    lb, ub = 0, 0
    for i in range(len(a)):
        if a[i] > 0:
            lb = rdn(lb + rdn(a[i] * var[i][0]))
            ub = rup(ub + rup(a[i] * var[i][1]))
        elif a[i] < 0:
            lb = rdn(lb + rdn(a[i] * var[i][1]))
            ub = rup(ub + rup(a[i] * var[i][0]))
    lb = rdn(lb + b)
    ub = rup(ub + b)
    return lb, ub


def symprop():
    global var
    ast_neurons = deque()
    output_bounds = []
    ev = [np.identity(len(X)), np.zeros(len(X))]
    for l in range(1, L + 1):
        ev = [np.dot(weights[l], ev[0]), np.dot(weights[l], ev[1]) + biases[l]]
        if l == L:
            break
        for i in range(len(ev[0])):
            lb, ub = e_bounds(ev[0][i], ev[1][i])
            if lb >= 0:
                act_pat[l][i] = [1, 1]
            elif ub <= 0:
                act_pat[l][i] = [0, 0]
                ev[0][i] = np.zeros(len(ev[0][i]))
                ev[1][i] = 0
            else:
                act_pat[l][i] = [0, 1]
                ast_neurons.append([l, i])
                ev[0] = np.hstack((ev[0], np.zeros((len(ev[0]), 1))))
                ev[0][i] = np.zeros(len(ev[0][i]))
                ev[1][i] = 0
                ev[0][i][len(ev[0][i]) - 1] = 1
                var.append([0, ub])
    for i in range(len(ev[0])):
        lb, ub = e_bounds(ev[0][i], ev[1][i])
        output_bounds.append([lb, ub])
    return ast_neurons, output_bounds


class LipNet:
    def __init__(self):
        self.Lub = 0
        self.actvp = []
        self.ast_ns = deque()
        self.H = []
        self.t = 0
        self.tev = []


def lip_bound(p):
    global pnorm
    if len(p.ast_ns) == 0:
        J = deepcopy(weights[L])
        for l in range(L - 1, 0, -1):
            J = mv_spmul(J, p.actvp[l])
            J = mm_mul(J, weights[l])
        return np.linalg.norm(J, pnorm)
    else:
        J = [[[0, 0] for j in range(len(weights[1][0]))] for i in range(len(weights[1]))]
        for i in range(len(J)):
            for j in range(len(J[0])):
                J[i][j] = [weights[1][i][j], weights[1][i][j]]
        for l in range(2, L + 1):
            J = ivim_spmul(p.actvp[l - 1], J)
            J = mim_mul(weights[l], J)
        U = [[0 for j in range(len(J[0]))] for i in range(len(J))]
        for i in range(len(J)):
            for j in range(len(J[0])):
                U[i][j] = max(abs(J[i][j][0]), abs(J[i][j][1]))
        return np.linalg.norm(U, pnorm)


def linprop(p):
    l = p.t
    ev = p.tev
    while l < L:
        if not p.ast_ns or p.ast_ns[0][0] == l:
            break
        for i in range(len(ev[0])):
            if p.actvp[l][i] == [0, 0]:
                ev[0][i] = np.zeros(len(ev[0][i]))
                ev[1][i] = 0
        l += 1
        ev = [np.dot(weights[l], ev[0]), np.dot(weights[l], ev[1]) + biases[l]]
    p.t = l
    p.tev = ev


def ffilter(p):
    A, b = deepcopy(p.H[0]), deepcopy(p.H[1])
    while len(p.ast_ns) > 0:
        l, i = p.ast_ns[0][0], p.ast_ns[0][1]
        A.append(p.tev[0][i].tolist())
        b.append(-p.tev[1][i])
        sol = solvers.lp(matrix([0.0] * len(X)), matrix(np.array(A)), matrix(b),
                         solver='glpk', options={'glpk': {'msg_lev': 'GLP_MSG_OFF'}})
        if sol['status'] != 'optimal':
            p.actvp[l][i] = [1, 1]
            p.ast_ns.popleft()
            linprop(p)
            A.pop(); b.pop()
            continue
        A.pop(); b.pop()
        A.append((-p.tev[0][i]).tolist())
        b.append(p.tev[1][i])
        sol = solvers.lp(matrix([0.0] * len(X)), matrix(np.array(A)), matrix(b),
                         solver='glpk', options={'glpk': {'msg_lev': 'GLP_MSG_OFF'}})
        if sol['status'] != 'optimal':
            p.actvp[l][i] = [0, 0]
            p.ast_ns.popleft()
            linprop(p)
            A.pop(); b.pop()
            continue
        A.pop(); b.pop()
        break


def branch(p, sgn, pq, pcnt_ref, glb_ref):
    # Nota: en el script original pq/pcnt/glb eran globals del modulo; aqui
    # se pasan como parametros para que run_lipbab() sea reentrante (se
    # pueda llamar en un loop sobre muchos x0/r sin arrastrar estado entre
    # llamadas). No cambia la logica del algoritmo, solo el mecanismo de
    # paso de estado.
    l, i = p.ast_ns[0][0], p.ast_ns[0][1]
    A, b = deepcopy(p.H[0]), deepcopy(p.H[1])
    A.append(-sgn * p.tev[0][i])
    b.append(sgn * p.tev[1][i])
    res = solvers.lp(matrix([0.0] * len(X)), matrix(np.array(A)), matrix(b),
                     solver='glpk', options={'glpk': {'msg_lev': 'GLP_MSG_OFF'}})
    if res['status'] == 'optimal':
        pb = LipNet()
        pcnt_ref[0] += 1
        pb.H = [A, b]
        pb.actvp = deepcopy(p.actvp)
        pb.ast_ns = deepcopy(p.ast_ns)
        pb.t = p.t
        pb.tev = deepcopy(p.tev)
        pb.actvp[l][i] = [max(0, sgn), max(0, sgn)]
        pb.ast_ns.popleft()
        linprop(pb)
        ffilter(pb)
        pb.Lub = lip_bound(pb)
        pq.put((-pb.Lub, pcnt_ref[0], pb))
        if len(pb.ast_ns) == 0:
            glb_ref[0] = max(glb_ref[0], pb.Lub)


# ===========================================================================
# INTEGRACION GENERICA (reescrita) -- reemplaza la seccion "custom network"
# del script original de 2023
# ===========================================================================

def run_lipbab(weight_bias_list, x0, r, pnorm_=2, af=1.0, max_iters=None,
               verbose=False):
    """
    weight_bias_list: [(W_1,b_1), ..., (W_L,b_L)] -- salida de
        VarMLP.get_weight_bias_list(). Ultima capa SIN ReLU (ver models.py).
    x0: np.array shape (input_dim,) -- punto base.
    r: radio del dominio, chi = bola L_inf {x : ||x - x0||_inf <= r}.
        (Reemplaza el `bnds=[[0.,1.0], ...]` hardcodeado del script
        original -- ahora consistente con montelip.py/exact_methods.py.)
    af: approximation factor (1.0 = exacto; el original tambien usaba 1).

    Devuelve dict {L_estimate, elapsed_s, n_explored, status}.
    """
    import time
    global var, act_pat, weights, biases, L, X, pnorm

    pnorm = pnorm_
    x0 = np.asarray(x0, dtype=np.float64)
    dim = len(x0)

    # cargar pesos/biases en la convencion 1-indexada del algoritmo original
    weights = [None] + [wb[0] for wb in weight_bias_list]
    biases = [None] + [wb[1] for wb in weight_bias_list]
    L = len(weight_bias_list)

    ls = [dim] + [w.shape[0] for w in weights[1:]]
    act_pat = [[[1, 1] for _ in range(e)] for e in ls]

    # DOMINIO: bola L_inf de radio r alrededor de x0 (antes: [0,1]^n fijo)
    bnds = [[x0[i] - r, x0[i] + r] for i in range(dim)]

    X = x0.tolist()
    var = [[b[0], b[1]] for b in bnds]

    HC = [[], []]
    for i in range(dim):
        b = bnds[i]
        a = [0] * dim; a[i] = -1.0
        HC[0].append(a); HC[1].append(-b[0])
        a = [0] * dim; a[i] = 1.0
        HC[0].append(a); HC[1].append(b[1])

    t0 = time.time()
    ast_neurons, _ = symprop()

    pcnt_ref = [0]
    glb_ref = [0.0]
    pq = PriorityQueue()

    p = LipNet()
    pcnt_ref[0] += 1
    p.H = deepcopy(HC)
    p.actvp = deepcopy(act_pat)
    p.ast_ns = deepcopy(ast_neurons)
    p.tev = [np.identity(dim), np.zeros(dim)]
    linprop(p)
    p.Lub = lip_bound(p)
    pq.put((-p.Lub, pcnt_ref[0], p))
    if len(p.ast_ns) == 0:
        glb_ref[0] = p.Lub

    i = 0
    while True:
        if max_iters is not None and i >= max_iters:
            return {'L_estimate': glb_ref[0], 'elapsed_s': time.time() - t0,
                    'status': 'MAX_ITERS', 'n_explored': i}
        if pq.empty():
            break
        tp = pq.get()[2]
        if tp.Lub <= af * glb_ref[0]:
            break
        i += 1
        if verbose and i % 500 == 0:
            print(f'  iter {i}: Lub={tp.Lub:.4f}')
        branch(tp, -1, pq, pcnt_ref, glb_ref)
        branch(tp, 1, pq, pcnt_ref, glb_ref)

    return {'L_estimate': float(glb_ref[0]), 'elapsed_s': time.time() - t0,
            'status': 'OK', 'n_explored': i}
