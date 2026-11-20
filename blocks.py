"""Direct statevector simulation for BIB6 on a four-qubit chain."""
import numpy as np

N_Q = 4
X2, Y2, I2 = np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]]), np.eye(2, dtype=complex)
CNOT_M = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], complex)
FAMILIAS = ('F1', 'F2', 'F3', 'F4', 'F5', 'F6')


def mat(tipo, ang):
    ang = np.atleast_1d(np.asarray(ang, float)); c = np.cos(ang / 2)[:, None, None]; s = np.sin(ang / 2)[:, None, None]
    if tipo == 'RX': return c * I2 - 1j * s * X2
    if tipo == 'RY': return c * I2 - 1j * s * Y2
    if tipo == 'RZ': return np.stack([np.stack([np.exp(-0.5j * ang), 0 * ang], -1), np.stack([0 * ang, np.exp(0.5j * ang)], -1)], -2)
    if tipo == 'RXX': return c * np.eye(4) - 1j * s * np.kron(X2, X2)
    if tipo == 'RYY': return c * np.eye(4) - 1j * s * np.kron(Y2, Y2)
    raise ValueError(tipo)


def aplica(psi, tipo, qs, ang=None):
    N = psi.shape[0]; t = psi.reshape((N,) + (2,) * N_Q)
    M = CNOT_M[None] if tipo == 'CNOT' else mat(tipo, np.broadcast_to(np.asarray(ang, float), (N,)))
    if M.shape[0] == 1 and N > 1: M = np.broadcast_to(M, (N,) + M.shape[1:])
    if len(qs) == 1:
        t = np.moveaxis(t, 1 + qs[0], 1); out = np.einsum('nab,nb...->na...', M, t); return np.moveaxis(out, 1, 1 + qs[0]).reshape(N, -1)
    t = np.moveaxis(t, (1 + qs[0], 1 + qs[1]), (1, 2)); sh = t.shape; t = t.reshape(N, 4, -1)
    out = np.einsum('nab,nbr->nar', M, t).reshape(sh); return np.moveaxis(out, (1, 2), (1 + qs[0], 1 + qs[1])).reshape(N, -1)


def estado_inicial(N): psi = np.zeros((N, 2 ** N_Q), complex); psi[:, 0] = 1.0; return psi
def p0(psi): return np.abs(psi[:, 0]) ** 2


ARESTAS = [(q, q + 1) for q in range(N_Q - 1)]
VIZ = {a: [b for b in (a - 1, a + 1) if 0 <= b < N_Q] for a in range(N_Q)}


def _fam():
    F = []
    F += [('F1', (('RY', (a,), 0), ('RX', (a,), 1)), f'a{a}') for a in range(N_Q)]
    F += [('F3', (('RY', (a,), 0), ('RXX', tuple(sorted((a, b))), 1)), f'a{a}-XX{min(a, b)}{max(a, b)}') for a in range(N_Q) for b in VIZ[a]]
    F += [('F2', (('RX', (a,), 0), ('RY', (a,), 1)), f'a{a}') for a in range(N_Q)]
    F += [('F4', (('RY', (a,), 0), ('RYY', tuple(sorted((a, b))), 1)), f'a{a}-YY{min(a, b)}{max(a, b)}') for a in range(N_Q) for b in VIZ[a]]
    F += [('F5', (('RY', (a,), 0), ('RZ', (a,), 1), ('RX', (a,), 2)), f'a{a}') for a in range(N_Q)]
    F += [('F6', (('RY', (a,), 0), ('CNOT', (a, b), None), ('RX', (r,), 1)), f'c{a}t{b}-RX{r}') for a in range(N_Q) for b in VIZ[a] for r in (a, b)]
    return [dict(id=i, fam=f, portas=p, nome=f'{f}:{n}', n_par=sum(g[2] is not None for g in p), n_ops=len(p), ancora=p[0][1][0]) for i, (f, p, n) in enumerate(F)]


BIB6 = _fam(); NB = len(BIB6)                       # four-qubit chain


def aplica_bloco(psi, b, x, w):
    for tipo, qs, iw in b['portas']: psi = aplica(psi, tipo, qs, None if iw is None else w[iw] * x)
    return psi
