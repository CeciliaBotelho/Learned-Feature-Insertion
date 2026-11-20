"""Construction-only features and circuit descriptors."""
import numpy as np
from scipy.stats import skew
from scipy.special import expit
from sklearn.feature_selection import f_classif

import blocks as L
from mechanism import calibrate, objective

def _auc(x, yy):
    r = np.argsort(np.argsort(x, kind='mergesort'), kind='mergesort') + 1.0
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True); s = np.zeros(len(cnt)); np.add.at(s, inv, r); r = (s / cnt)[inv]
    n1 = yy.sum(); n0 = len(yy) - n1
    if not n0 or not n1:
        return 0.5
    return float((r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _im(x, yy, nb=8):
    q = np.unique(np.quantile(x, np.linspace(0, 1, nb + 1)[1:-1])); b = np.digitize(x, q); mi = 0.0
    for v in np.unique(b):
        pb = np.mean(b == v)
        for c in (0, 1):
            pj = np.mean((b == v) & (yy == c)); pc = np.mean(yy == c)
            if pj > 0: mi += pj * np.log(pj / (pb * pc))
    return mi


_cc = lambda a, b: 0.0 if a.std() < 1e-12 or b.std() < 1e-12 else float(np.corrcoef(a, b)[0, 1])
NB = len(L.BIB6)
def entradas(psi, Zs, ys, circ, livres):
    """Build per-feature descriptors and shared context from construction data."""
    P = L.p0(psi); s = 2 * P - 1; a, b = calibrate(s, ys); f_cur = objective(np.r_[a, b], s, ys); res = ys - expit(a * s + b)
    usadas = [c['k'] for c in circ]; e = len(circ); ops = sum(L.BIB6[c['bloco']]['n_ops'] for c in circ)
    ctx = [e / 16, ops / 16, (16 - ops) / 16, f_cur, e / 8] + [sum(L.BIB6[c['bloco']]['fam'] == f for c in circ) / 8 for f in L.FAMILIAS]
    carga = np.zeros(4)
    for c in circ:
        for gg in L.BIB6[c['bloco']]['portas']: carga[list(gg[1])] += 1
    ctx += list(carga / 8); oh = np.zeros((3, NB))
    for i, c in enumerate(reversed(circ[-3:])): oh[i, c['bloco']] = 1
    ctx += list(oh.ravel())
    with np.errstate(all='ignore'): F = np.nan_to_num(f_classif(Zs[:, livres], ys)[0], nan=0.0, posinf=0.0)
    out = []
    for j, k in enumerate(livres):
        x = Zs[:, k]; red = [abs(_cc(x, Zs[:, u])) for u in usadas]
        out.append([k / 15, float(np.log1p(F[j])), x.mean() / np.pi, x.std() / np.pi, float(skew(x)) if x.std() > 1e-12 else 0.0, _auc(x, ys) - 0.5, _cc(x, ys), _im(x, ys),
                    max(red) if red else 0.0, float(np.mean(red)) if red else 0.0, _cc(x, res), abs(_cc(x, res)), _cc(x, s)] + ctx)
    return np.array(out, np.float32), f_cur
