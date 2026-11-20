"""Política fatorada (torch, float64, 1 thread). Arquitetura FIXA:
  φ_k = tanh(W_f d_k + b_f) (12 → 32) por feature livre (descritores sem índice); g = média_k φ_k;
  h = tanh(W_h [ctx(123), g] + b_h) (155 → 64);
  parar: σ(w_s·h + b_s), b_s inicial = log(1/3) (P(parar) inicial ≈ 0,25);
  feature (ponteiro): u_k = v·tanh(A [φ_k, h] + a) (96 → 32 → 1), softmax sobre as livres;
  família: W_F [h, φ_k*] (96 → 6), máscara: famílias com algum bloco que cabe no orçamento;
  posição (bloco BIB6): W_B [h, φ_k*, onehot(fam)] (102 → 36), máscara: blocos da família escolhida que cabem;
  pesos: W_W [h, φ_k*, onehot(bloco)] (132 → 3 × 9), uma ficha por parâmetro em {−1, −0,5, −0,25, −0,1, 0, 0,1, 0,25, 0,5, 1}; só n_par posições.
Parada forçada (sem escolha, log-prob 0) quando não há feature livre ou nenhum bloco cabe."""
import numpy as np
import torch
import mechanism as SI
torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
L = SI.L; NB = SI.NB; FAMS = SI.FAMS; NT = len(SI.TOKENS)
FAM_DE = np.array([FAMS.index(b['fam']) for b in L.BIB6]); OPS = np.array([b['n_ops'] for b in L.BIB6]); NPAR = np.array([b['n_par'] for b in L.BIB6])


class Politica(torch.nn.Module):
    def __init__(s, semente):
        super().__init__(); torch.manual_seed(semente); Lin = torch.nn.Linear
        s.f = Lin(12, 32); s.h = Lin(123 + 32, 64); s.st = Lin(64, 1); s.pa = Lin(96, 32); s.pv = Lin(32, 1); s.fa = Lin(96, 6); s.bl = Lin(102, NB); s.w = Lin(132, 3 * NT)
        with torch.no_grad(): s.st.bias.fill_(float(np.log(1 / 3)))
    def n_param(s): return int(sum(p.numel() for p in s.parameters()))


def mascaras(livres, rest):
    fb = OPS <= rest; mf = np.array([bool(np.any(fb & (FAM_DE == i))) for i in range(len(FAMS))]); return fb, mf


def logp_passo(pol, desc, ctx, rest, acao):
    """Log-prob de UMA decisão (com gradiente). acao = None (parar) ou dict(j = posição da feature entre as livres, fam, bloco, tokens)."""
    D = torch.as_tensor(desc); phi = torch.tanh(pol.f(D)); h = torch.tanh(pol.h(torch.cat([torch.as_tensor(ctx), phi.mean(0)])))
    ls = torch.nn.functional.logsigmoid(pol.st(h))[0]; lns = torch.nn.functional.logsigmoid(-pol.st(h))[0]
    if acao is None: return ls, dict(p_parar=float(torch.exp(ls.detach())))
    fb, mf = mascaras(None, rest); u = pol.pv(torch.tanh(pol.pa(torch.cat([phi, h.expand(len(phi), 64)], 1))))[:, 0]
    lf = torch.log_softmax(u, 0)[acao['j']]; pk = phi[acao['j']]
    zf = pol.fa(torch.cat([h, pk])).masked_fill(torch.as_tensor(~mf), -np.inf); lfa = torch.log_softmax(zf, 0)[acao['fam']]
    ofam = torch.zeros(len(FAMS)); ofam[acao['fam']] = 1
    zb = pol.bl(torch.cat([h, pk, ofam])).masked_fill(torch.as_tensor(~(fb & (FAM_DE == acao['fam']))), -np.inf); lb = torch.log_softmax(zb, 0)[acao['bloco']]
    ob = torch.zeros(NB); ob[acao['bloco']] = 1; zw = pol.w(torch.cat([h, pk, ob])).view(3, NT)
    lw = sum(torch.log_softmax(zw[i], 0)[acao['tokens'][i]] for i in range(NPAR[acao['bloco']]))
    return lns + lf + lfa + lb + lw, {}


def _cat(p, rng): return int(min(np.searchsorted(np.cumsum(p), rng.random() * np.sum(p), side='right'), len(p) - 1))


@torch.no_grad()
def amostra_passo(pol, desc, ctx, rest, rng, temperatura=1.0):
    D = torch.as_tensor(desc); phi = torch.tanh(pol.f(D)); h = torch.tanh(pol.h(torch.cat([torch.as_tensor(ctx), phi.mean(0)])))
    if rng.random() < float(torch.sigmoid(pol.st(h))[0]): return None
    fb, mf = mascaras(None, rest); u = pol.pv(torch.tanh(pol.pa(torch.cat([phi, h.expand(len(phi), 64)], 1))))[:, 0]
    j = _cat(torch.softmax(u, 0).numpy(), rng); pk = phi[j]
    fam = _cat(torch.softmax(pol.fa(torch.cat([h, pk])).masked_fill(torch.as_tensor(~mf), -np.inf), 0).numpy(), rng)
    ofam = torch.zeros(len(FAMS)); ofam[fam] = 1; mb = fb & (FAM_DE == fam)
    b = _cat(torch.softmax(pol.bl(torch.cat([h, pk, ofam])).masked_fill(torch.as_tensor(~mb), -np.inf), 0).numpy(), rng)
    ob = torch.zeros(NB); ob[b] = 1; zw = pol.w(torch.cat([h, pk, ob])).view(3, NT)
    tok = [_cat(torch.softmax(zw[i], 0).numpy(), rng) for i in range(NPAR[b])]
    return dict(j=j, fam=fam, bloco=b, tokens=tok)


def episodio(pol, amb, rng, cont):
    """Gera um circuito completo. Só a proposta escolhida em cada estado é aplicada (direta, incremental) na construção."""
    psi = L.estado_inicial(SI.N_C); circ, passos = [], []
    while True:
        rest = SI.MAX_OPS - int(sum(OPS[c['bloco']] for c in circ)); livres = [k for k in range(SI.D_FEAT) if k not in {c['k'] for c in circ}]
        if not livres or not np.any(OPS <= rest): cont['paradas_forcadas'] += 1; break
        livres, desc, ctx = amb.descritores(psi, circ); a = amostra_passo(pol, desc, ctx, rest, rng); cont['decisoes'] += 1
        passos.append(dict(desc=desc, ctx=ctx, rest=rest, acao=a, livres=livres))
        if a is None: break
        k = livres[a['j']]; w = SI.TOKENS[a['tokens']]; t0 = SI.time.process_time(); psi = L.aplica_bloco(psi, L.BIB6[a['bloco']], amb.T['ZC'][:, k], w)
        cont['aplicacoes_bloco_C'] += 1; cont['cpu_simulacao'] += SI.time.process_time() - t0
        circ.append(dict(bloco=int(a['bloco']), k=int(k), w=w.tolist()))
    r, BV, JC = amb.recompensa(circ, psi); cont['episodios'] += 1
    return dict(circ=circ, passos=passos, recompensa=r, BCE_V=BV, J_C=JC)


def logp_episodio(pol, ep): return sum(logp_passo(pol, p['desc'], p['ctx'], p['rest'], p['acao'])[0] for p in ep['passos']) if ep['passos'] else torch.zeros(())
