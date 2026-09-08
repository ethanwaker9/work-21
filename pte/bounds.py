import numpy as np
from itertools import product


def clearing_degree(g, overflow=2, maxdeg=None):
    md = maxdeg if maxdeg is not None else (overflow + 1) * g + 6
    pts = []
    vals = []
    for I in range(overflow):
        for j in range(g):
            pts.append(I + j / g)
            vals.append(j / g)
    pts = np.array(pts, dtype=np.float64)
    vals = np.array(vals, dtype=np.float64)
    lo, hi = pts.min(), pts.max()
    u = (2.0 * pts - (hi + lo)) / (hi - lo)
    for D in range(1, md + 1):
        A = np.polynomial.chebyshev.chebvander(u, D)
        sol, res, rank, sv = np.linalg.lstsq(A, vals, rcond=None)
        err = np.max(np.abs(A @ sol - vals))
        if err < 1e-8:
            return D
    return None


def carry_multilinear_degree(s):
    idx = list(range(2 * s))
    coeff = {}
    for assign in product([0, 1], repeat=2 * s):
        u = assign[:s]
        v = assign[s:]
        val = sum((u[i] + v[i]) << i for i in range(s))
        f = 1 if val >= (1 << s) else 0
        if f:
            coeff[assign] = f
    top = 0
    keys = [tuple(idx)]
    c = 0.0
    for assign in product([0, 1], repeat=2 * s):
        u = assign[:s]
        v = assign[s:]
        val = sum((u[i] + v[i]) << i for i in range(s))
        f = 1.0 if val >= (1 << s) else 0.0
        sgn = (-1.0) ** (2 * s - sum(assign))
        c += sgn * f
    return 2 * s if abs(c) > 1e-9 else None


def rounds_lower_bound(n, log_p, level_budget):
    return int(np.ceil(n / (log_p + level_budget)))


def rounds_upper_bound(n, w):
    return int(np.ceil(n / w))


def digit_arith_lower_bound(n, log_g, log_p, level_budget):
    s = max(2, int(np.ceil(n / max(1, log_g))))
    return int(np.ceil(np.log2(2 * s) / (log_p + level_budget)))


def tradeoff_curve(n, log_p, level_budget, log_g_values):
    out = []
    for lg in log_g_values:
        a2b = int(np.ceil(lg / (log_p + level_budget)))
        arith = digit_arith_lower_bound(n, lg, log_p, level_budget)
        out.append((lg, a2b, arith))
    return out
