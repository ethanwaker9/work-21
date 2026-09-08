import numpy as np
from sympy import Poly, GF, symbols, factorint
from .aring import poly_eval_int, two_slot_poly, ARing

X = symbols('X')


def linear_simple_factors(coeffs, p):
    f = Poly(list(reversed([int(c) for c in coeffs])), X, domain=GF(p))
    fac = f.factor_list()
    out = []
    for g, e in fac[1]:
        if g.degree() == 1 and e == 1:
            root = (-int(g.all_coeffs()[1]) * pow(int(g.all_coeffs()[0]), -1, p)) % p
            out.append(root)
    return out


def native_slot_capacity(coeffs, M):
    caps = []
    for p in factorint(int(M)):
        caps.append(len(linear_simple_factors(coeffs, int(p))))
    return min(caps) if caps else 0


def capacity_upper_bound(M, d):
    return min(int(d), min(int(p) for p in factorint(int(M))))


def search_two_slot(M, d, tries=1500, seed=11):
    b, base, basis = two_slot_poly(M, d)
    rng = np.random.default_rng(seed)
    best = None
    cur = list(base[:d])
    cands = [list(cur)]
    for _ in range(tries):
        cand = list(cur)
        for row in basis:
            k = int(rng.integers(-1, 2))
            if k:
                cand = [c + k * r for c, r in zip(cand, row)]
        cands.append(cand)
    for cand in cands:
        cc = list(cand) + [1]
        v1 = poly_eval_int(cc, b)
        v2 = poly_eval_int(cc, b + 1)
        if abs(v1) != M or abs(v2) != M:
            continue
        cap = native_slot_capacity(cc, M)
        rr = int(sum(1 for z in np.roots([float(c) for c in reversed(cc)])
                     if abs(z.imag) < 1e-9 * max(1.0, abs(z))))
        nrm = max(abs(x) for x in cc[:d])
        score = (-cap, rr, nrm)
        if best is None or score < best[0]:
            best = (score, cc)
    return b, (best[1] if best else list(base[:d]) + [1])


def subspace_distance(b, d, M):
    from fractions import Fraction as Fr
    vb = [Fr(b) ** i for i in range(d)]
    vc = [Fr(b + 1) ** i for i in range(d)]
    tb = Fr(-M - b ** d)
    tc = Fr(-M - (b + 1) ** d)
    g11 = sum(x * x for x in vb)
    g12 = sum(x * y for x, y in zip(vb, vc))
    g22 = sum(y * y for y in vc)
    det = g11 * g22 - g12 * g12
    a1 = (g22 * tb - g12 * tc) / det
    a2 = (g11 * tc - g12 * tb) / det
    proj = [a1 * x + a2 * y for x, y in zip(vb, vc)]
    return float(sum(float(x) ** 2 for x in proj)) ** 0.5 / (d ** 0.5)


def two_slot_instance(M, d):
    b, coeffs = search_two_slot(M, d)
    v1 = poly_eval_int(coeffs, b)
    v2 = poly_eval_int(coeffs, b + 1)
    ok = (abs(v1) == M and abs(v2) == M)
    cap = native_slot_capacity(coeffs, M) if ok else 0
    return {
        "b": b,
        "coeffs": coeffs,
        "norm": max(abs(c) for c in coeffs[:d]),
        "F(b)": v1,
        "F(b+1)": v2,
        "valid": ok,
        "capacity": cap,
        "real_roots": int(sum(1 for z in np.roots([float(c) for c in reversed(coeffs)])
                              if abs(z.imag) < 1e-9 * max(1.0, abs(z)))),
    }


def crt_pack(coeffs, b, M, m1, m2, d):
    A = np.zeros((d, d), dtype=np.float64)
    for j in range(d):
        A[0, j] = (b ** j)
        A[1, j] = ((b + 1) ** j)
    tgt = np.zeros(d)
    tgt[0] = m1
    tgt[1] = m2
    sol, *_ = np.linalg.lstsq(A[:2], tgt[:2], rcond=None)
    return sol


def capacity_table(word_bits, degrees):
    rows = []
    for n in word_bits:
        M = 1 << n
        for d in degrees:
            if d > n or n % d:
                continue
            R = ARing(M, d)
            cap = native_slot_capacity(R.F, M)
            rows.append({
                "n": n, "d": d, "b": R.b, "capacity": cap,
                "bound": capacity_upper_bound(M, d), "norm": max(abs(c) for c in R.F[:d]),
            })
    return rows
