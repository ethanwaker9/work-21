import numpy as np
from .ckks import Ciphertext, Plaintext
from .lintrans import LinTransform, slot_to_coeff_matrix, coeff_to_slot_matrices


def truncate(ctx, ct):
    r = ctx.ring
    lvl = ct.level
    Ql = r.ctx.modulus(r.primes(lvl))
    u = int(round(Ql / ct.scale))
    out = ctx.mul_int(ct, u)
    out = Ciphertext(out.c, lvl, ct.scale * u, ctx)
    while out.level > 0:
        out = ctx.rescale(out)
    return Ciphertext(out.c, 0, float(r.q[0]), ctx)


def mod_raise(ctx, ct, level=None):
    r = ctx.ring
    lvl = ctx.L if level is None else level
    q0 = r.q[0]
    coeffs = r.ctx.ntt[q0].inverse(ct.c[0][0]), r.ctx.ntt[q0].inverse(ct.c[1][0])
    pr = r.primes(lvl)
    out = []
    for cc in coeffs:
        cen = np.where(cc > q0 // 2, cc - q0, cc)
        out.append(r.ntt(r.lift(cen, pr), pr))
    return Ciphertext(out, lvl, ct.scale, ctx)


def cheb_coeffs(f, lo, hi, deg):
    k = np.arange(deg + 1)
    nodes = np.cos(np.pi * (2 * np.arange(deg + 1) + 1) / (2 * (deg + 1)))
    x = 0.5 * (hi - lo) * nodes + 0.5 * (hi + lo)
    y = np.array([f(t) for t in x], dtype=np.complex128)
    c = np.zeros(deg + 1, dtype=np.complex128)
    for j in range(deg + 1):
        c[j] = (2.0 / (deg + 1)) * np.sum(y * np.cos(np.pi * j * (2 * np.arange(deg + 1) + 1) / (2 * (deg + 1))))
    c[0] /= 2.0
    return c


class ChebEvaluator:
    def __init__(self, ctx):
        self.ctx = ctx

    def eval(self, x, coeffs, lo, hi):
        ctx = self.ctx
        a = 2.0 / (hi - lo)
        bshift = -(hi + lo) / (hi - lo)
        y = ctx.mul_const(x, a)
        y = ctx.add_const(y, bshift)
        d = len(coeffs) - 1
        e = 0
        while (1 << (e + 1)) <= d:
            e += 1
        T = {1: y}
        for j in range(1, e + 1):
            p = T[1 << (j - 1)]
            sq = ctx.mul(p, p)
            T[1 << j] = ctx.add_const(ctx.mul_int(sq, 2), -1.0)
        return self._rec(T, list(coeffs), e)

    def _rec(self, T, c, e):
        ctx = self.ctx
        while len(c) > 1 and abs(c[-1]) < 1e-14:
            c = c[:-1]
        d = len(c) - 1
        if d == 0:
            return ("const", complex(c[0]))
        while e > 0 and (1 << e) > d:
            e -= 1
        if e == 0:
            base = ctx.mul_const(T[1], c[1])
            return ctx.add_const(base, complex(c[0]))
        k = 1 << e
        hi = [2.0 * c[i] for i in range(k, d + 1)]
        beta = c[k]
        hi[0] -= beta
        low = [c[i] for i in range(k)]
        for i in range(k + 1, d + 1):
            idx = abs(2 * k - i)
            low[idx] -= c[i]
        A = self._rec(T, hi, e - 1)
        B = self._rec(T, low, e - 1)
        Tk = T[k]
        if isinstance(A, tuple):
            prod = ctx.mul_const(Tk, A[1])
        else:
            prod = ctx.mul(A, Tk)
        if isinstance(B, tuple):
            return ctx.add_const(prod, B[1])
        return ctx.add(prod, B)

    def _align(self, Tk, other):
        return self.ctx.drop_to(Tk, other.level)


def make_lut_coeffs(p, table, order=1):
    a = np.zeros(p, dtype=np.complex128)
    for m in range(p):
        a[m] = sum(table[j] * np.exp(-2j * np.pi * j * m / p) for j in range(p)) / p
    q = order + 1
    c = np.zeros(q * p, dtype=np.complex128)
    for m in range(p):
        ks = np.array([m + t * p for t in range(q)], dtype=np.float64)
        A = np.vstack([ks ** e for e in range(q)])
        rhs = np.zeros(q, dtype=np.complex128)
        rhs[0] = a[m]
        sol = np.linalg.solve(A, rhs)
        for t in range(q):
            c[m + t * p] = sol[t]
    return c


class Bootstrapper:
    def __init__(self, ctx, K=8, s=5, deg=24, p=16, order=1):
        self.ctx = ctx
        self.K = K
        self.s = s
        self.deg = deg
        self.p = p
        self.order = order
        E0 = slot_to_coeff_matrix(ctx)
        A0, _ = coeff_to_slot_matrices(ctx)
        self.s2c = LinTransform(ctx, E0, name="StC")
        self.c2s = LinTransform(ctx, A0, name="CtS")
        rho = (K + 1.0) / (1 << s)
        self.rho = rho
        self.exp_coeffs = cheb_coeffs(lambda t: np.exp(2j * np.pi * t), -rho, rho, deg)
        self.cheb = ChebEvaluator(ctx)
        self.levels_used = None

    def exp_of(self, ct):
        ctx = self.ctx
        y = ctx.mul_const(ct, 1.0 / (1 << self.s))
        u = self.cheb.eval(y, self.exp_coeffs, -self.rho, self.rho)
        for _ in range(self.s):
            u = ctx.mul(u, u)
        return u

    def _pow_tree(self, u, exps):
        cache = {1: u}

        def get(e):
            if e in cache:
                return cache[e]
            a = e // 2
            v = self.ctx.mul(get(a), get(e - a))
            cache[e] = v
            return v

        for e in sorted(exps):
            if e >= 1:
                get(e)
        return cache

    def lut_on_exp(self, u, tables, p):
        ctx = self.ctx
        outs = []
        deg = (self.order + 1) * p
        k = max(1, int(np.ceil(np.sqrt(deg))))
        gsteps = [g for g in range(k, deg, k)]
        cache = self._pow_tree(u, list(range(1, k)) + gsteps)
        for table in tables:
            c = make_lut_coeffs(p, table, self.order)
            acc = None
            for g in [0] + gsteps:
                chunk = [c[i] for i in range(g, min(g + k, deg))]
                part = None
                for i in range(1, len(chunk)):
                    if abs(chunk[i]) < 1e-13:
                        continue
                    t = ctx.mul_const(cache[i], chunk[i])
                    part = t if part is None else ctx.add(part, t)
                if part is not None:
                    if abs(chunk[0]) > 1e-13:
                        part = ctx.add_const(part, chunk[0])
                elif abs(chunk[0]) > 1e-13:
                    part = ctx.add_const(ctx.mul_const(cache[1], 0.0), chunk[0])
                else:
                    continue
                if g:
                    part = ctx.mul(part, cache[g])
                acc = part if acc is None else ctx.add(acc, part)
            outs.append(acc)
        return outs

    def clear_and_lut(self, ct, tables, p):
        ctx = self.ctx
        ctx.boot_count += 1
        a = self.s2c.apply(ct)
        a = truncate(ctx, a)
        a = mod_raise(ctx, a)
        v = self.c2s.apply(a)
        h = ctx.add(v, ctx.conjugate(v))
        u = self.exp_of(h)
        return self.lut_on_exp(u, tables, p)

    def small_part(self, h):
        ctx = self.ctx
        u = self.exp_of(h)
        w = ctx.sub(u, ctx.conjugate(u))
        return ctx.mul_const(w, -0.25j / np.pi)

    def clear_small(self, ct):
        ctx = self.ctx
        ctx.boot_count += 1
        a = self.s2c.apply(ct)
        a = truncate(ctx, a)
        a = mod_raise(ctx, a)
        v = self.c2s.apply(a)
        h = ctx.add(v, ctx.conjugate(v))
        u = self.exp_of(h)
        w = ctx.sub(u, ctx.conjugate(u))
        return ctx.mul_const(w, -0.5j / (2 * np.pi))
