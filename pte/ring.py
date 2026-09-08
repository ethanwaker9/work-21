import numpy as np
from .rns import RnsContext


class Ring:
    def __init__(self, n, log_q, num_levels, log_p, num_aux):
        assert log_q <= 31 and log_p <= 31
        two_n = 2 * n
        from .rns import find_primes, find_primes_up
        qs = find_primes(num_levels + 1, log_q, two_n)
        ps = find_primes_up(num_aux, log_p, two_n, avoid=set(qs))
        self.ctx = RnsContext(n, qs, ps)
        self.n = n
        self.L = num_levels
        self.log_q = log_q
        self.num_aux = num_aux
        self.q = self.ctx.q
        self.p = self.ctx.p

    def primes(self, lvl, with_aux=False):
        if with_aux:
            return self.q[:lvl + 1] + self.p
        return self.q[:lvl + 1]

    def ntt(self, mat, primes):
        out = np.empty_like(mat)
        for i, qi in enumerate(primes):
            out[i] = self.ctx.ntt[qi].forward(mat[i])
        return out

    def intt(self, mat, primes):
        out = np.empty_like(mat)
        for i, qi in enumerate(primes):
            out[i] = self.ctx.ntt[qi].inverse(mat[i])
        return out

    def sample_uniform(self, primes, rng):
        n = self.n
        out = np.zeros((len(primes), n), dtype=np.int64)
        for i, qi in enumerate(primes):
            out[i] = rng.integers(0, qi, size=n, dtype=np.int64)
        return out

    def sample_gauss(self, primes, sigma, rng):
        e = np.rint(rng.normal(0.0, sigma, size=self.n)).astype(np.int64)
        return self.lift(e, primes)

    def sample_ternary_sparse(self, h, rng):
        n = self.n
        v = np.zeros(n, dtype=np.int64)
        idx = rng.choice(n, size=h, replace=False)
        v[idx] = rng.integers(0, 2, size=h, dtype=np.int64) * 2 - 1
        return v

    def lift(self, coeffs, primes):
        n = self.n
        out = np.zeros((len(primes), n), dtype=np.int64)
        arr = np.asarray(coeffs)
        if arr.dtype == object:
            for i, qi in enumerate(primes):
                out[i] = np.array([int(c) % qi for c in arr], dtype=np.int64)
        else:
            a64 = arr.astype(np.int64)
            for i, qi in enumerate(primes):
                out[i] = a64 % qi
        return out

    def center(self, mat, primes):
        return self.ctx.from_rns(mat, primes)


def fast_base_conv(ring, mat, src, dst):
    n = ring.n
    k = len(src)
    Qs = 1
    for qi in src:
        Qs *= qi
    terms = np.zeros((k, n), dtype=np.int64)
    fl = np.zeros(n, dtype=np.float64)
    for i, qi in enumerate(src):
        Qi = Qs // qi
        inv = pow(Qi % qi, qi - 2, qi)
        t = (mat[i] * np.int64(inv)) % qi
        terms[i] = t
        fl += t.astype(np.float64) / float(qi)
    alpha = np.floor(fl + 1e-7).astype(np.int64)
    out = np.zeros((len(dst), n), dtype=np.int64)
    for j, pj in enumerate(dst):
        acc = np.zeros(n, dtype=np.int64)
        for i, qi in enumerate(src):
            Qi = (Qs // qi) % pj
            acc = (acc + terms[i] * np.int64(Qi)) % pj
        acc = (acc - (alpha % pj) * np.int64(Qs % pj)) % pj
        out[j] = acc
    return out
