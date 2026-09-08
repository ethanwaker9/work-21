import numpy as np
from ..rns import NTT, find_primes


class CGGIParams:
    def __init__(self, n=512, N=1024, logq=30, logBg=10, l=3, ks_base=5, ks_levels=6, sigma=2.0):
        self.n = n
        self.N = N
        self.q = find_primes(1, logq, 2 * N)[0]
        self.Bg = 1 << logBg
        self.logBg = logBg
        self.l = l
        self.ks_base = ks_base
        self.ks_levels = ks_levels
        self.sigma = sigma


class CGGI:
    def __init__(self, params=None, seed=0):
        self.P = params or CGGIParams()
        self.rng = np.random.default_rng(seed)
        self.ntt = NTT(self.P.N, self.P.q)
        self.boot_count = 0

    def keygen(self):
        P = self.P
        self.s = self.rng.integers(0, 2, size=P.n).astype(np.int64)
        self.z = self.rng.integers(0, 2, size=P.N).astype(np.int64)
        self.z_ntt = self.ntt.forward(self.z % P.q)
        self.bk = [self._rgsw_ntt(int(b)) for b in self.s]
        self.ksk = self._ksk()
        return self.s

    def _rlwe(self, m):
        P = self.P
        a = self.rng.integers(0, P.q, size=P.N).astype(np.int64)
        e = np.rint(self.rng.normal(0, P.sigma, size=P.N)).astype(np.int64)
        an = self.ntt.forward(a)
        b = (self.ntt.inverse((an * self.z_ntt) % P.q) + m + e) % P.q
        return np.stack([b, a])

    def _rgsw_ntt(self, bit):
        r = self._rgsw(bit)
        return np.stack([[self.ntt.forward(r[i][t]) for t in range(2)] for i in range(r.shape[0])])

    def _rgsw(self, bit):
        P = self.P
        rows = []
        for i in range(P.l):
            g = int(round(P.q / (P.Bg ** (i + 1))))
            m = np.zeros(P.N, dtype=np.int64)
            m[0] = (bit * g) % P.q
            c0 = self._rlwe(m)
            rows.append(c0)
            c1 = self._rlwe(np.zeros(P.N, dtype=np.int64))
            c1[1] = (c1[1] + m) % P.q
            rows.append(c1)
        return np.stack(rows)

    def _decomp(self, poly):
        P = self.P
        v = poly % P.q
        v = np.where(v > P.q // 2, v - P.q, v)
        out = np.zeros((P.l, P.N), dtype=np.int64)
        base = P.Bg
        Bl = base ** P.l
        w = (v * Bl + P.q // 2) // P.q
        for i in range(P.l - 1, -1, -1):
            d = w % base
            d = np.where(d > base // 2, d - base, d)
            out[i] = d
            w = (w - d) // base
        return out

    def _extmul(self, acc, rgsw):
        P = self.P
        d0 = self._decomp(acc[0])
        d1 = self._decomp(acc[1])
        res = np.zeros((2, P.N), dtype=np.int64)
        accn = np.zeros((2, P.N), dtype=np.int64)
        for i in range(P.l):
            for j, dd in ((0, d0[i]), (1, d1[i])):
                row = rgsw[2 * i + j]
                dn = self.ntt.forward(dd % P.q)
                for t in range(2):
                    accn[t] = (accn[t] + dn * row[t]) % P.q
        for t in range(2):
            res[t] = self.ntt.inverse(accn[t])
        return res

    def _rot_poly(self, c, k):
        P = self.P
        N = P.N
        k = k % (2 * N)
        out = np.zeros_like(c)
        idx = (np.arange(N) + k) % (2 * N)
        sgn = np.where(idx >= N, -1, 1)
        pos = idx % N
        out[pos] = c * sgn
        return out % P.q

    def bootstrap(self, ct, mu):
        P = self.P
        self.boot_count += 1
        N, q = P.N, P.q
        scale = 2 * N
        bb = int(round(int(ct[0]) * scale / q)) % scale
        aa = [int(round(int(x) * scale / q)) % scale for x in ct[1:]]
        tv = np.full(N, mu % q, dtype=np.int64)
        acc = np.stack([self._rot_poly(tv, -bb), np.zeros(N, dtype=np.int64)])
        for i in range(P.n):
            if aa[i] == 0:
                continue
            prod = self._extmul(acc, self.bk[i])
            rot = np.stack([self._rot_poly(prod[0], aa[i]), self._rot_poly(prod[1], aa[i])])
            acc = (rot - prod + acc) % q
        return self._extract(acc)

    def _extract(self, acc):
        P = self.P
        N, q = P.N, P.q
        a = np.zeros(N, dtype=np.int64)
        a[0] = acc[1][0]
        for i in range(1, N):
            a[i] = (-acc[1][N - i]) % q
        big = np.concatenate([[acc[0][0]], a])
        return self._keyswitch(big)

    def _ksk(self):
        P = self.P
        base = 1 << P.ks_base
        out = np.zeros((P.N, P.ks_levels, base, P.n + 1), dtype=np.int64)
        for i in range(P.N):
            for j in range(P.ks_levels):
                for v in range(base):
                    a = self.rng.integers(0, P.q, size=P.n).astype(np.int64)
                    e = int(np.rint(self.rng.normal(0, P.sigma)))
                    val = int(self.z[i]) * v * (P.q // (base ** (j + 1)))
                    b = (int(np.dot(a, self.s) % P.q) + val + e) % P.q
                    out[i, j, v, 0] = b
                    out[i, j, v, 1:] = a
        return out

    def _keyswitch(self, big):
        P = self.P
        base = 1 << P.ks_base
        res = np.zeros(P.n + 1, dtype=np.int64)
        res[0] = big[0] % P.q
        for i in range(P.N):
            a = int(big[1 + i]) % P.q
            w = int(round(a * (base ** P.ks_levels) / P.q)) % (base ** P.ks_levels)
            for j in range(P.ks_levels):
                digit = (w >> (P.ks_base * (P.ks_levels - 1 - j))) & (base - 1)
                if digit:
                    res = (res - self.ksk[i, j, digit]) % P.q
        return res

    def encrypt_bit(self, bit):
        P = self.P
        a = self.rng.integers(0, P.q, size=P.n).astype(np.int64)
        e = int(np.rint(self.rng.normal(0, P.sigma)))
        mu = (P.q // 8) if bit else (-(P.q // 8))
        b = (int(np.dot(a, self.s) % P.q) + mu + e) % P.q
        out = np.zeros(P.n + 1, dtype=np.int64)
        out[0] = b
        out[1:] = a
        return out

    def phase(self, ct):
        P = self.P
        v = (int(ct[0]) - int(np.dot(ct[1:], self.s))) % P.q
        return v - P.q if v > P.q // 2 else v

    def decrypt_bit(self, ct):
        return 1 if self.phase(ct) > 0 else 0

    def _const(self, val):
        P = self.P
        out = np.zeros(P.n + 1, dtype=np.int64)
        out[0] = val % P.q
        return out

    def gate(self, kind, a, b=None):
        P = self.P
        q = P.q
        e8 = q // 8
        if kind == "not":
            return (-a) % q
        if kind == "and":
            c = (a + b + self._const(-e8)) % q
        elif kind == "or":
            c = (a + b + self._const(e8)) % q
        elif kind == "nand":
            c = (self._const(e8) - a - b) % q
        elif kind == "xor":
            c = (2 * (a + b) + self._const(2 * e8)) % q
        elif kind == "xnor":
            c = (-2 * (a + b) + self._const(-2 * e8)) % q
        else:
            raise ValueError(kind)
        return self.bootstrap(c, e8)

    def mux(self, s, a, b):
        return self.gate("or", self.gate("and", s, a), self.gate("and", self.gate("not", s), b))


def ripple_add_gates(n):
    return 5 * n - 3


def schoolbook_mul_gates(n):
    return n * n + n * (5 * n - 3)
