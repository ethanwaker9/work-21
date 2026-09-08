import numpy as np
from ..rns import NTT, find_primes
from ..aring import ARing


class CGGIWord:
    def __init__(self, engine, n_bits):
        self.e = engine
        self.n = n_bits
        self.gates = 0

    def encrypt(self, m):
        return [self.e.encrypt_bit((int(m) >> i) & 1) for i in range(self.n)]

    def decrypt(self, bits):
        return sum(self.e.decrypt_bit(bits[i]) << i for i in range(self.n))

    def _g(self, kind, a, b=None):
        self.gates += 1
        return self.e.gate(kind, a, b)

    def full_add(self, a, b, c):
        s1 = self._g("xor", a, b)
        s = self._g("xor", s1, c)
        t1 = self._g("and", a, b)
        t2 = self._g("and", s1, c)
        co = self._g("or", t1, t2)
        return s, co

    def add(self, A, B):
        out = []
        c = None
        for i in range(self.n):
            if c is None:
                out.append(self._g("xor", A[i], B[i]))
                c = self._g("and", A[i], B[i])
            else:
                s, c = self.full_add(A[i], B[i], c)
                out.append(s)
        return out

    def mul(self, A, B):
        acc = None
        for j in range(self.n):
            part = [self._g("and", A[i], B[j]) for i in range(self.n - j)]
            row = [None] * j + part
            row = row[:self.n]
            if acc is None:
                acc = row
                for i in range(j):
                    acc[i] = self.e.encrypt_bit(0)
            else:
                for i in range(j):
                    row[i] = self.e.encrypt_bit(0)
                acc = self.add(acc, row)
        return acc

    def bitwise(self, kind, A, B):
        return [self._g(kind, A[i], B[i]) for i in range(self.n)]


def add_gate_count(n):
    return 2 + 5 * (n - 1)


def mul_gate_count(n):
    g = 0
    for j in range(n):
        g += n - j
        if j > 0:
            g += add_gate_count(n)
    return g


def bitwise_gate_count(n):
    return n


class REFHE:
    def __init__(self, n_bits, logq=30, levels=6, sigma=3.2, seed=0, engine=None):
        self.n = n_bits
        self.R = ARing(1 << n_bits, n_bits)
        assert self.R.b == 2
        self.levels = levels
        self.rng = np.random.default_rng(seed)
        self.qs = [(1 << (logq + n_bits * i)) - 1 for i in range(levels + 1)]
        self.qs = [q + (2 - q % 2) for q in self.qs]
        self.sigma = sigma
        self.engine = engine
        self.Fred = np.array([-c for c in self.R.F[:self.n]], dtype=object)

    def _modmul(self, a, b, q):
        r = np.zeros(2 * self.n - 1, dtype=object)
        for i, x in enumerate(a):
            if x:
                for j, y in enumerate(b):
                    r[i + j] = (r[i + j] + int(x) * int(y))
        for i in range(len(r) - 1, self.n - 1, -1):
            v = r[i]
            if v:
                r[i] = 0
                for j in range(self.n):
                    r[i - self.n + j] = r[i - self.n + j] + v * int(self.Fred[j])
        return np.array([int(v) % q for v in r[:self.n]], dtype=object)

    def _center(self, a, q):
        return np.array([int(v) - q if int(v) > q // 2 else int(v) for v in a], dtype=object)

    def keygen(self):
        self.s = np.array([int(x) for x in self.rng.integers(-1, 2, size=self.n)], dtype=object)
        self.s2 = None
        return self.s

    def encrypt(self, m, level=None):
        lvl = self.levels if level is None else level
        q = self.qs[lvl]
        mu = np.array(self.R.digits(int(m) % (1 << self.n)), dtype=object)
        a = np.array([int(self.rng.integers(0, 1 << 62)) % q for _ in range(self.n)], dtype=object)
        e = np.array([int(round(x)) for x in self.rng.normal(0, self.sigma, size=self.n)], dtype=object)
        te = self._modmul(np.array([-2, 1] + [0] * (self.n - 2), dtype=object), e, q)
        b = (-self._modmul(a, self.s, q) + mu + te) % q
        return {"c": [np.array([int(v) % q for v in b], dtype=object), a], "lvl": lvl}

    def decrypt(self, ct):
        q = self.qs[ct["lvl"]]
        v = (ct["c"][0] + self._modmul(ct["c"][1], self.s, q)) % q
        v = self._center(v, q)
        val = 0
        for i in range(self.n - 1, -1, -1):
            val = val * 2 + int(v[i])
        return val % (1 << self.n)

    def add(self, x, y):
        q = self.qs[x["lvl"]]
        return {"c": [(x["c"][i] + y["c"][i]) % q for i in range(2)], "lvl": x["lvl"]}

    def mul(self, x, y):
        q = self.qs[x["lvl"]]
        d0 = self._modmul(x["c"][0], y["c"][0], q)
        d1 = (self._modmul(x["c"][0], y["c"][1], q) + self._modmul(x["c"][1], y["c"][0], q)) % q
        d2 = self._modmul(x["c"][1], y["c"][1], q)
        s2 = self._modmul(self.s, self.s, q)
        c0 = (d0 + self._modmul(d2, s2, q)) % q
        c1 = d1
        return {"c": [c0, c1], "lvl": x["lvl"]}

    def mod_switch(self, ct):
        lvl = ct["lvl"]
        if lvl == 0:
            return ct
        q = self.qs[lvl]
        qn = self.qs[lvl - 1]
        out = []
        for c in ct["c"]:
            cc = self._center(c, q)
            scaled = np.array([int(round(int(v) * qn / q)) for v in cc], dtype=object)
            out.append(np.array([int(v) % qn for v in scaled], dtype=object))
        return {"c": out, "lvl": lvl - 1}

    def noise(self, ct):
        q = self.qs[ct["lvl"]]
        v = (ct["c"][0] + self._modmul(ct["c"][1], self.s, q)) % q
        v = self._center(v, q)
        return max(abs(int(x)) for x in v)

    def bootstrap_ops(self):
        return {"pbs": self.n, "repack": self.n}
