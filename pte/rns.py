import numpy as np


def is_prime(n):
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def find_primes(count, bits, two_n, avoid=()):
    out = []
    cand = (1 << bits)
    cand -= cand % two_n
    cand += 1
    while len(out) < count:
        cand -= two_n
        if cand <= two_n:
            raise ValueError("prime pool exhausted")
        if cand in avoid or cand in out:
            continue
        if is_prime(cand):
            out.append(cand)
    return out


def find_primes_up(count, bits, two_n, avoid=()):
    out = []
    cand = (1 << bits)
    cand -= cand % two_n
    cand += 1
    while len(out) < count:
        cand += two_n
        if cand in avoid or cand in out:
            continue
        if is_prime(cand):
            out.append(cand)
    return out


def primitive_root(q):
    fac = []
    m = q - 1
    x = 2
    while x * x <= m:
        if m % x == 0:
            fac.append(x)
            while m % x == 0:
                m //= x
        x += 1
    if m > 1:
        fac.append(m)
    g = 2
    while True:
        if all(pow(g, (q - 1) // f, q) != 1 for f in fac):
            return g
        g += 1


def _bitrev_table(n):
    bits = n.bit_length() - 1
    idx = np.arange(n, dtype=np.int64)
    out = np.zeros(n, dtype=np.int64)
    for b in range(bits):
        out |= ((idx >> b) & 1) << (bits - 1 - b)
    return out


class NTT:
    def __init__(self, n, q):
        self.n = n
        self.q = int(q)
        g = primitive_root(q)
        psi = pow(g, (q - 1) // (2 * n), q)
        ipsi = pow(psi, q - 2, q)
        rev = _bitrev_table(n)
        pw = np.ones(n, dtype=object)
        ipw = np.ones(n, dtype=object)
        for i in range(1, n):
            pw[i] = int(pw[i - 1]) * psi % q
            ipw[i] = int(ipw[i - 1]) * ipsi % q
        self.psi_rev = np.array([int(pw[r]) for r in rev], dtype=np.int64)
        self.ipsi_rev = np.array([int(ipw[r]) for r in rev], dtype=np.int64)
        self.n_inv = pow(n, q - 2, q)
        self.stages = []
        t = n
        m = 1
        while m < n:
            t //= 2
            self.stages.append((m, t, self.psi_rev[m:2 * m].copy()))
            m *= 2
        self.istages = []
        t = 1
        m = n
        while m > 1:
            h = m // 2
            self.istages.append((h, t, self.ipsi_rev[h:2 * h].copy()))
            t *= 2
            m //= 2

    def forward(self, a):
        q = self.q
        x = a.astype(np.int64, copy=True)
        for m, t, tw in self.stages:
            x = x.reshape(m, 2 * t)
            lo = x[:, :t]
            hi = x[:, t:]
            v = (hi * tw[:, None]) % q
            x = np.concatenate((lo + v, lo - v), axis=1) % q
        return x.reshape(self.n)

    def inverse(self, a):
        q = self.q
        x = a.astype(np.int64, copy=True)
        for h, t, tw in self.istages:
            x = x.reshape(h, 2 * t)
            lo = x[:, :t]
            hi = x[:, t:]
            s = (lo + hi) % q
            d = ((lo - hi) * tw[:, None]) % q
            x = np.concatenate((s, d), axis=1)
        return (x.reshape(self.n) * self.n_inv) % q


class RnsContext:
    def __init__(self, n, primes, aux_primes):
        self.n = n
        self.q = [int(x) for x in primes]
        self.p = [int(x) for x in aux_primes]
        self.all = self.q + self.p
        self.ntt = {qi: NTT(n, qi) for qi in self.all}
        self.qv = np.array(self.q, dtype=np.int64)
        self.Q = 1
        for qi in self.q:
            self.Q *= qi
        self.P = 1
        for pi in self.p:
            self.P *= pi

    def modulus(self, primes):
        m = 1
        for qi in primes:
            m *= qi
        return m

    def to_rns(self, coeffs, primes):
        arr = np.array([int(c) for c in coeffs], dtype=object)
        out = np.zeros((len(primes), self.n), dtype=np.int64)
        for i, qi in enumerate(primes):
            out[i] = np.array([int(c % qi) for c in arr], dtype=np.int64)
        return out

    def from_rns(self, mat, primes):
        Q = self.modulus(primes)
        acc = np.zeros(self.n, dtype=object)
        for i, qi in enumerate(primes):
            Qi = Q // qi
            inv = pow(Qi % qi, qi - 2, qi)
            f = Qi * inv
            row = mat[i].astype(object)
            acc = acc + row * f
        half = Q // 2
        acc = acc % Q
        return np.array([int(v - Q) if v > half else int(v) for v in acc], dtype=object)


def vadd(a, b, primes):
    out = np.empty_like(a)
    for i, qi in enumerate(primes):
        out[i] = (a[i] + b[i]) % qi
    return out


def vsub(a, b, primes):
    out = np.empty_like(a)
    for i, qi in enumerate(primes):
        out[i] = (a[i] - b[i]) % qi
    return out


def vneg(a, primes):
    out = np.empty_like(a)
    for i, qi in enumerate(primes):
        out[i] = (-a[i]) % qi
    return out


def vmul(a, b, primes):
    out = np.empty_like(a)
    for i, qi in enumerate(primes):
        out[i] = (a[i] * b[i]) % qi
    return out


def vscal(a, s, primes):
    out = np.empty_like(a)
    for i, qi in enumerate(primes):
        out[i] = (a[i] * np.int64(int(s) % qi)) % qi
    return out
