import numpy as np
from fractions import Fraction


def poly_eval_int(coeffs, x):
    v = 0
    for c in reversed(coeffs):
        v = v * x + c
    return v


def digits_base(v, b, length):
    out = []
    for _ in range(length):
        out.append(v % b)
        v //= b
    return out


def balanced_digits(T, b, length):
    out = []
    v = T
    for _ in range(length):
        r = v % b
        if r > b // 2:
            r -= b
        out.append(r)
        v = (v - r) // b
    if v != 0:
        return None
    return out


def real_root_count(coeffs):
    rts = np.roots([float(c) for c in reversed(coeffs)])
    return int(sum(1 for z in rts if abs(z.imag) < 1e-9 * max(1.0, abs(z))))


def find_poly(M, d, max_tries=600, seed=0):
    M = int(M)
    b0 = max(2, int(round(M ** (1.0 / d))))
    while b0 ** d < M:
        b0 += 1
    while b0 > 2 and (b0 - 1) ** d >= M:
        b0 -= 1
    if b0 ** d == M and d >= 2:
        cand = [b0, -1] + [0] * (d - 2) + [1]
        if poly_eval_int(cand, b0) == M and real_root_count(cand) == 0:
            return b0, cand
    rng = np.random.default_rng(seed)
    best = None
    for b in (b0, b0 + 1, b0 + 2, b0 + 3):
        if b < 2 or b ** d < M:
            continue
        base = balanced_digits(M - b ** d, b, d)
        if base is None:
            base = [-x for x in digits_base(b ** d - M, b, d)]
        cand = list(base) + [1]
        if poly_eval_int(cand, b) != M:
            continue
        score = (real_root_count(cand), max(abs(c) for c in cand[:d]))
        if best is None or score < best[0]:
            best = (score, b, cand)
        if score[0] == 0:
            return b, cand
        for _ in range(max_tries):
            delta = rng.integers(-1, 2, size=d - 1)
            c = list(base)
            for i in range(d - 1):
                if delta[i]:
                    c[i] -= b * int(delta[i])
                    c[i + 1] += int(delta[i])
            cc = c + [1]
            if poly_eval_int(cc, b) != M:
                continue
            sc = (real_root_count(cc), max(abs(x) for x in cc[:d]))
            if sc < best[0]:
                best = (sc, b, cc)
            if sc[0] == 0:
                return b, cc
    return best[1], best[2]


def lll(basis, delta=0.99):
    B = [list(map(int, row)) for row in basis]
    n = len(B)
    if n == 0:
        return B
    m = len(B[0])

    def gso(B):
        A = np.array(B, dtype=np.float64)
        Bs = np.zeros((n, m))
        mu = np.zeros((n, n))
        for i in range(n):
            v = A[i].copy()
            for j in range(i):
                d = float(np.dot(Bs[j], Bs[j]))
                if d > 0:
                    mu[i, j] = float(np.dot(A[i], Bs[j])) / d
                    v = v - mu[i, j] * Bs[j]
            Bs[i] = v
        return Bs, mu

    Bs, mu = gso(B)
    k = 1
    guard = 0
    while k < n and guard < 200 * n * n:
        guard += 1
        for j in range(k - 1, -1, -1):
            q = int(round(mu[k][j]))
            if q:
                B[k] = [B[k][t] - q * B[j][t] for t in range(m)]
                Bs, mu = gso(B)
        if float(np.dot(Bs[k], Bs[k])) >= (delta - mu[k][k - 1] ** 2) * float(np.dot(Bs[k - 1], Bs[k - 1])):
            k += 1
        else:
            B[k], B[k - 1] = B[k - 1], B[k]
            Bs, mu = gso(B)
            k = max(k - 1, 1)
    return B


class ARing:
    def __init__(self, M, d, coeffs=None, b=None, seed=0):
        self.M = int(M)
        self.d = int(d)
        if coeffs is None:
            b, coeffs = find_poly(self.M, self.d, seed=seed)
        self.b = int(b)
        self.F = [int(c) for c in coeffs]
        assert len(self.F) == self.d + 1 and self.F[-1] == 1
        assert poly_eval_int(self.F, self.b) == self.M
        self.red = np.array([-c for c in self.F[:self.d]], dtype=np.float64)
        rts = np.roots([float(c) for c in reversed(self.F)])
        self.roots_r = np.array([z.real for z in rts if abs(z.imag) < 1e-9 * max(1.0, abs(z))])
        self.roots_c = np.array(sorted([z for z in rts if z.imag > 1e-9 * max(1.0, abs(z))],
                                       key=lambda z: -z.real), dtype=np.complex128)
        self.r1 = len(self.roots_r)
        self.r2 = len(self.roots_c)
        self.block = self.r1 + self.r2
        self.Uc = self.roots_c[:, None] ** np.arange(self.d)[None, :]
        self.Ur = (self.roots_r[:, None] ** np.arange(self.d)[None, :]) if self.r1 else np.zeros((0, self.d))
        full = np.vstack([self.Uc, np.conj(self.Uc), self.Ur.astype(np.complex128)])
        inv = np.linalg.inv(full)
        self.Vc = inv[:, :self.r2]
        self.Vr = inv[:, 2 * self.r2:]
        self.Tt = self._mul_matrix([-self.b, 1])
        self.Ttinv = np.linalg.inv(self.Tt)
        self.t_slots_c = self.roots_c - self.b
        self.tinv_slots_c = 1.0 / (self.roots_c - self.b)
        self.t_slots_r = self.roots_r - self.b
        self.gamma = self.d * (max(abs(c) for c in self.F[:self.d]) + 2)
        self.tau_norm = float(np.max(np.sum(np.abs(np.vstack([self.Uc, self.Ur.astype(np.complex128)])), axis=1)))

    def ok(self):
        return self.r1 == 0

    def _mul_matrix(self, poly):
        d = self.d
        Mx = np.zeros((d, d), dtype=np.float64)
        for j in range(d):
            e = np.zeros(d)
            e[j] = 1.0
            Mx[:, j] = self.mul_poly(poly, e)
        return Mx

    def reduce(self, c):
        c = np.array(c, dtype=np.float64)
        d = self.d
        for i in range(len(c) - 1, d - 1, -1):
            v = c[i]
            if v == 0.0:
                continue
            c[i] = 0.0
            c[i - d:i] += v * self.red
        return c[:d]

    def mul_poly(self, a, c):
        return self.reduce(np.convolve(np.asarray(a, dtype=np.float64), np.asarray(c, dtype=np.float64)))

    def digits(self, m):
        return digits_base(int(m) % self.M, self.b, self.d)

    def triangle(self, m):
        return self.Ttinv @ np.array(self.digits(m), dtype=np.float64)

    def mul_t(self, g):
        return self.Tt @ np.asarray(g, dtype=np.float64)

    def decode_triangle(self, g):
        h = np.rint(self.mul_t(g)).astype(object)
        return int(poly_eval_int([int(x) for x in h], self.b) % self.M)

    def mul(self, g1, g2):
        return self.mul_poly(g1, g2)

    def slots(self, m):
        g = self.triangle(m)
        return self.slot_vec(g)

    def slot_vec(self, g):
        g = np.asarray(g, dtype=np.float64)
        zc = self.Uc @ g
        if self.r1:
            zr = self.Ur @ g
            return np.concatenate([zc, zr.astype(np.complex128)])
        return zc

    def coeffs_from_slots(self, z):
        zc = np.asarray(z[:self.r2], dtype=np.complex128)
        g = 2.0 * np.real(self.Vc @ zc)
        if self.r1:
            g = g + np.real(self.Vr @ np.asarray(z[self.r2:], dtype=np.complex128))
        return g

    def coeff_matrix(self):
        S = self.block
        Mx = np.zeros((self.d, S), dtype=np.complex128)
        Mx[:, :self.r2] = self.Vc
        if self.r1:
            Mx[:, self.r2:] = self.Vr
        return Mx

    def slot_matrix(self):
        if self.r1:
            return np.vstack([self.Uc, self.Ur.astype(np.complex128)])
        return self.Uc


def two_slot_poly(M, d):
    M = int(M)
    b = max(2, int(round(M ** (1.0 / d))))
    while b ** d < M:
        b += 1
    while b > 2 and (b - 1) ** d >= M:
        b -= 1
    q2 = b * (b + 1)
    cur = [0] * d
    cur[0] = -M
    cur[d - 2] += q2
    cur[d - 1] += -(2 * b + 1)
    basis = []
    for i in range(d - 2):
        row = [0] * d
        row[i] += q2
        row[i + 1] += -(2 * b + 1)
        row[i + 2] += 1
        basis.append(row)
    for i in range(d - 2):
        q = (2 * cur[i] + q2) // (2 * q2)
        if q:
            cur[i] -= q * q2
            cur[i + 1] += q * (2 * b + 1)
            cur[i + 2] -= q
    red = lll(basis)
    cur = babai(cur, red)
    return b, list(cur) + [1], red


def babai(target, basis):
    k = len(basis)
    if k == 0:
        return list(target)
    A = np.array(basis, dtype=np.float64)
    m = A.shape[1]
    Bs = np.zeros((k, m))
    for i in range(k):
        v = A[i].copy()
        for j in range(i):
            dd = float(np.dot(Bs[j], Bs[j]))
            if dd > 0:
                v = v - (float(np.dot(A[i], Bs[j])) / dd) * Bs[j]
        Bs[i] = v
    w = [int(x) for x in target]
    for i in range(k - 1, -1, -1):
        dd = float(np.dot(Bs[i], Bs[i]))
        if dd <= 0:
            continue
        num = float(np.dot(np.array(w, dtype=np.float64), Bs[i]))
        q = int(round(num / dd))
        if q:
            w = [a - q * x for a, x in zip(w, basis[i])]
    return w
