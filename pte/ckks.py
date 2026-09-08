import numpy as np
from .ring import Ring, fast_base_conv


class Ciphertext:
    __slots__ = ("c", "level", "scale", "ctx")

    def __init__(self, c, level, scale, ctx=None):
        self.c = c
        self.level = level
        self.scale = scale
        self.ctx = ctx

    def copy(self):
        return Ciphertext([x.copy() for x in self.c], self.level, self.scale, self.ctx)

    def nbytes(self):
        return sum(x.size * 8 for x in self.c)


class Plaintext:
    __slots__ = ("m", "level", "scale")

    def __init__(self, m, level, scale):
        self.m = m
        self.level = level
        self.scale = scale

    def nbytes(self):
        return self.m.size * 8


class Encoder:
    def __init__(self, n):
        self.n = n
        self.M = 2 * n
        self.slots = n // 2
        g = 1
        rot = np.zeros(self.slots, dtype=np.int64)
        for j in range(self.slots):
            rot[j] = g
            g = (g * 5) % self.M
        self.rot = rot
        self.zeta = np.exp(2j * np.pi / self.M)
        self.pw = self.zeta ** np.arange(n)
        self.ipw = self.zeta ** (-np.arange(n))
        self.idx = ((rot - 1) // 2).astype(np.int64)
        self.cidx = ((((self.M - rot) % self.M) - 1) // 2).astype(np.int64)

    def decode_poly(self, coeffs):
        u = np.asarray(coeffs, dtype=np.complex128) * self.pw
        A = np.fft.ifft(u) * self.n
        return A[self.idx]

    def encode_poly(self, z):
        z = np.asarray(z, dtype=np.complex128)
        A = np.zeros(self.n, dtype=np.complex128)
        A[self.idx] = z
        A[self.cidx] = np.conj(z)
        u = np.fft.fft(A) / self.n
        return (u * self.ipw).real


class CkksContext:
    def __init__(self, n, log_q, levels, log_p, num_aux, log_delta,
                 h=None, sigma=3.2, dnum=None, seed=0):
        self.ring = Ring(n, log_q, levels, log_p, num_aux)
        self.n = n
        self.L = levels
        self.slots = n // 2
        self.delta = float(1 << log_delta)
        self.log_delta = log_delta
        self.sigma = sigma
        self.enc = Encoder(n)
        self.rng = np.random.default_rng(seed)
        self.h = h if h is not None else max(16, n // 16)
        self.dnum = dnum if dnum else max(1, -(-(levels + 1) // max(1, num_aux)))
        gs = -(-(levels + 1) // self.dnum)
        qmax = 1
        for qi in sorted(self.ring.q)[-gs:]:
            qmax *= qi
        assert self.ring.ctx.P >= qmax, "auxiliary modulus too small for dnum"

        self.sk = None
        self.rlk = None
        self.rot_keys = {}
        self.conj_key = None
        self.ks_count = 0
        self.mul_count = 0
        self.boot_count = 0
        self.aut_cache = {}
        self._grp = None

    def groups(self):
        if self._grp is None:
            idx = list(range(self.L + 1))
            size = (len(idx) + self.dnum - 1) // self.dnum
            self._grp = [idx[i:i + size] for i in range(0, len(idx), size)]
        return self._grp

    def keygen(self):
        r = self.ring
        self.sk = r.sample_ternary_sparse(self.h, self.rng)
        pr = r.primes(self.L, True)
        self.sk_c = r.lift(self.sk, pr)
        self.sk_ntt = r.ntt(self.sk_c, pr)
        s2 = np.empty_like(self.sk_ntt)
        for i, qi in enumerate(pr):
            s2[i] = (self.sk_ntt[i] * self.sk_ntt[i]) % qi
        self.rlk = self._ksk_from(s2)
        return self.sk

    def _ksk_from(self, target_ntt):
        r = self.ring
        pr = r.primes(self.L, True)
        P = r.ctx.P
        Q = r.ctx.Q
        keys = []
        for grp in self.groups():
            Qj = 1
            for i in grp:
                Qj *= r.q[i]
            fac = (Q // Qj) * pow((Q // Qj) % Qj, -1, Qj) % Q
            a = r.sample_uniform(pr, self.rng)
            e = r.sample_gauss(pr, self.sigma, self.rng)
            ent = r.ntt(e, pr)
            b = np.empty_like(a)
            for i, qi in enumerate(pr):
                cf = np.int64((P % qi) * (fac % qi) % qi)
                t = (target_ntt[i] * cf) % qi
                b[i] = (-(a[i] * self.sk_ntt[i]) % qi + t + ent[i]) % qi
            keys.append((b, a))
        return keys

    def encrypt(self, pt):
        r = self.ring
        lvl = pt.level
        pr = r.primes(lvl)
        a = r.sample_uniform(pr, self.rng)
        ent = r.ntt(r.sample_gauss(pr, self.sigma, self.rng), pr)
        skn = self.sk_ntt[:lvl + 1]
        b = np.empty_like(a)
        for i, qi in enumerate(pr):
            b[i] = (-(a[i] * skn[i]) % qi + pt.m[i] + ent[i]) % qi
        return Ciphertext([b, a], lvl, pt.scale, self)

    def decrypt(self, ct):
        r = self.ring
        pr = r.primes(ct.level)
        skn = self.sk_ntt[:ct.level + 1]
        acc = ct.c[0].copy()
        pw = None
        for j in range(1, len(ct.c)):
            pw = skn.copy() if pw is None else np.array(
                [(pw[i] * skn[i]) % qi for i, qi in enumerate(pr)], dtype=np.int64)
            for i, qi in enumerate(pr):
                acc[i] = (acc[i] + ct.c[j][i] * pw[i]) % qi
        return r.intt(acc, pr)

    def encode(self, z, level=None, scale=None):
        lvl = self.L if level is None else level
        sc = self.delta if scale is None else scale
        m = self.enc.encode_poly(np.asarray(z))
        return self._pt_from_real(m * sc, lvl, sc)

    def encode_coeff(self, coeffs, level=None, scale=None):
        lvl = self.L if level is None else level
        sc = self.delta if scale is None else scale
        return self._pt_from_real(np.asarray(coeffs, dtype=np.float64) * sc, lvl, sc)

    def _pt_from_real(self, vals, lvl, sc):
        v = np.asarray(vals, dtype=np.float64)
        if np.max(np.abs(v)) < 4.0e18:
            coe = np.rint(v).astype(np.int64)
        else:
            coe = np.array([int(round(float(x))) for x in v], dtype=object)
        pr = self.ring.primes(lvl)
        return Plaintext(self.ring.ntt(self.ring.lift(coe, pr), pr), lvl, sc)

    def decode(self, ct):
        return self.enc.decode_poly(self.raw_coeffs(ct)) / ct.scale

    def decode_coeff(self, ct):
        return self.raw_coeffs(ct) / ct.scale

    def raw_coeffs(self, ct):
        pr = self.ring.primes(ct.level)
        m = self.decrypt(ct)
        coe = self.ring.ctx.from_rns(m, pr)
        return np.array([float(x) for x in coe])

    def pt_scale(self, level):
        return float(self.ring.q[level])

    def pt(self, vals, level):
        return self.encode(vals, level=level, scale=self.pt_scale(level))

    def step(self, ct, target_scale):
        l = ct.level
        ql = self.ring.q[l]
        u = int(round(target_scale * ql / ct.scale))
        if u < 1:
            u = 1
        t = self.mul_int(ct, u)
        t = Ciphertext(t.c, l, ct.scale * u, self)
        return self.rescale(t)

    def align(self, a, b):
        while a.level > b.level:
            a = self.step(a, b.scale if a.level - 1 == b.level else self.delta)
        while b.level > a.level:
            b = self.step(b, a.scale if b.level - 1 == a.level else self.delta)
        if abs(a.scale / b.scale - 1.0) > 1e-9:
            tgt = self.delta
            a = self.step(a, tgt)
            b = self.step(b, tgt)
        return a, b

    def add(self, a, b):
        a, b = self.align(a, b)
        pr = self.ring.primes(min(a.level, b.level))
        k = max(len(a.c), len(b.c))
        out = []
        for i in range(k):
            if i < len(a.c) and i < len(b.c):
                v = np.empty((len(pr), self.n), dtype=np.int64)
                for j, qi in enumerate(pr):
                    v[j] = (a.c[i][j] + b.c[i][j]) % qi
                out.append(v)
            elif i < len(a.c):
                out.append(a.c[i][:len(pr)].copy())
            else:
                out.append(b.c[i][:len(pr)].copy())
        return Ciphertext(out, min(a.level, b.level), a.scale, self)

    def sub(self, a, b):
        a, b = self.align(a, b)
        pr = self.ring.primes(min(a.level, b.level))
        k = max(len(a.c), len(b.c))
        out = []
        for i in range(k):
            if i < len(a.c) and i < len(b.c):
                v = np.empty((len(pr), self.n), dtype=np.int64)
                for j, qi in enumerate(pr):
                    v[j] = (a.c[i][j] - b.c[i][j]) % qi
                out.append(v)
            elif i < len(a.c):
                out.append(a.c[i][:len(pr)].copy())
            else:
                v = np.empty((len(pr), self.n), dtype=np.int64)
                for j, qi in enumerate(pr):
                    v[j] = (-b.c[i][j]) % qi
                out.append(v)
        return Ciphertext(out, min(a.level, b.level), a.scale, self)

    def add_const(self, ct, val):
        v = complex(val)
        if abs(v.imag) > 1e-14:
            pt = self.encode(np.full(self.slots, v), level=ct.level, scale=ct.scale)
            return self.add_plain(ct, pt)
        pr = self.ring.primes(ct.level)
        c0 = ct.c[0].copy()
        cst = int(round(v.real * ct.scale))
        for j, qi in enumerate(pr):
            c0[j] = (c0[j] + np.int64(cst % qi)) % qi
        return Ciphertext([c0] + [x.copy() for x in ct.c[1:]], ct.level, ct.scale, self)

    def add_plain(self, ct, pt):
        pr = self.ring.primes(ct.level)
        c0 = ct.c[0].copy()
        for j, qi in enumerate(pr):
            c0[j] = (c0[j] + pt.m[j]) % qi
        return Ciphertext([c0] + [x.copy() for x in ct.c[1:]], ct.level, ct.scale, self)

    def mul_int(self, ct, s):
        pr = self.ring.primes(ct.level)
        out = []
        for x in ct.c:
            v = np.empty_like(x)
            for j, qi in enumerate(pr):
                v[j] = (x[j] * np.int64(int(s) % qi)) % qi
            out.append(v)
        return Ciphertext(out, ct.level, ct.scale, self)

    def mul_plain(self, ct, pt):
        pr = self.ring.primes(ct.level)
        out = []
        for x in ct.c:
            v = np.empty((len(pr), self.n), dtype=np.int64)
            for j, qi in enumerate(pr):
                v[j] = (x[j] * pt.m[j]) % qi
            out.append(v)
        return Ciphertext(out, ct.level, ct.scale * pt.scale, self)

    def tensor(self, a, b):
        lvl = min(a.level, b.level)
        pr = self.ring.primes(lvl)
        d0 = np.empty((len(pr), self.n), dtype=np.int64)
        d1 = np.empty_like(d0)
        d2 = np.empty_like(d0)
        for j, qi in enumerate(pr):
            d0[j] = (a.c[0][j] * b.c[0][j]) % qi
            d1[j] = (a.c[0][j] * b.c[1][j] % qi + a.c[1][j] * b.c[0][j] % qi) % qi
            d2[j] = (a.c[1][j] * b.c[1][j]) % qi
        return Ciphertext([d0, d1, d2], lvl, a.scale * b.scale, self)

    def key_switch(self, ct3, ksk):
        r = self.ring
        lvl = ct3.level
        pr = r.primes(lvl)
        prp = r.primes(lvl, True)
        na = len(r.p)
        dc = r.intt(ct3.c[2], pr)
        acc0 = np.zeros((len(prp), self.n), dtype=np.int64)
        acc1 = np.zeros((len(prp), self.n), dtype=np.int64)
        sel = list(range(lvl + 1)) + list(range(self.L + 1, self.L + 1 + na))
        for gi, grp in enumerate(self.groups()):
            sub = [i for i in grp if i <= lvl]
            if not sub:
                continue
            src = [r.q[i] for i in sub]
            keep = set(sub)
            dstidx = [k for k in range(len(prp)) if k not in keep]
            dst = [prp[k] for k in dstidx]
            part = dc[sub]
            ext = fast_base_conv(r, part, src, dst)
            full = np.zeros((len(prp), self.n), dtype=np.int64)
            for k, i in enumerate(sub):
                full[i] = part[k]
            for k, di in enumerate(dstidx):
                full[di] = ext[k]
            fn = r.ntt(full, prp)
            b, a = ksk[gi]
            for k, qi in enumerate(prp):
                acc0[k] = (acc0[k] + fn[k] * b[sel[k]]) % qi
                acc1[k] = (acc1[k] + fn[k] * a[sel[k]]) % qi
        self.ks_count += 1
        r0 = self._mod_down(acc0, lvl)
        r1 = self._mod_down(acc1, lvl)
        out0 = np.empty((len(pr), self.n), dtype=np.int64)
        out1 = np.empty_like(out0)
        for j, qi in enumerate(pr):
            out0[j] = (ct3.c[0][j] + r0[j]) % qi
            out1[j] = (ct3.c[1][j] + r1[j]) % qi
        return Ciphertext([out0, out1], lvl, ct3.scale, self)

    def _mod_down(self, acc, lvl):
        r = self.ring
        pr = r.primes(lvl)
        topc = r.intt(acc[lvl + 1:], r.p)
        ext = r.ntt(fast_base_conv(r, topc, r.p, pr), pr)
        out = np.empty((len(pr), self.n), dtype=np.int64)
        for j, qi in enumerate(pr):
            inv = pow(r.ctx.P % qi, qi - 2, qi)
            out[j] = ((acc[j] - ext[j]) * np.int64(inv)) % qi
        return out

    def relin(self, ct):
        if len(ct.c) < 3:
            return ct
        return self.key_switch(ct, self.rlk)

    def mul(self, a, b):
        self.mul_count += 1
        a, b = self.align(a, b)
        return self.rescale(self.relin(self.tensor(a, b)))

    def mul_const(self, ct, c):
        v = np.full(self.slots, c, dtype=np.complex128)
        return self.rescale(self.mul_plain(ct, self.pt(v, ct.level)))

    def mul_plain_rs(self, a, pt):
        return self.rescale(self.mul_plain(a, pt))

    def rescale(self, ct):
        r = self.ring
        lvl = ct.level
        pr = r.primes(lvl)
        ql = pr[-1]
        out = []
        for x in ct.c:
            last = r.ctx.ntt[ql].inverse(x[lvl])
            new = np.empty((lvl, self.n), dtype=np.int64)
            for j, qi in enumerate(pr[:-1]):
                t = r.ctx.ntt[qi].forward(last % qi)
                inv = pow(ql % qi, qi - 2, qi)
                new[j] = ((x[j] - t) * np.int64(inv)) % qi
            out.append(new)
        return Ciphertext(out, lvl - 1, ct.scale / ql, self)

    def drop_to(self, ct, lvl):
        if ct.level <= lvl:
            return ct
        return Ciphertext([x[:lvl + 1].copy() for x in ct.c], lvl, ct.scale, self)

    def _aut(self, g):
        if g not in self.aut_cache:
            n = self.n
            i = np.arange(n, dtype=np.int64)
            t = (i * g) % (2 * n)
            idx = t % n
            sgn = np.where(t >= n, -1, 1).astype(np.int64)
            self.aut_cache[g] = (idx, sgn)
        return self.aut_cache[g]

    def apply_aut(self, mat, primes, g):
        idx, sgn = self._aut(g)
        out = np.empty_like(mat)
        for j, qi in enumerate(primes):
            v = np.zeros(self.n, dtype=np.int64)
            v[idx] = mat[j] * sgn
            out[j] = v % qi
        return out

    def gen_rot_key(self, k):
        k = k % self.slots
        if k == 0 or k in self.rot_keys:
            return
        r = self.ring
        pr = r.primes(self.L, True)
        g = pow(5, k, 2 * self.n)
        self.rot_keys[k] = self._ksk_from(r.ntt(self.apply_aut(self.sk_c, pr, g), pr))

    def gen_conj_key(self):
        if self.conj_key is None:
            r = self.ring
            pr = r.primes(self.L, True)
            self.conj_key = self._ksk_from(
                r.ntt(self.apply_aut(self.sk_c, pr, 2 * self.n - 1), pr))

    def _aut_ct(self, ct, g, ksk):
        r = self.ring
        pr = r.primes(ct.level)
        au = [r.ntt(self.apply_aut(r.intt(x, pr), pr, g), pr) for x in ct.c]
        tmp = Ciphertext([au[0], np.zeros_like(au[1]), au[1]], ct.level, ct.scale, self)
        return self.key_switch(tmp, ksk)

    def rotate(self, ct, k):
        k = k % self.slots
        if k == 0:
            return ct.copy()
        self.gen_rot_key(k)
        return self._aut_ct(ct, pow(5, k, 2 * self.n), self.rot_keys[k])

    def conjugate(self, ct):
        self.gen_conj_key()
        return self._aut_ct(ct, 2 * self.n - 1, self.conj_key)
