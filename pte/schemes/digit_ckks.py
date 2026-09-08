import numpy as np
from ..bootstrap import Bootstrapper


class RadixDigitFHE:
    def __init__(self, ctx, n_bits, nu=4, boot=None, mode="log"):
        self.ctx = ctx
        self.n = n_bits
        self.nu = nu
        self.base = 1 << nu
        self.k = n_bits // nu
        self.S = ctx.slots
        assert self.S % self.k == 0
        self.W = self.S // self.k
        self.mode = mode
        self.boot = boot if boot is not None else Bootstrapper(ctx, K=8, s=3, deg=24, p=self.base)
        self.mask_cache = {}

    def mask(self, positions, weight=1.0):
        key = (tuple(sorted(positions)), weight)
        if key not in self.mask_cache:
            v = np.zeros(self.S, dtype=np.complex128)
            for w in range(self.W):
                for q in key[0]:
                    v[w * self.k + q] = weight
            self.mask_cache[key] = v
        return self.mask_cache[key]

    def digits(self, m):
        out = []
        for _ in range(self.k):
            out.append(m % self.base)
            m //= self.base
        return out

    def encrypt(self, words, level=None):
        lvl = self.ctx.L if level is None else level
        z = np.zeros(self.S, dtype=np.complex128)
        for w, m in enumerate(words):
            dg = self.digits(int(m) % (1 << self.n))
            for j in range(self.k):
                z[w * self.k + j] = dg[j] / self.base
        return self.ctx.encrypt(self.ctx.encode(z, level=lvl))

    def decrypt(self, ct):
        z = self.ctx.decode(ct).real
        out = []
        for w in range(self.W):
            v = 0
            for j in range(self.k):
                v += int(round(z[w * self.k + j] * self.base)) * (self.base ** j)
            out.append(v % (1 << self.n))
        return out

    def add(self, a, b):
        return self.ctx.add(a, b)

    def broadcast(self, ct, i):
        ctx = self.ctx
        cur = ctx.rescale(ctx.mul_plain(ct, ctx.pt(self.mask([i]), ct.level)))
        if i:
            cur = ctx.rotate(cur, i)
        step = 1
        while step < self.k:
            cur = ctx.add(cur, ctx.rotate(cur, -step))
            step *= 2
        return cur

    def mul(self, a, b):
        ctx = self.ctx
        acc = None
        for i in range(self.k):
            bi = self.broadcast(a, i)
            t = ctx.mul(bi, b)
            t = ctx.rescale(ctx.mul_plain(t, ctx.pt(self.mask(range(self.k - i), self.base), t.level)))
            if i:
                t = ctx.rotate(t, -i)
            acc = t if acc is None else ctx.add(acc, t)
        return self.reduce(acc)

    def _low_digit(self, ct):
        tab = [j / self.base for j in range(self.base)]
        return self.boot.clear_and_lut(ct, [tab], self.base)[0]

    def reduce(self, ct):
        if self.mode == "seq":
            return self._reduce_seq(ct)
        return self._reduce_log(ct)

    def _shift_up(self, ct):
        ctx = self.ctx
        t = ctx.rescale(ctx.mul_plain(ct, ctx.pt(self.mask(range(self.k - 1)), ct.level)))
        return ctx.rotate(t, -1)

    def _magnitude_rounds(self, ct):
        ctx = self.ctx
        cur = ct
        mag = 3 + int(np.ceil(np.log2(max(2, self.k)) / self.nu))
        for _ in range(mag):
            s = self._low_digit(cur)
            carry = ctx.mul_const(ctx.sub(cur, s), 1.0 / self.base)
            cur = ctx.add(s, self._shift_up(carry))
        return cur

    def _split_digit_carry(self, cur):
        ctx = self.ctx
        pp = 2 * self.base
        v = ctx.mul_const(cur, 0.5)
        tabs = [[(t % self.base) / self.base for t in range(pp)],
                [float(t // self.base) / self.base for t in range(pp)]]
        return self.boot.clear_and_lut(v, tabs, pp)

    def _reduce_seq(self, ct):
        ctx = self.ctx
        cur = self._magnitude_rounds(ct)
        for _ in range(self.k):
            s, g = self._split_digit_carry(cur)
            cur = ctx.add(s, self._shift_up(g))
        return self._low_digit(cur)

    def _reduce_log(self, ct):
        ctx = self.ctx
        cur = self._magnitude_rounds(ct)
        tabG = [1.0 if j == self.base - 1 else 0.0 for j in range(self.base)]
        tabS = [j / self.base for j in range(self.base)]
        outs = self.boot.clear_and_lut(cur, [tabS, tabG], self.base)
        s, P = outs[0], outs[1]
        G = ctx.mul_const(ctx.sub(cur, s), 1.0 / self.base)
        G = self._guard(G, 6)
        P = self._guard(P, 6)
        s = self._guard(s, 4)
        step = 1
        while step < self.k:
            Gs = ctx.rotate(self._maskcut(G, step), -step)
            Ps = ctx.rotate(self._maskcut(P, step), -step)
            G = ctx.add(G, ctx.mul(P, Gs))
            P = ctx.mul(P, Ps)
            step *= 2
            if step < self.k:
                G = self._guard(G, 6)
                P = self._guard(P, 6)
        cin = ctx.rotate(self._maskcut(G, 1), -1)
        res = ctx.add(s, ctx.mul_const(cin, 1.0 / self.base))
        return self._low_digit(res)

    def _bool_refresh(self, ct):
        return self.boot.clear_and_lut(ct, [[0.0, 1.0]], 2)[0]

    def _guard(self, ct, need=3):
        if ct.level < need:
            return self._bool_refresh(ct)
        return ct

    def _maskcut(self, ct, step):
        ctx = self.ctx
        return ctx.rescale(ctx.mul_plain(ct, ctx.pt(self.mask(range(self.k - step)), ct.level)))

    def refresh(self, ct):
        return self._low_digit(ct)

    def a2b(self, ct):
        tabs = [[float((j >> l) & 1) for j in range(self.base)] for l in range(self.nu)]
        return self.boot.clear_and_lut(ct, tabs, self.base)

    def b2a(self, planes):
        ctx = self.ctx
        acc = None
        for l, pl in enumerate(planes):
            t = ctx.mul_const(pl, float(1 << l) / self.base)
            acc = t if acc is None else ctx.add(acc, t)
        return acc

    def b2b(self, plane):
        return self.boot.clear_and_lut(plane, [[0.0, 1.0]], 2)[0]
