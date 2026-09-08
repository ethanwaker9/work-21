import numpy as np
from .aring import ARing
from .ckks import Ciphertext
from .lintrans import LinTransform, coeff_to_slot_matrices, slot_to_coeff_matrix
from .bootstrap import Bootstrapper, truncate, mod_raise


def tile(vec, reps):
    return np.tile(np.asarray(vec, dtype=np.complex128), reps)


def block_matrix(block, reps):
    s = block.shape[0]
    out = np.zeros((s * reps, s * reps), dtype=np.complex128)
    for i in range(reps):
        out[i * s:(i + 1) * s, i * s:(i + 1) * s] = block
    return out


class TriangleFHE:
    def __init__(self, ctx, M, d, digits_per_round=1, boot=None, aring=None):
        self.ctx = ctx
        self.R = aring if aring is not None else ARing(M, d)
        assert self.R.r1 == 0, "arithmetic ring must have no real roots"
        self.M = self.R.M
        self.d = self.R.d
        self.b = self.R.b
        self.B = self.d // 2
        self.S = ctx.slots
        assert self.S % self.B == 0
        self.W = self.S // self.B
        self.c = digits_per_round
        assert self.B % self.c == 0
        self.p = self.b ** self.c
        self.rbits = max(1, int(round(np.log2(self.b))))
        self.exact_radix = (1 << self.rbits) == self.b
        self.boot = boot if boot is not None else Bootstrapper(ctx, K=8, s=3, deg=24, p=self.p)
        self._build()

    def _build(self):
        ctx = self.ctx
        R = self.R
        B, W = self.B, self.W
        Vc = R.Vc
        Uc = R.Uc
        V0 = Vc[:B, :].copy()
        V1 = Vc[B:, :].copy()
        Vsp = V0.copy()
        Vsp[0, :] = Vc[0, :] + Vc[self.d - 1, :]
        self.lt_z2c0 = LinTransform(ctx, block_matrix(V0, W), name="Z2C0")
        self.lt_z2c1 = LinTransform(ctx, block_matrix(V1, W), name="Z2C1")
        self.lt_z2c0s = LinTransform(ctx, block_matrix(Vsp, W), name="Z2C0s")
        self.lt_c2z0 = LinTransform(ctx, block_matrix(Uc[:, :B], W), name="C2Z0")
        self.lt_c2z1 = LinTransform(ctx, block_matrix(Uc[:, B:], W), name="C2Z1")
        self.t_vec = tile(R.t_slots_c, W)
        self.tinv_vec = tile(R.tinv_slots_c, W)
        A0, A1 = coeff_to_slot_matrices(ctx)
        self.lt_c2s0 = LinTransform(ctx, A0, name="CtS0")
        self.lt_c2s1 = LinTransform(ctx, A1, name="CtS1")
        self.lt_s2c = LinTransform(ctx, slot_to_coeff_matrix(ctx), name="StC")
        self.mask_cache = {}
        bitmat = np.zeros((self.d, self.d * self.rbits), dtype=np.float64)
        for i in range(self.d):
            for l in range(self.rbits):
                bitmat[i, i * self.rbits + l] = float(1 << l)
        self.bit_to_digit = bitmat

    def mask(self, positions):
        key = tuple(sorted(positions))
        if key not in self.mask_cache:
            v = np.zeros(self.S, dtype=np.complex128)
            for k in range(self.W):
                for q in key:
                    v[k * self.B + q] = 1.0
            self.mask_cache[key] = v
        return self.mask_cache[key]

    def words_to_slots(self, words):
        z = np.zeros(self.S, dtype=np.complex128)
        for k, m in enumerate(words):
            z[k * self.B:(k + 1) * self.B] = self.R.slots(m)
        return z

    def encrypt(self, words, level=None):
        lvl = self.ctx.L if level is None else level
        z = self.words_to_slots(words)
        return self.ctx.encrypt(self.ctx.encode(z, level=lvl))

    def decrypt(self, ct):
        z = self.ctx.decode(ct)
        out = []
        for k in range(self.W):
            g = self.R.coeffs_from_slots(z[k * self.B:(k + 1) * self.B])
            out.append(self.R.decode_triangle(g))
        return out

    def coeff_error(self, ct, words):
        z = self.ctx.decode(ct)
        err = 0.0
        for k in range(self.W):
            g = self.R.coeffs_from_slots(z[k * self.B:(k + 1) * self.B])
            ref = self.R.triangle(words[k])
            err = max(err, float(np.max(np.abs(g - ref - np.rint(g - ref)))))
        return err

    def add(self, a, b):
        return self.ctx.add(a, b)

    def mul(self, a, b):
        ctx = self.ctx
        t = ctx.mul(a, b)
        return ctx.rescale(ctx.mul_plain(t, ctx.pt(self.t_vec, t.level)))

    def mul_flat(self, a, words):
        ctx = self.ctx
        z = np.zeros(self.S, dtype=np.complex128)
        for k, m in enumerate(words):
            z[k * self.B:(k + 1) * self.B] = self.R.Uc @ np.array(self.R.digits(m), dtype=np.float64)
        return ctx.rescale(ctx.mul_plain(a, ctx.pt(z, a.level)))

    def z2c(self, ct, special=False):
        ctx = self.ctx
        lt0 = self.lt_z2c0s if special else self.lt_z2c0
        w0 = lt0.apply(ct)
        h0 = ctx.add(w0, ctx.conjugate(w0))
        w1 = self.lt_z2c1.apply(ct)
        h1 = ctx.add(w1, ctx.conjugate(w1))
        return h0, h1

    def c2z(self, h0, h1):
        return self.ctx.add(self.lt_c2z0.apply(h0), self.lt_c2z1.apply(h1))

    def _pack(self, h0, h1):
        ctx = self.ctx
        return ctx.add(h0, ctx.mul_const(h1, 1j))

    def _raise(self, comb):
        ctx = self.ctx
        a = self.lt_s2c.apply(comb)
        a = truncate(ctx, a)
        a = mod_raise(ctx, a)
        v0 = self.lt_c2s0.apply(a)
        v1 = self.lt_c2s1.apply(a)
        return ctx.add(v0, ctx.conjugate(v0)), ctx.add(v1, ctx.conjugate(v1))

    def refresh_overflow(self, ct):
        self.ctx.boot_count += 1
        h0, h1 = self.z2c(ct)
        g0, g1 = self._raise(self._pack(h0, h1))
        return self.c2z(g0, g1)

    def refresh_noise(self, ct):
        ctx = self.ctx
        ctx.boot_count += 1
        tct = ctx.rescale(ctx.mul_plain(ct, ctx.pt(self.t_vec, ct.level)))
        h0, h1 = self.z2c(tct)
        comb = self._pack(h0, h1)
        a = self.lt_s2c.apply(comb)
        a = truncate(ctx, a)
        a = mod_raise(ctx, a)
        v0 = self.lt_c2s0.apply(a)
        v1 = self.lt_c2s1.apply(a)
        e0 = self.boot.small_part(ctx.add(v0, ctx.conjugate(v0)))
        e1 = self.boot.small_part(ctx.add(v1, ctx.conjugate(v1)))
        te = self.c2z(e0, e1)
        corr = ctx.rescale(ctx.mul_plain(te, ctx.pt(self.tinv_vec, te.level)))
        return ctx.sub(ct, corr)

    def refresh(self, ct):
        return self.refresh_noise(self.refresh_overflow(ct))

    def _tables(self):
        b, c, p = self.b, self.c, self.p
        idt = [0.0] * p
        digit = [[0.0] * p for _ in range(c)]
        bits = [[[0.0] * p for _ in range(self.rbits)] for _ in range(c)]
        for i in range(c):
            for v in range(b ** (i + 1)):
                t = (-v * b ** (c - i - 1)) % p
                D = v // (b ** i)
                digit[i][t] = float(D)
                for l in range(self.rbits):
                    bits[i][l][t] = float((D >> l) & 1)
                if i == c - 1:
                    idt[t] = -v / float(p)
        return idt, digit, bits

    def a2b(self, ct, cutoff_bits=18):
        ctx = self.ctx
        c, B, d = self.c, self.B, self.d
        idt, digit, bits = self._tables()
        cores = list(self.z2c(ct, special=True))
        planes = [[None, None] for _ in range(self.rbits)]
        rounds = d // c
        w = self.c * self.rbits
        nu = int(np.ceil(cutoff_bits / max(1, w))) + 1
        window = []
        for j in range(rounds):
            half = 0 if j < rounds // 2 else 1
            K = (j * c) % B
            core = cores[half]
            cur = core if K == 0 else ctx.rotate(core, K)
            cur = ctx.rescale(ctx.mul_plain(cur, ctx.pt(self.mask(range(c)), cur.level)))
            for l, sp in enumerate(window):
                term = sp if l == 0 else ctx.mul_const(sp, float(self.b) ** (-c * l))
                cur = ctx.sub(cur, term)
            tabs = [idt] + [bits[i][l] for i in range(c) for l in range(self.rbits)]
            outs = self.boot.clear_and_lut(cur, tabs, self.p)
            y = outs[0]
            idx = 1
            for i in range(c):
                for l in range(self.rbits):
                    o = outs[idx]
                    idx += 1
                    o = ctx.rescale(ctx.mul_plain(o, ctx.pt(self.mask([i]), o.level)))
                    o = o if K == 0 else ctx.rotate(o, -K)
                    tgt = planes[l][half]
                    planes[l][half] = o if tgt is None else ctx.add(tgt, o)
            if j + 1 < rounds:
                spread = None
                for i in range(c):
                    part = ctx.rotate(y, (c - 1 - i)) if (c - 1 - i) else y
                    part = ctx.rescale(ctx.mul_plain(
                        part, ctx.pt(self.mask([i]) * (float(self.b) ** (-(i + 1))), part.level)))
                    spread = part if spread is None else ctx.add(spread, part)
                window.insert(0, spread)
                if len(window) > nu:
                    window.pop()
        return [p for pair in planes for p in pair]

    def b2a(self, planes):
        ctx = self.ctx
        R, B, W = self.R, self.B, self.W
        acc = None
        for l in range(self.rbits):
            for h in range(2):
                idxs = [h * B + q for q in range(B)]
                Mx = np.zeros((B, B), dtype=np.complex128)
                col = np.zeros((self.d, B), dtype=np.float64)
                for q in range(B):
                    col[h * B + q, q] = float(1 << l)
                blk = R.Uc @ (R.Ttinv @ col)
                lt = LinTransform(ctx, block_matrix(blk, W), name=f"B2A{l}{h}")
                t = lt.apply(planes[2 * l + h])
                acc = t if acc is None else ctx.add(acc, t)
        return acc

    def b2b(self, plane):
        self.ctx.boot_count += 1
        return self.boot.clear_and_lut(plane, [[0.0, 1.0]], 2)[0]

    def bool_and(self, a, b):
        return self.ctx.mul(a, b)

    def bool_xor(self, a, b):
        ctx = self.ctx
        s = ctx.add(a, b)
        return ctx.sub(s, ctx.mul_int(ctx.mul(a, b), 2))

    def bool_not(self, a):
        return self.ctx.sub(self.ctx.add_const(self.ctx.mul_const(a, 0.0), 1.0), a)
