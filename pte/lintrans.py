import numpy as np


def block_diag(block, reps):
    s = block.shape[0]
    S = s * reps
    out = np.zeros((S, S), dtype=np.complex128)
    for b in range(reps):
        out[b * s:(b + 1) * s, b * s:(b + 1) * s] = block
    return out


def matrix_diagonals(mat, tol=1e-11):
    S = mat.shape[0]
    idx = np.arange(S)
    diags = {}
    for k in range(S):
        d = mat[idx, (idx + k) % S]
        if np.max(np.abs(d)) > tol:
            diags[k] = d.copy()
    return diags


class LinTransform:
    def __init__(self, ctx, mat, scale=None, tol=1e-11, name=""):
        self.ctx = ctx
        self.S = mat.shape[0]
        self.diags = matrix_diagonals(mat, tol)
        self.keys = sorted(self.diags.keys())
        self.name = name
        m = max(1, len(self.keys))
        self.n1 = max(1, int(round(np.sqrt(m))))
        self._pt = {}
        self._lvl = None

    def num_rotations(self):
        rs = set()
        for k in self.keys:
            rs.add(k % self.n1)
            rs.add(k - (k % self.n1))
        rs.discard(0)
        return len(rs)

    def num_plain_mults(self):
        return len(self.keys)

    def prepare(self, level):
        if self._lvl == level:
            return
        self._pt = {}
        for k in self.keys:
            g = k - (k % self.n1)
            v = np.roll(self.diags[k], g % self.S)
            self._pt[k] = self.ctx.pt(v, level)
        self._lvl = level

    def apply(self, ct):
        ctx = self.ctx
        if not self.keys:
            raise ValueError("empty transform")
        self.prepare(ct.level)
        babies = {}
        for k in self.keys:
            j = k % self.n1
            if j not in babies:
                babies[j] = ct if j == 0 else ctx.rotate(ct, j)
        groups = {}
        for k in self.keys:
            groups.setdefault(k - (k % self.n1), []).append(k)
        acc = None
        for g in sorted(groups):
            part = None
            for k in groups[g]:
                t = ctx.mul_plain(babies[k % self.n1], self._pt[k])
                part = t if part is None else ctx.add(part, t)
            part = ctx.rescale(part)
            if g:
                part = ctx.rotate(part, g)
            acc = part if acc is None else ctx.add(acc, part)
        return acc


def eval_matrix(ctx, lo, hi):
    e = ctx.enc
    return e.zeta ** np.outer(e.rot, np.arange(lo, hi))


def slot_to_coeff_matrix(ctx):
    return eval_matrix(ctx, 0, ctx.slots)


def coeff_to_slot_matrices(ctx):
    n = ctx.n
    S = ctx.slots
    E0 = eval_matrix(ctx, 0, S)
    E1 = eval_matrix(ctx, S, n)
    return np.conj(E0).T / n, np.conj(E1).T / n
