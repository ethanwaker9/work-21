import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pte.rns import NTT, find_primes
from pte.ckks import CkksContext, Encoder
from pte.lintrans import LinTransform, slot_to_coeff_matrix, coeff_to_slot_matrices
from pte.bootstrap import Bootstrapper, make_lut_coeffs
from pte.aring import ARing, poly_eval_int
from pte.scheme import TriangleFHE
from pte.schemes.digit_ckks import RadixDigitFHE
from pte.schemes.cggi import CGGI, CGGIParams
from pte.schemes.scalar_word import CGGIWord, REFHE, add_gate_count, mul_gate_count
from pte.batching import native_slot_capacity, capacity_upper_bound, two_slot_instance
from pte.bounds import clearing_degree, carry_multilinear_degree


def small_ctx(levels=32, n=256, seed=7):
    ctx = CkksContext(n=n, log_q=30, levels=levels, log_p=31, num_aux=9,
                      log_delta=30, h=16, seed=seed)
    ctx.keygen()
    return ctx


def test_ntt_roundtrip_and_negacyclic():
    n = 32
    q = find_primes(1, 30, 2 * n)[0]
    T = NTT(n, q)
    rng = np.random.default_rng(0)
    a = rng.integers(0, q, size=n).astype(np.int64)
    assert np.array_equal(T.inverse(T.forward(a)), a)
    b = rng.integers(0, q, size=n).astype(np.int64)
    c = T.inverse((T.forward(a) * T.forward(b)) % q)
    ref = np.zeros(n, dtype=object)
    for i in range(n):
        for j in range(n):
            k = i + j
            if k < n:
                ref[k] += int(a[i]) * int(b[j])
            else:
                ref[k - n] -= int(a[i]) * int(b[j])
    assert np.array_equal(c, np.array([int(x) % q for x in ref]))


def test_encoder_roundtrip():
    E = Encoder(64)
    rng = np.random.default_rng(1)
    z = rng.random(32) + 1j * rng.random(32)
    assert np.max(np.abs(E.decode_poly(E.encode_poly(z)) - z)) < 1e-10


def test_ckks_operations():
    ctx = small_ctx(levels=8, n=256)
    rng = np.random.default_rng(2)
    z = rng.random(ctx.slots) - 0.5
    w = rng.random(ctx.slots) - 0.5
    a = ctx.encrypt(ctx.encode(z))
    b = ctx.encrypt(ctx.encode(w))
    assert np.max(np.abs(ctx.decode(a).real - z)) < 1e-4
    assert np.max(np.abs(ctx.decode(ctx.mul(a, b)).real - z * w)) < 1e-4
    assert np.max(np.abs(ctx.decode(ctx.rotate(a, 3)).real - np.roll(z, -3))) < 1e-4
    assert np.max(np.abs(ctx.decode(ctx.conjugate(a)).real - z)) < 1e-4


def test_linear_transforms():
    ctx = small_ctx(levels=8, n=128)
    S = ctx.slots
    rng = np.random.default_rng(3)
    w0 = rng.random(S) - 0.5
    w1 = rng.random(S) - 0.5
    ct = ctx.encrypt(ctx.encode(w0 + 1j * w1))
    res = LinTransform(ctx, slot_to_coeff_matrix(ctx)).apply(ct)
    coe = ctx.decode_coeff(res)
    assert np.max(np.abs(coe[:S] - w0)) < 1e-4
    assert np.max(np.abs(coe[S:] - w1)) < 1e-4
    A0, A1 = coeff_to_slot_matrices(ctx)
    u0 = LinTransform(ctx, A0).apply(res)
    u0 = ctx.add(u0, ctx.conjugate(u0))
    assert np.max(np.abs(ctx.decode(u0).real - w0)) < 1e-3


@pytest.mark.parametrize("order", [0, 1, 2])
def test_lut_interpolation_order(order):
    p = 16
    tab = [j / p for j in range(p)]
    c = make_lut_coeffs(p, tab, order)
    eps = 1e-4
    err = 0.0
    for j in range(p):
        u = np.exp(2j * np.pi * (j / p + eps))
        err = max(err, abs(sum(c[k] * u ** k for k in range(len(c))) - tab[j]))
    assert err < 16.0 * (eps ** (order + 1)) * (p ** (order + 2))


def test_bootstrap_lut():
    ctx = small_ctx(levels=32, n=128)
    bs = Bootstrapper(ctx, K=8, s=3, deg=24, p=16)
    rng = np.random.default_rng(4)
    jj = rng.integers(0, 16, size=ctx.slots)
    ii = rng.integers(-8, 9, size=ctx.slots)
    ct = ctx.encrypt(ctx.encode((ii + jj / 16).astype(np.complex128)))
    outs = bs.clear_and_lut(ct, [[j / 16 for j in range(16)], [float(j) for j in range(16)]], 16)
    assert np.max(np.abs(ctx.decode(outs[0]).real - jj / 16)) < 1e-3
    assert np.max(np.abs(ctx.decode(outs[1]).real - jj)) < 0.2


@pytest.mark.parametrize("n_bits,r", [(8, 1), (8, 2), (16, 2), (16, 4), (32, 4), (64, 8)])
def test_triangle_encoding_plain(n_bits, r):
    R = ARing(1 << n_bits, n_bits // r)
    rng = np.random.default_rng(5)
    for _ in range(20):
        m1 = int(rng.integers(0, 1 << 60)) % R.M
        m2 = int(rng.integers(0, 1 << 60)) % R.M
        g = R.triangle(m1)
        assert np.max(np.abs(g)) < 1.0
        assert R.decode_triangle(g) == m1
        h = R.mul_t(R.mul(g, R.triangle(m2)))
        assert R.decode_triangle(h) == (m1 * m2) % R.M


@pytest.mark.parametrize("M,d", [(1000003, 8), (65537, 4), (255, 8), (4294967291, 8)])
def test_triangle_encoding_general_modulus(M, d):
    R = ARing(M, d)
    assert poly_eval_int(R.F, R.b) == M
    rng = np.random.default_rng(6)
    for _ in range(10):
        m1 = int(rng.integers(0, 1 << 60)) % M
        m2 = int(rng.integers(0, 1 << 60)) % M
        assert R.decode_triangle(R.triangle(m1)) == m1
        assert R.decode_triangle(R.mul_t(R.mul(R.triangle(m1), R.triangle(m2)))) == (m1 * m2) % M


def test_scheme_arithmetic_and_refresh():
    ctx = small_ctx(levels=32, n=256)
    n_bits, r = 8, 2
    sch = TriangleFHE(ctx, 1 << n_bits, n_bits // r, digits_per_round=1,
                      boot=Bootstrapper(ctx, K=8, s=3, deg=24, p=1 << r))
    rng = np.random.default_rng(7)
    w1 = [int(x) for x in rng.integers(0, 1 << n_bits, size=sch.W)]
    w2 = [int(x) for x in rng.integers(0, 1 << n_bits, size=sch.W)]
    ct1, ct2 = sch.encrypt(w1), sch.encrypt(w2)
    assert sch.decrypt(ct1) == w1
    assert sch.decrypt(sch.add(ct1, ct2)) == [(a + b) % (1 << n_bits) for a, b in zip(w1, w2)]
    ref = [(a * b) % (1 << n_bits) for a, b in zip(w1, w2)]
    cm = sch.mul(ct1, ct2)
    assert sch.decrypt(cm) == ref
    assert sch.decrypt(sch.refresh(cm)) == ref


@pytest.mark.parametrize("r,c", [(1, 2), (2, 1), (4, 1)])
def test_scheme_conversions(r, c):
    ctx = small_ctx(levels=32, n=256)
    n_bits = 8
    p = (1 << r) ** c
    sch = TriangleFHE(ctx, 1 << n_bits, n_bits // r, digits_per_round=c,
                      boot=Bootstrapper(ctx, K=8, s=3, deg=24, p=p))
    rng = np.random.default_rng(8)
    w = [int(x) for x in rng.integers(0, 1 << n_bits, size=sch.W)]
    planes = sch.a2b(sch.encrypt(w))
    B = sch.B
    dec = [ctx.decode(pl).real for pl in planes]
    for k in range(min(sch.W, 6)):
        for i in range(sch.d):
            half = 0 if i < B else 1
            D = (w[k] // (sch.b ** i)) % sch.b
            for l in range(sch.rbits):
                assert abs(dec[2 * l + half][k * B + i % B] - ((D >> l) & 1)) < 0.3
    assert sch.decrypt(sch.b2a(planes)) == w


@pytest.mark.parametrize("mode", ["log", "seq"])
def test_digit_scheme(mode):
    ctx = small_ctx(levels=32, n=256)
    n_bits = 8
    sch = RadixDigitFHE(ctx, n_bits, nu=4, boot=Bootstrapper(ctx, K=8, s=3, deg=24, p=16),
                        mode=mode)
    rng = np.random.default_rng(9)
    w1 = [int(x) for x in rng.integers(0, 1 << n_bits, size=sch.W)]
    w2 = [int(x) for x in rng.integers(0, 1 << n_bits, size=sch.W)]
    ct1, ct2 = sch.encrypt(w1), sch.encrypt(w2)
    assert sch.decrypt(ct1) == w1
    assert sch.decrypt(sch.mul(ct1, ct2)) == [(a * b) % (1 << n_bits) for a, b in zip(w1, w2)]


def test_gate_scheme_and_word_circuits():
    e = CGGI(CGGIParams(n=128, N=256, logq=30, logBg=10, l=3, ks_base=5, ks_levels=6), seed=3)
    e.keygen()
    for x in (0, 1):
        for y in (0, 1):
            a, b = e.encrypt_bit(x), e.encrypt_bit(y)
            assert e.decrypt_bit(e.gate("and", a, b)) == (x & y)
            assert e.decrypt_bit(e.gate("xor", a, b)) == (x ^ y)
    W = CGGIWord(e, 4)
    A, B = W.encrypt(11), W.encrypt(6)
    W.gates = 0
    assert W.decrypt(W.add(A, B)) == (11 + 6) % 16
    assert W.gates == add_gate_count(4)
    W.gates = 0
    assert W.decrypt(W.mul(A, B)) == (11 * 6) % 16
    assert W.gates == mul_gate_count(4)


def test_scalar_word_scheme():
    r = REFHE(8, logq=60, levels=4, seed=1)
    r.keygen()
    rng = np.random.default_rng(11)
    a, b = int(rng.integers(0, 256)), int(rng.integers(0, 256))
    c1, c2 = r.encrypt(a), r.encrypt(b)
    assert r.decrypt(c1) == a
    assert r.decrypt(r.add(c1, c2)) == (a + b) % 256
    assert r.decrypt(r.mul(c1, c2)) == (a * b) % 256


@pytest.mark.parametrize("n,d", [(8, 8), (16, 8), (32, 16), (64, 16)])
def test_native_slot_capacity(n, d):
    M = 1 << n
    R = ARing(M, d)
    cap = native_slot_capacity(R.F, M)
    assert cap <= capacity_upper_bound(M, d)
    assert capacity_upper_bound(M, d) == 2
    inst = two_slot_instance(M, d)
    assert inst["valid"]
    assert inst["capacity"] == 2


@pytest.mark.parametrize("g", [2, 3, 4, 5, 6, 8])
def test_clearing_degree(g):
    d2 = clearing_degree(g, 2)
    d3 = clearing_degree(g, 3)
    assert d2 >= g + 1
    assert d3 >= g + 1
    assert abs(d2 - (2 * g - 1)) <= 2
    assert abs(d3 - (3 * g - 1)) <= 4


@pytest.mark.parametrize("s", [1, 2, 3, 4, 5])
def test_carry_degree(s):
    assert carry_multilinear_degree(s) == 2 * s
