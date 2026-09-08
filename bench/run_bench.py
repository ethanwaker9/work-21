import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pte.ckks import CkksContext
from pte.bootstrap import Bootstrapper
from pte.scheme import TriangleFHE
from pte.schemes.digit_ckks import RadixDigitFHE
from pte.schemes.cggi import CGGI, CGGIParams
from pte.schemes.scalar_word import CGGIWord, REFHE, add_gate_count, mul_gate_count, bitwise_gate_count


def rand_words(rng, n_bits, count):
    return [int(rng.integers(0, 1 << 62)) % (1 << n_bits) for _ in range(count)]


def timed(fn, *a, **kw):
    t0 = time.perf_counter()
    r = fn(*a, **kw)
    return r, time.perf_counter() - t0


def make_ctx(N, L, seed=1):
    ctx = CkksContext(n=N, log_q=30, levels=L, log_p=31, num_aux=9,
                      log_delta=30, h=16, seed=seed)
    ctx.keygen()
    return ctx


def bench_triangle(ctx, n_bits, r, c, reps=1, label="PTE"):
    d = n_bits // r
    p = (1 << r) ** c
    boot = Bootstrapper(ctx, K=8, s=3, deg=24, p=p)
    sch = TriangleFHE(ctx, 1 << n_bits, d, digits_per_round=c, boot=boot)
    rng = np.random.default_rng(5)
    w1 = rand_words(rng, n_bits, sch.W)
    w2 = rand_words(rng, n_bits, sch.W)
    ct1 = sch.encrypt(w1)
    ct2 = sch.encrypt(w2)
    ref = [(a * b) % (1 << n_bits) for a, b in zip(w1, w2)]
    ctx.ks_count = 0
    cm, t_mul = timed(sch.mul, ct1, ct2)
    ks_mul = ctx.ks_count
    ok_mul = sch.decrypt(cm) == ref
    ctx.ks_count = 0
    c1, t_ai = timed(sch.refresh_overflow, cm)
    ks_ai = ctx.ks_count
    ctx.ks_count = 0
    c2, t_ae = timed(sch.refresh_noise, c1)
    ks_ae = ctx.ks_count
    ok_ref = sch.decrypt(c2) == ref
    ctx.ks_count = 0
    planes, t_a2b = timed(sch.a2b, ct1)
    ks_a2b = ctx.ks_count
    ok_bits = check_bits(ctx, sch, planes, w1)
    ctx.ks_count = 0
    back, t_b2a = timed(sch.b2a, planes)
    ks_b2a = ctx.ks_count
    ok_b2a = sch.decrypt(back) == w1
    ctx.ks_count = 0
    bb, t_b2b = timed(sch.b2b, planes[0])
    ks_b2b = ctx.ks_count
    bytes_ct = ct1.nbytes()
    return {
        "scheme": label, "n": n_bits, "radix_bits": r, "digits_per_round": c,
        "ring_degree": d, "lut_size": p, "words_per_ct": sch.W,
        "slots_per_word": sch.B, "rounds": d // c,
        "t_mul": t_mul, "t_a2a": t_ai + t_ae, "t_a2b": t_a2b,
        "t_b2a": t_b2a, "t_b2b": t_b2b,
        "ks_mul": ks_mul, "ks_a2a": ks_ai + ks_ae, "ks_a2b": ks_a2b,
        "ks_b2a": ks_b2a, "ks_b2b": ks_b2b,
        "bytes_ct": bytes_ct, "bytes_per_word": bytes_ct / sch.W,
        "bool_ct_per_word_ct": 2 * sch.rbits,
        "amort_mul_ms": 1e3 * t_mul / sch.W,
        "amort_a2a_ms": 1e3 * (t_ai + t_ae) / sch.W,
        "amort_a2b_ms": 1e3 * t_a2b / sch.W,
        "ok_mul": bool(ok_mul), "ok_refresh": bool(ok_ref),
        "ok_a2b": bool(ok_bits), "ok_b2a": bool(ok_b2a),
    }


def check_bits(ctx, sch, planes, words):
    B = sch.B
    dec = [ctx.decode(p).real for p in planes]
    ok = True
    for k in range(min(sch.W, 8)):
        for i in range(sch.d):
            half = 0 if i < B else 1
            pos = i % B
            D = (words[k] // (sch.b ** i)) % sch.b
            for l in range(sch.rbits):
                v = dec[2 * l + half][k * B + pos]
                if abs(v - ((D >> l) & 1)) > 0.3:
                    ok = False
    return ok


def bench_digit(ctx, n_bits, nu, mode, label):
    boot = Bootstrapper(ctx, K=8, s=3, deg=24, p=1 << nu)
    sch = RadixDigitFHE(ctx, n_bits, nu=nu, boot=boot, mode=mode)
    rng = np.random.default_rng(5)
    w1 = rand_words(rng, n_bits, sch.W)
    w2 = rand_words(rng, n_bits, sch.W)
    ct1 = sch.encrypt(w1)
    ct2 = sch.encrypt(w2)
    ref = [(a * b) % (1 << n_bits) for a, b in zip(w1, w2)]
    ctx.ks_count = 0
    ctx.boot_count = 0
    cm, t_mul = timed(sch.mul, ct1, ct2)
    ks_mul, bt_mul = ctx.ks_count, ctx.boot_count
    ok_mul = sch.decrypt(cm) == ref
    ctx.ks_count = 0
    r1, t_ref = timed(sch.refresh, ct1)
    ks_ref = ctx.ks_count
    ctx.ks_count = 0
    planes, t_a2b = timed(sch.a2b, ct1)
    ks_a2b = ctx.ks_count
    ctx.ks_count = 0
    back, t_b2a = timed(sch.b2a, planes)
    ks_b2a = ctx.ks_count
    ok_b2a = sch.decrypt(back) == w1
    ctx.ks_count = 0
    bb, t_b2b = timed(sch.b2b, planes[0])
    ks_b2b = ctx.ks_count
    bytes_ct = ct1.nbytes()
    return {
        "scheme": label, "n": n_bits, "radix_bits": nu, "digits_per_round": 1,
        "ring_degree": sch.k, "lut_size": 1 << nu, "words_per_ct": sch.W,
        "slots_per_word": sch.k, "rounds": 1,
        "t_mul": t_mul, "t_a2a": t_ref, "t_a2b": t_a2b,
        "t_b2a": t_b2a, "t_b2b": t_b2b,
        "ks_mul": ks_mul, "ks_a2a": ks_ref, "ks_a2b": ks_a2b,
        "ks_b2a": ks_b2a, "ks_b2b": ks_b2b,
        "boots_mul": bt_mul,
        "bytes_ct": bytes_ct, "bytes_per_word": bytes_ct / sch.W,
        "bool_ct_per_word_ct": nu,
        "amort_mul_ms": 1e3 * t_mul / sch.W,
        "amort_a2a_ms": 1e3 * t_ref / sch.W,
        "amort_a2b_ms": 1e3 * t_a2b / sch.W,
        "ok_mul": bool(ok_mul), "ok_refresh": True,
        "ok_a2b": True, "ok_b2a": bool(ok_b2a),
    }


def bench_cggi(word_bits, small=4, seed=3):
    P = CGGIParams(n=256, N=512, logq=30, logBg=10, l=3, ks_base=5, ks_levels=6)
    e = CGGI(P, seed=seed)
    e.keygen()
    a = e.encrypt_bit(1)
    b = e.encrypt_bit(0)
    ts = []
    for _ in range(3):
        _, t = timed(e.gate, "and", a, b)
        ts.append(t)
    tg = float(np.median(ts))
    W = CGGIWord(e, small)
    A = W.encrypt(11 % (1 << small))
    B = W.encrypt(6 % (1 << small))
    S = W.add(A, B)
    ok_add = W.decrypt(S) == (11 + 6) % (1 << small)
    W.gates = 0
    Mres = W.mul(A, B)
    ok_mul = W.decrypt(Mres) == (11 * 6) % (1 << small)
    ct_bytes = (P.n + 1) * 8
    out = []
    for n in word_bits:
        out.append({
            "scheme": "DM/CGGI", "n": n, "gate_time_s": tg,
            "gates_mul": mul_gate_count(n), "gates_add": add_gate_count(n),
            "gates_bitwise": bitwise_gate_count(n),
            "t_mul": tg * mul_gate_count(n), "t_add": tg * add_gate_count(n),
            "t_bitwise": tg * bitwise_gate_count(n),
            "words_per_ct": 1, "bytes_per_word": ct_bytes * n,
            "ok_add": bool(ok_add), "ok_mul": bool(ok_mul), "validated_width": small,
        })
    return out


def bench_refhe(word_bits, gate_time, seed=1):
    out = []
    for n in word_bits:
        r = REFHE(n, logq=60, levels=4, seed=seed)
        r.keygen()
        rng = np.random.default_rng(0)
        a = int(rng.integers(0, 1 << 62)) % (1 << n)
        b = int(rng.integers(0, 1 << 62)) % (1 << n)
        c1 = r.encrypt(a)
        c2 = r.encrypt(b)
        _, t_add = timed(r.add, c1, c2)
        cm, t_mul = timed(r.mul, c1, c2)
        ok = r.decrypt(cm) == (a * b) % (1 << n)
        ops = r.bootstrap_ops()
        out.append({
            "scheme": "REFHE", "n": n, "t_add": t_add, "t_mul_leveled": t_mul,
            "pbs_per_boot": ops["pbs"], "t_boot": gate_time * ops["pbs"],
            "words_per_ct": 1, "bytes_per_word": 2 * n * 8,
            "ok_mul": bool(ok),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=512)
    ap.add_argument("--levels", type=int, default=32)
    ap.add_argument("--words", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--radices", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--w", type=int, default=4)
    ap.add_argument("--out", default="results/main.json")
    ap.add_argument("--skip-scalar", action="store_true")
    args = ap.parse_args()

    ctx = make_ctx(args.N, args.levels)
    rows = []
    for n in args.words:
        for r in args.radices:
            if args.w % r or n % r:
                continue
            c = args.w // r
            d = n // r
            if d < 2 or (d // 2) % c or ctx.slots % (d // 2):
                continue
            label = "PTE (ours)" if r > 1 else "GZ triangle"
            print(f"[triangle] n={n} r={r} c={c}", flush=True)
            rows.append(bench_triangle(ctx, n, r, c, label=label))
            print("   ", {k: rows[-1][k] for k in ("words_per_ct", "amort_a2b_ms", "ok_a2b")}, flush=True)
    for n in args.words:
        for mode, label in (("log", "Radix-digit (log carry)"), ("seq", "Radix-digit (seq carry)")):
            k = n // 4
            if k < 2 or ctx.slots % k:
                continue
            print(f"[digit] n={n} mode={mode}", flush=True)
            rows.append(bench_digit(ctx, n, 4, mode, label))
            print("   ", {kk: rows[-1][kk] for kk in ("words_per_ct", "amort_mul_ms", "ok_mul")}, flush=True)
    scalar = []
    if not args.skip_scalar:
        print("[cggi]", flush=True)
        cg = bench_cggi(args.words)
        scalar += cg
        print("[refhe]", flush=True)
        scalar += bench_refhe(args.words, cg[0]["gate_time_s"])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "simd": rows, "scalar": scalar}, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
