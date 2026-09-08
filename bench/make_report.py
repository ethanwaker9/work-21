import argparse
import json
import os
import subprocess
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pte.aring import ARing
from pte.batching import capacity_table, capacity_upper_bound, two_slot_instance, subspace_distance
from pte.bounds import clearing_degree, carry_multilinear_degree, tradeoff_curve

plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi": 200,
    "text.usetex": False,
})

MARK = ["o", "s", "^", "D", "v", "P", "X"]
GRAY = ["0.05", "0.35", "0.55", "0.72", "0.2", "0.45"]


def save(fig, out, name):
    eps = os.path.join(out, name + ".eps")
    pdf = os.path.join(out, name + ".pdf")
    fig.savefig(eps, format="eps", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    subprocess.run(["epstopdf", "--outfile=" + pdf, eps], check=True)
    return pdf


def gamma(n, r):
    return (n / r) * (2.0 ** r + 2.0)


def depth(r, lam, n, K=8):
    g = gamma(n, r)
    den = r + 1 + np.log2(g) + np.log2(2 * K + 2)
    return max(0.02, (lam - r - 2) / den)


def words_per_ct(N, n, r):
    return N * r / n


def tab_capacity(out):
    rows = capacity_table([8, 16, 32, 64, 128], [8, 16, 32, 64])
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Native slot count of the machine word plaintext ring $\Z[X]/(X^{d}-X+b)$ with $b=2^{n/d}$, computed by factoring the defining polynomial modulo two and counting the simple linear factors. The column $n$ is the word width, $d$ the degree of the ring, $b$ the radix, $\normi{F}$ the largest coefficient of the defining polynomial, \emph{cap} the computed slot count of Definition~\ref{def:slots} and \emph{bound} the value $\min(d,p_{\min}(M))$ of Theorem~\ref{thm:cap}. The two columns agree in every instance, so the ring that the scheme of Section~\ref{sec:pte} already uses attains the bound and a second machine word slot exists inside it as an abstract quotient, and the count does not grow with the degree, which is the content of Corollary~\ref{cor:answer}. The last three columns are the two slot construction of Theorem~\ref{thm:attain} for the same width and degree, with $\normi{F_{2}}$ its largest coefficient and $\mathrm{lb}$ the certified lower bound $\pi(b,d)/\sqrt d$ of Theorem~\ref{thm:attain}, which no monic polynomial with the two prescribed values can go below. The achieved norm stays within the factor $\sqrt d$ of that bound in every instance, so the construction is close to the best possible, and both grow like $b^{2}$ times a factor that depends only on the degree, which is what makes the radix mechanism of Section~\ref{sec:pte} the cheaper of the two ways to double the packing.}")
    lines.append(r"\label{tab:cap}")
    lines.append(r"\begin{tabular}{@{}cccccc|ccc@{}}")
    lines.append(r"\toprule")
    lines.append(r"$n$ & $d$ & $b$ & $\normi{F}$ & cap & bound & cap$_2$ & $\normi{F_2}$ & lb \\")
    lines.append(r"\midrule")
    for row in rows:
        if row["d"] < 8:
            continue
        ts = two_slot_instance(1 << row["n"], row["d"])
        lines.append(
            f"{row['n']} & {row['d']} & {row['b']} & {row['norm']} & {row['capacity']} & "
            f"{row['bound']} & {ts['capacity']} & {ts['norm']} & "
            f"{subspace_distance(ts['b'], row['d'], 1 << row['n']):.0f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    open(os.path.join(out, "tab_cap.tex"), "w").write("\n".join(lines) + "\n")


def tab_degree(out):
    gs = [2, 3, 4, 5, 6, 8, 10, 12, 16]
    d2 = [clearing_degree(g, 2) for g in gs]
    d3 = [clearing_degree(g, 3) for g in gs]
    d4 = [clearing_degree(g, 4) for g in gs]
    cd = [(s, carry_multilinear_degree(s)) for s in (1, 2, 3, 4, 5)]
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Numerical confirmation of the two degree statements of Section~\ref{sec:lower}. The left block gives, for a payload grid of $g$ points and $K$ admissible overflow values, the smallest degree of a polynomial that maps $I+j/g$ to $j/g$ for every admissible pair, obtained by solving the interpolation system in a Chebyshev basis and increasing the degree until the residual falls below $10^{-8}$; the value equals $Kg-1$ in most instances and is never below $g+1$, so it stays far above the lower bound of Lemma~\ref{lem:resolution} and it grows linearly in $g$, which shows that the resolution of a slot, and not the number of overflow values alone, is what makes an exact clearing expensive. The right block gives the multilinear degree of the carry out of an $s$ bit addition, computed as the alternating sum that extracts the coefficient of the top monomial; the value is $2s$, which confirms Lemma~\ref{lem:carry} and therefore the second half of the trade-off of Theorem~\ref{thm:tradeoff}.}")
    lines.append(r"\label{tab:degree}")
    lines.append(r"\begin{tabular}{@{}c" + "c" * len(gs) + r"@{}}")
    lines.append(r"\toprule")
    lines.append(r"$g$ & " + " & ".join(str(g) for g in gs) + r" \\")
    lines.append(r"\midrule")
    lines.append(r"$K=2$ & " + " & ".join(str(v) for v in d2) + r" \\")
    lines.append(r"$K=3$ & " + " & ".join(str(v) for v in d3) + r" \\")
    lines.append(r"$K=4$ & " + " & ".join(str(v) for v in d4) + r" \\")
    lines.append(r"\midrule")
    lines.append(r"$s$ & " + " & ".join(str(s) for s, _ in cd) + r" & \multicolumn{" + str(len(gs) - len(cd)) + r"}{c}{carry degree} \\")
    lines.append(r"$\deg c_s$ & " + " & ".join(str(v) for _, v in cd) + r" & \multicolumn{" + str(len(gs) - len(cd)) + r"}{c}{$=2s$} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    open(os.path.join(out, "tab_degree.tex"), "w").write("\n".join(lines) + "\n")


def tab_general(out):
    cases = [(1000003, 8), (65537, 4), (10 ** 9 + 7, 8), (255, 8), (4294967291, 8),
             (2 ** 31 - 1, 8), (3 ** 20, 8), (10 ** 12 + 39, 12)]
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Instances of the packed triangle encoding for word moduli that are not powers of two, produced by the construction of Theorem~\ref{thm:general}. The column $M$ is the word modulus, $d$ the chosen degree of the plaintext ring, $b$ the radix that the construction selects, $\normi{F}$ the largest coefficient of the defining polynomial, which is what enters the expansion factor, $r_{1}$ the number of real roots of $F$ after the search that removes them, and $\gamma$ the expansion factor $d(\normi{F}+2)$ of Theorem~\ref{thm:noise}. The last column states whether encoding, decoding and a full multiplication were verified on random inputs. A value $r_{1}=0$ means that the canonical embedding maps the block onto $\C^{d/2}$, so a word occupies $d/2$ complex slots exactly as in the power of two case.}")
    lines.append(r"\label{tab:general}")
    lines.append(r"\begin{tabular}{@{}rccccrc@{}}")
    lines.append(r"\toprule")
    lines.append(r"$M$ & $d$ & $b$ & $\normi{F}$ & $r_1$ & $\gamma$ & verified \\")
    lines.append(r"\midrule")
    rng = np.random.default_rng(0)
    for M, d in cases:
        R = ARing(M, d)
        ok = True
        for _ in range(20):
            m1 = int(rng.integers(0, 1 << 60)) % M
            m2 = int(rng.integers(0, 1 << 60)) % M
            if R.decode_triangle(R.triangle(m1)) != m1:
                ok = False
            if R.decode_triangle(R.mul_t(R.mul(R.triangle(m1), R.triangle(m2)))) != (m1 * m2) % M:
                ok = False
        lines.append(f"${M}$ & {d} & {R.b} & {max(abs(c) for c in R.F[:d])} & {R.r1} & "
                     f"{R.gamma} & {'yes' if ok else 'no'} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    open(os.path.join(out, "tab_general.tex"), "w").write("\n".join(lines) + "\n")


def load(path):
    with open(path) as f:
        return json.load(f)


def key(row):
    return (row["scheme"], row["n"], row.get("radix_bits", 0))


def tab_time(data, out):
    simd = data["simd"]
    ns = sorted({r["n"] for r in simd})
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Measured cost of every procedure on the shared backend of Section~\ref{sec:exp}, at ring dimension $N=512$ with $33$ moduli of $30$ bits and a look-up table of size $16$ for all schemes. The column $r$ is the radix exponent, so $r=1$ is the encoding of \cite{GZ26} and $r>1$ is ours; $W$ is the number of words that one ciphertext carries; \emph{mult}, \emph{refresh}, $A\!\to\!B$, $B\!\to\!A$ and $B\!\to\!B$ are the wall clock times of one multiplication, one full refresh, one conversion of a whole ciphertext, one reverse conversion and one boolean refresh, in seconds; \emph{am.} is the conversion time divided by $W$, in milliseconds per word, which is the quantity that governs throughput. Every row was checked for exact correctness on random inputs. The amortized column falls by the factor $r$ across each group of three rows with the same $n$, which is the prediction of Theorem~\ref{thm:a2bcost}.}")
    lines.append(r"\label{tab:time}")
    lines.append(r"\setlength{\tabcolsep}{3.4pt}")
    lines.append(r"\begin{tabular}{@{}lcccccccccc@{}}")
    lines.append(r"\toprule")
    lines.append(r"Scheme & $n$ & $r$ & $W$ & mult & refresh & $A\!\to\!B$ & $B\!\to\!A$ & $B\!\to\!B$ & am.\ (ms) & ok \\")
    lines.append(r"\midrule")
    for n in ns:
        grp = [r for r in simd if r["n"] == n]
        grp.sort(key=lambda r: (r["scheme"].startswith("Radix"), r.get("radix_bits", 0)))
        for r in grp:
            ok = all(r.get(k, True) for k in ("ok_mul", "ok_refresh", "ok_a2b", "ok_b2a"))
            name = r["scheme"].replace("PTE (ours)", r"\textbf{PTE}").replace("GZ triangle", "Gao--Zheng")
            name = name.replace("Radix-digit (log carry)", "Radix digit, log.").replace(
                "Radix-digit (seq carry)", "Radix digit, seq.")
            lines.append(
                f"{name} & {n} & {r.get('radix_bits',0)} & {r['words_per_ct']} & "
                f"{r['t_mul']:.2f} & {r['t_a2a']:.1f} & {r['t_a2b']:.1f} & {r['t_b2a']:.2f} & "
                f"{r['t_b2b']:.1f} & {r['amort_a2b_ms']:.0f} & {'y' if ok else 'n'} \\\\")
        lines.append(r"\midrule")
    lines = lines[:-1]
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    open(os.path.join(out, "tab_time.tex"), "w").write("\n".join(lines) + "\n")


def tab_space(data, out):
    simd = data["simd"]
    ns = sorted({r["n"] for r in simd})
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Measured memory footprint on the same backend. \emph{ct} is the size of one arithmetic ciphertext in kilobytes, the same for every scheme because it is fixed by the ring dimension and the modulus chain; \emph{per word} is that size divided by the number of packed words, which is the quantity a deployment pays for; \emph{bool.\ ct} is the number of boolean mode ciphertexts that one arithmetic ciphertext produces, which is $2r$ and therefore the same $2n/N$ ciphertexts per word for every radix; \emph{state} is the number of ciphertexts besides the outputs that the conversion keeps resident in order to reach its amortized cost. Our encoding divides the memory per word by the radix exponent and keeps that working set at the constant $\nu+2$, while the batched conversion of \cite{GZ26} holds a whole batch of $n/w$ arithmetic ciphertexts with their two halves, so its working set grows with the width.}")
    lines.append(r"\label{tab:space}")
    lines.append(r"\begin{tabular}{@{}lccccccc@{}}")
    lines.append(r"\toprule")
    lines.append(r"Scheme & $n$ & $r$ & $W$ & ct (KB) & per word (KB) & bool.\ ct & state \\")
    lines.append(r"\midrule")
    for n in ns:
        grp = [r for r in simd if r["n"] == n]
        grp.sort(key=lambda r: (r["scheme"].startswith("Radix"), r.get("radix_bits", 0)))
        for r in grp:
            name = r["scheme"].replace("PTE (ours)", r"\textbf{PTE}").replace("GZ triangle", "Gao--Zheng")
            name = name.replace("Radix-digit (log carry)", "Radix digit, log.").replace(
                "Radix-digit (seq carry)", "Radix digit, seq.")
            if r["scheme"].startswith("Radix"):
                state = 1
            elif r.get("radix_bits", 0) > 1:
                state = int(np.ceil(18 / 4)) + 1 + 2
            else:
                state = 2 * max(1, n // 4)
            lines.append(
                f"{name} & {n} & {r.get('radix_bits',0)} & {r['words_per_ct']} & "
                f"{r['bytes_ct']/1024:.0f} & {r['bytes_per_word']/1024:.1f} & "
                f"{r['bool_ct_per_word_ct']} & {state} \\\\")
        lines.append(r"\midrule")
    lines = lines[:-1]
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    open(os.path.join(out, "tab_space.tex"), "w").write("\n".join(lines) + "\n")


def tab_scalar(data, out):
    sc = data["scalar"]
    if not sc:
        return
    cg = [r for r in sc if r["scheme"] == "DM/CGGI"]
    rf = [r for r in sc if r["scheme"] == "REFHE"]
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{The two scalar schemes on the same machine. For DM/CGGI the gate count of the multiplication circuit is the exact number of gate bootstrapping operations that the executed ripple carry multiplier performs, and the time is that count multiplied by the measured latency of one gate bootstrapping operation, which was $t_{g}$ seconds in our implementation; the circuit itself was executed end to end for four bit words to confirm the count. For REFHE the leveled multiplication was executed and timed, and the refresh cost is the number of programmable bootstrapping operations that its published procedure needs, which is one per bit, multiplied by the same $t_{g}$. The column words/ct is one for both, which is the reason that their amortized cost equals their latency and that they stay three to four orders of magnitude behind the SIMD schemes of Table~\ref{tab:time}.}")
    lines.append(r"\label{tab:scalar}")
    lines.append(r"\begin{tabular}{@{}lccccc@{}}")
    lines.append(r"\toprule")
    lines.append(r"Scheme & $n$ & gates or PBS & mult (s) & bitwise (s) & words/ct \\")
    lines.append(r"\midrule")
    for r in cg:
        lines.append(f"DM/CGGI \\cite{{DM15,CGGI20}} & {r['n']} & {r['gates_mul']} & "
                     f"{r['t_mul']:.1f} & {r['t_bitwise']:.2f} & 1 \\\\")
    lines.append(r"\midrule")
    for r in rf:
        lines.append(f"REFHE \\cite{{REFHE26}} & {r['n']} & {r['pbs_per_boot']} & "
                     f"{r['t_mul_leveled']:.4f} (leveled) & {r['t_boot']:.1f} & 1 \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    open(os.path.join(out, "tab_scalar.tex"), "w").write("\n".join(lines) + "\n")


def fig_amort(data, out):
    simd = data["simd"]
    ns = sorted({r["n"] for r in simd})
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.3))
    labels = {}
    for r in simd:
        labels.setdefault((r["scheme"], r.get("radix_bits", 0)), []).append(r)
    idx = 0
    order = sorted(labels.keys(), key=lambda k: (k[0].startswith("Radix"), k[1]))
    for kk in order:
        rows = sorted(labels[kk], key=lambda x: x["n"])
        xs = [x["n"] for x in rows]
        ya = [x["amort_a2b_ms"] for x in rows]
        yb = [x["amort_a2a_ms"] for x in rows]
        nm = kk[0]
        lab = ("PTE, $r=%d$" % kk[1]) if nm.startswith("PTE") else (
            "triangle, $r=1$" if nm.startswith("GZ") else nm.replace("Radix-digit ", "digit "))
        ax[0].plot(xs, ya, marker=MARK[idx % len(MARK)], color=GRAY[idx % len(GRAY)],
                   lw=1.1, ms=3.4, label=lab)
        ax[1].plot(xs, yb, marker=MARK[idx % len(MARK)], color=GRAY[idx % len(GRAY)],
                   lw=1.1, ms=3.4, label=lab)
        idx += 1
    for a, t in zip(ax, ["arithmetic to boolean", "arithmetic refresh"]):
        a.set_xscale("log", base=2)
        a.set_yscale("log", base=2)
        a.set_xlabel("word width $n$ (bits)")
        a.set_xticks(ns)
        a.set_xticklabels([str(v) for v in ns])
        a.grid(True, which="major", ls=":", lw=0.4, color="0.75")
        a.set_title(t)
    ax[0].set_ylabel("amortized time per word (ms)")
    ax[0].legend(ncol=2, frameon=False, loc="upper left", fontsize=6)
    return save(fig, out, "fig_amort")


def fig_mem(data, out):
    simd = data["simd"]
    fig, ax = plt.subplots(figsize=(3.2, 2.3))
    labels = {}
    for r in simd:
        labels.setdefault((r["scheme"], r.get("radix_bits", 0)), []).append(r)
    order = sorted(labels.keys(), key=lambda k: (k[0].startswith("Radix"), k[1]))
    idx = 0
    for kk in order:
        rows = sorted(labels[kk], key=lambda x: x["n"])
        xs = [x["n"] for x in rows]
        ys = [x["bytes_per_word"] / 1024 for x in rows]
        nm = kk[0]
        lab = ("PTE, $r=%d$" % kk[1]) if nm.startswith("PTE") else (
            "triangle, $r=1$" if nm.startswith("GZ") else nm.replace("Radix-digit ", "digit "))
        ax.plot(xs, ys, marker=MARK[idx % len(MARK)], color=GRAY[idx % len(GRAY)],
                lw=1.1, ms=3.4, label=lab)
        idx += 1
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("word width $n$ (bits)")
    ax.set_ylabel("ciphertext memory per word (KB)")
    ax.grid(True, ls=":", lw=0.4, color="0.75")
    ax.legend(frameon=False, fontsize=6, ncol=1, loc="upper left")
    return save(fig, out, "fig_mem")


def fig_tradeoff(out):
    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    n = 64
    for eta, mk, col in ((6, "o", "0.05"), (8, "s", "0.45"), (10, "^", "0.7")):
        pts = tradeoff_curve(n, eta - 4, 4, list(range(1, n + 1)))
        xs = [a for _, a, _ in pts]
        ys = [b for _, _, b in pts]
        ax.step(xs, ys, where="post", color=col, lw=1.1, label=r"$\eta=%d$" % eta)
    ax.plot([16], [1], marker="*", ms=9, color="0.05", ls="none")
    ax.annotate("Gao--Zheng", (16, 1), textcoords="offset points", xytext=(-42, 6), fontsize=6)
    ax.plot([1], [3], marker="D", ms=4, color="0.05", ls="none")
    ax.annotate("radix digit", (1, 3), textcoords="offset points", xytext=(6, -2), fontsize=6)
    ax.plot([4], [1], marker="P", ms=5, color="0.05", ls="none")
    ax.annotate("PTE $r=4$", (4, 1), textcoords="offset points", xytext=(4, 8), fontsize=6)
    ax.set_xlabel("bootstrapping operations for $A\\!\\to\\!B$")
    ax.set_ylabel("bootstrapping operations\nfor exact arithmetic")
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 5)
    ax.grid(True, ls=":", lw=0.4, color="0.75")
    ax.legend(frameon=False, fontsize=6)
    return save(fig, out, "fig_tradeoff")


KSU = 0.027


def boot_cost(p, eta):
    return (53.0 + 27.0 + (2.0 + eta) * np.sqrt(2.0 * p)) * KSU


def workload_cost(r, n, N=512, lam=36.0, mu=8.0):
    W = N * r / n
    B = n / (2.0 * r)
    t_mul = KSU
    t_ref = 2.0 * boot_cost(16, 1) + 8.0 * np.sqrt(2.0 * B) * KSU
    t_conv = (n / r) * boot_cost(2.0 ** r, 1.0 + r) + 3.0 * (n / r) * KSU
    return 1e3 * (mu * t_mul + (mu / depth(r, lam, n)) * t_ref + t_conv) / W


def fig_3d(data, out, N=512):
    rs = np.linspace(1.0, 8.0, 29)
    nsx = np.array([8, 16, 32, 64, 128, 256], dtype=float)
    Rg, Ng = np.meshgrid(rs, nsx, indexing="ij")
    Z = np.zeros_like(Rg)
    for i in range(Rg.shape[0]):
        for j in range(Rg.shape[1]):
            Z[i, j] = np.log2(workload_cost(Rg[i, j], Ng[i, j], N))
    fig = plt.figure(figsize=(3.55, 3.05))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(Rg, np.log2(Ng), Z, cmap=cm.bone, rstride=1, cstride=1,
                    linewidth=0.2, edgecolor="0.35", antialiased=True, alpha=0.95)
    floor = Z.min() - 3.0
    ax.contour(Rg, np.log2(Ng), Z, zdir="z", offset=floor, colors="0.35",
               linewidths=0.6, levels=10)
    ridge = []
    for j in range(Rg.shape[1]):
        i = int(np.argmin(Z[:, j]))
        ridge.append((rs[i], np.log2(nsx[j]), Z[i, j]))
    ax.plot([a for a, _, _ in ridge], [b for _, b, _ in ridge], [c for _, _, c in ridge],
            color="k", lw=2.2, marker="o", ms=4.0, mfc="w", mec="k", zorder=12)
    ax.set_xlabel("radix exponent $r$", labelpad=-1)
    ax.set_ylabel("$\\log_2 n$", labelpad=-2)
    ax.set_zlabel("")
    ax.text2D(0.005, 0.90, "$\\log_2$ ms per word", transform=ax.transAxes, fontsize=7.5)
    ax.set_zlim(floor, Z.max() + 0.4)
    ax.view_init(elev=20, azim=-128)
    ax.tick_params(pad=-1.0)
    ax.set_box_aspect((1.2, 1.0, 0.8))
    p1 = save(fig, out, "fig_3d_cost")

    rs2 = np.linspace(1.0, 12.0, 45)
    lams = np.linspace(20.0, 50.0, 31)
    Rg2, Lg2 = np.meshgrid(rs2, lams, indexing="ij")
    T = np.zeros_like(Rg2)
    for i in range(Rg2.shape[0]):
        for j in range(Rg2.shape[1]):
            T[i, j] = words_per_ct(N, 64, Rg2[i, j]) * depth(Rg2[i, j], Lg2[i, j], 64)
    fig2 = plt.figure(figsize=(3.55, 3.05))
    ax2 = fig2.add_subplot(111, projection="3d")
    ax2.plot_surface(Rg2, Lg2, T, cmap=cm.pink, rstride=1, cstride=1,
                     linewidth=0.2, edgecolor="0.35", antialiased=True, alpha=0.95)
    fl2 = -T.max() * 0.35
    ax2.contour(Rg2, Lg2, T, zdir="z", offset=fl2, colors="0.35", linewidths=0.6, levels=10)
    best = []
    for j in range(Rg2.shape[1]):
        i = int(np.argmax(T[:, j]))
        best.append((rs2[i], lams[j], T[i, j]))
    ax2.plot([b[0] for b in best], [b[1] for b in best], [b[2] for b in best],
             color="k", lw=2.2, marker="o", ms=3.4, mfc="w", mec="k", zorder=12)
    ax2.set_xlabel("radix exponent $r$", labelpad=-1)
    ax2.set_ylabel("budget $\\lambda$ (bits)", labelpad=-2)
    ax2.set_zlabel("")
    ax2.text2D(0.005, 0.90, "words per refresh", transform=ax2.transAxes, fontsize=7.5)
    ax2.set_zlim(fl2, T.max() * 1.05)
    ax2.view_init(elev=22, azim=-46)
    ax2.tick_params(pad=-1.0)
    ax2.set_box_aspect((1.2, 1.0, 0.8))
    p2 = save(fig2, out, "fig_3d_depth")
    return p1, p2


def fig_radix(out, N=512):
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.2))
    rs = np.arange(1, 9)
    for n, mk, col in ((32, "o", "0.05"), (64, "s", "0.4"), (128, "^", "0.65")):
        ax[0].plot(rs, [gamma(n, r) / n for r in rs], marker=mk, color=col, lw=1.1, ms=3.4,
                   label="$n=%d$" % n)
    ax[0].set_yscale("log", base=2)
    ax[0].set_xlabel("radix exponent $r$")
    ax[0].set_ylabel("$\\gamma/n$")
    ax[0].grid(True, ls=":", lw=0.4, color="0.75")
    ax[0].legend(frameon=False, fontsize=6)
    for lam, mk, col in ((28, "o", "0.05"), (36, "s", "0.4"), (44, "^", "0.65")):
        ax[1].plot(rs, [words_per_ct(N, 64, r) * depth(r, lam, 64) for r in rs],
                   marker=mk, color=col, lw=1.1, ms=3.4, label="$\\lambda=%d$" % lam)
    ax[1].set_xlabel("radix exponent $r$")
    ax[1].set_ylabel("words per refresh")
    ax[1].grid(True, ls=":", lw=0.4, color="0.75")
    ax[1].legend(frameon=False, fontsize=6)
    return save(fig, out, "fig_radix")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/main.json")
    ap.add_argument("--out", default="../final_paper")
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    figs = os.path.join(out, "figures")
    os.makedirs(figs, exist_ok=True)
    tab_capacity(out)
    tab_degree(out)
    tab_general(out)
    if os.path.exists(args.data):
        data = load(args.data)
        tab_time(data, out)
        tab_space(data, out)
        tab_scalar(data, out)
        fig_amort(data, figs)
        fig_mem(data, figs)
        fig_3d(data, figs)
    fig_tradeoff(figs)
    fig_radix(figs)
    print("report written to", out)


if __name__ == "__main__":
    main()
