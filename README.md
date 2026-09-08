# Packed Triangle Encoding for SIMD Arithmetic and Logic FHE

This repository contains implementation and benchmark suite for the packed triangle encoding, together with
re-implementations of the schemes it is compared against, all on one shared backend with an RNS-CKKS engine with NTT and hybrid key switching, the discrete-CKKS bootstrapping primitive, the arithmetic ring layer, the five schemes, the theory support routines for the slot capacity and the degree lower bounds, and the benchmark harness.

## Files and Contents
Setting the radix exponent to one recovers the encoding that the packed triangle encoding is
compared against, so the two are measured by exactly the same code path.
| Module | Contents |
| --- | --- |
| `pte/rns.py` | prime search, negacyclic NTT, RNS context |
| `pte/ring.py` | ring context, sampling, fast base conversion |
| `pte/ckks.py` | RNS-CKKS: encode, encrypt, add, multiply, relinearize, rescale, rotate, conjugate, level and scale alignment |
| `pte/lintrans.py` | diagonal linear transforms, slot to coefficient and coefficient to slot matrices |
| `pte/bootstrap.py` | truncation, modulus raising, Chebyshev evaluation, look-up tables of interpolation order `kappa`, the bootstrapping primitive |
| `pte/aring.py` | arithmetic ring `Z[X]/(F)`, triangle encoding for any radix and any modulus, LLL, Babai, two slot polynomial search |
| `pte/scheme.py` | the packed triangle scheme as encode, add, multiply, refresh, arithmetic-to-boolean, boolean-to-arithmetic, boolean refresh and boolean gates |
| `pte/schemes/digit_ckks.py` | radix digit SIMD scheme with sequential and carry lookahead reduction |
| `pte/schemes/cggi.py` | LWE/RGSW gate bootstrapping |
| `pte/schemes/scalar_word.py` | integer circuits on the gate scheme, and the scalar machine word scheme over `Z[X]/(X^n - X + 2)` |
| `pte/batching.py` | native slot capacity, the two slot construction and its verification |
| `pte/bounds.py` | minimal clearing degree, carry multilinear degree, trade-off curve |
| `bench/run_bench.py` | benchmark harness, writes JSON |



## Running the Measurements
This requires Python 3.10 or newer, along with numpy>=1.24, scipy>=1.10, matplotlib>=3.6, sympy>=1.12, pytest>=7.0.
```
python3 bench/run_bench.py --N 512 --levels 32 --words 8 16 32 64 --radices 1 2 4 --w 4 \
        --out results/main.json
python3 bench/make_report.py --data results/main.json --out ../final_paper
```
The first command runs every scheme at every configuration, checks each of them for exact
correctness against the plain computation, and writes one JSON file. The second command reads
that file and outputs the EPS figures. A full run takes roughly half an hour on one core; `--words 8 16` cuts it to a few minutes.
```
python3 -c "from pte.batching import capacity_table; print(capacity_table([8,16,32,64],[8,16,32,64]))"
python3 -c "from pte.bounds import clearing_degree; print([clearing_degree(g,2) for g in range(2,13)])"
python3 -c "from pte.bounds import carry_multilinear_degree; print([carry_multilinear_degree(s) for s in range(1,6)])"
python3 -c "from pte.batching import two_slot_instance; print(two_slot_instance(1<<32,32))"
```

## Layout of a ciphertext
An arithmetic ciphertext of ring dimension `N` holds `W = N r / n` machine words of `n` bits at
radix `2^r`. Word `k` occupies the `d/2` consecutive complex slots starting at `k d / 2`, where
`d = n / r` is the degree of the arithmetic ring, and each slot carries one coordinate of the
canonical embedding of the triangle encoding of that word. The conversion produces `2 r` bit
plane ciphertexts; plane `(l, h)` carries bit `l` of digit `i` of every word, in slot `i mod d/2`
of the corresponding block, for `i` in the half `h`.
