# Fibered Lattice Sieving

This repository contains our implementation of research in precise relation collection in dimension visibility counts. We did this in two layers:
* the **geometric layer**, which lists the points of a sublattice
  `Lambda` of index `r` inside a `t`-dimensional cuboid `H`, the operation
  that drives relation collection in the number field sieve and its tower
  variants;
* the **arithmetic layer**, which counts exactly the points of a cuboid that
  are simultaneously visible from a finite set of reference points, that is
  the admissible or duplicate-free candidates of a sieve region.


## Usage and Running
The runtime of pipeline takes about fifteen minutes on one core laptop. The requirements are Python 3.9 or newer, `numpy` and `matplotlib` (needed only by `experiments.py`), and `epstopdf` for plots. To reproduce the experiments:
```bash
python3 experiments.py
```

List the points of a lattice of index `r` inside a cuboid:
```python
from latsieve import fibered_sieve, sieve_level
box = [(-16, 16), (-16, 16), (-16, 16), (0, 16)]
lam = [12345, 6789, 4242]
print(sieve_level(box, 32003), len(fibered_sieve(32003, lam, box)))
```

Count the admissible points of a region precisely:
```python
from visibility import count_exact
print(count_exact([(0, 0, 0), (1, 2, 1)], 200))
```

Validate a deficient region with exact rational arithmetic:
```python
from visibility import certify_deficit
N, lo, hi, deficient = certify_deficit([(1, 2, 1)], 7)
print(N, deficient)
```


## Files and Contents

* `latsieve.py` the sieving library:
  * `qlattice_basis(r, alpha)` basis of the lattice
    `{c : c[0] = alpha[0] c[1] + ... + alpha[t-2] c[t-1] mod r}` of index `r`,
    which is the shape a factor-base ideal takes inside a special-q lattice;
  * `sieve_level(box, r)` the level of the pair, that is the least `j` with
    `I_0 ... I_j >= r`;
  * `brute_sieve`, `line_sieve`, `plane_sieve` the exhaustive baselines,
    the last one using the Franke–Kleinjung walk of `fk_reduce` and `fk_next`;
  * `space_sieve` / `TransitionSieve` a transition vector walk in the sense
    of Grémy, with short vector steps and a Babai fall back; it is fast and
    incomplete, and its recall is measured in the experiments;
  * `FiberedSieve` / `fibered_sieve` the algorithm of our research as it slices
    the region at the level, freezes one reduced basis of the section lattice,
    and resolves each fiber exactly by an arithmetic progression, by the
    Franke–Kleinjung walk, or by `EnumContext.enumerate`;
  * `EnumContext`, `lll`, `gram_schmidt` LLL reduction and Fincke–Pohst
    enumeration for the skew metric that turns the cuboid into a unit cube;
  * `array_collection`, `blocked_collection` relation collection with one
    accumulator array over the whole region, and with one array per block;
  * `content`, `primitive_points`, `count_primitive_cuboid`,
    `primitive_density` the admissibility filter and its density.

* `visibility.py` the arithmetic library:
  * `count_exact(S, L)` exact value of `N_S(L)`, the number of points of
    `[1,L]^k` visible from every point of `S`, by the fiber sieve;
  * `count_bruteforce(S, L)` scan of the region, cost `L^k`;
  * `count_marking(S, L)` Eratosthenes style marking, memory `L^k`;
  * `density_bounds(S, k, P)` exact rational bounds for the asymptotic
    density from the Euler product together with a rigorous tail estimate;
  * `certify_deficit(S, L, P)` exact certificate for a deficient region;
  * `B_criterion(S, k, u, Dmax)` truncated deficit functional `B_u(S)` with
    its tail bound;
  * `certified_lower_bound`, `certify_sd_minimum` the explicit lower bound
    of our research and the certification threshold it produces;
  * `jordan_scan_counts(k, Lmax)` incremental exact counts for the origin
    through the Jordan totient identity, used as an independent check;
  * `sd_scan_exact(S, Lmax, P)` scan of the yields `N_S(L)/L^k`.

* `experiments.py` the core of experiments. It writes
  * `results/e1_correctness.txt` cross-validation of the three counters,
  * `results/e2_timing.txt` counting time and memory,
  * `results/e3_sd.txt`, `results/e3_certified.txt` worst-case yields and
    deficit certificates,
  * `results/e4_bu.txt` values of `B_u`,
  * `results/e5_sieve_validation.txt` the sieves against exhaustive search,
  * `results/e6_sieve_bench.txt` time, work and recall against the level,
  * `results/e7_scaling.txt` scaling in the size of the region,
  * `results/e8_memory.txt` working memory of the relation collection,
  * `results/e9_yield.txt` certified worst-case yields,
  * `results/e10_nodes.txt` enumeration nodes and the shape hypothesis.


## Conventions

A search space is a list of half open integer intervals `[(lo, hi), ...]`, and
the convention of relation collection is used, so the origin belongs to it. A
lattice is described by its index `r` and by the integers `alpha[0..t-2]`, and
the reduction data may be arbitrary residues modulo `r`. For the arithmetic
layer the reference points are integer tuples of a common length `k`, the
region is always `[1,L]^k`, and a reference point lying inside the region is
never counted, because the greatest common divisor of the zero vector is `0`.
Counting functions return exact integers, and no floating point enters any
reported count or certificate; floating point is used in the lattice
reduction, in the plots, and in the truncated evaluation of `B_u(S)`, whose
truncation error is controlled by the returned tail bound.

