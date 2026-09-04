import math
import random
from fractions import Fraction
from math import gcd


def primes_upto(n):
    if n < 2:
        return []
    sieve = bytearray([1]) * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i:: i] = bytearray(len(sieve[i * i:: i]))
    return [p for p in range(2, n + 1) if sieve[p]]


def content(vec):
    g = 0
    for c in vec:
        g = gcd(g, abs(c))
    return g


def box_sides(box):
    return [hi - lo for (lo, hi) in box]


def box_volume(box):
    v = 1
    for (lo, hi) in box:
        v *= (hi - lo)
    return v


def qlattice_basis(r, lam):
    t = len(lam) + 1
    rows = [[0] * t for _ in range(t)]
    rows[0][0] = r
    for i in range(1, t):
        rows[i][0] = lam[i - 1] % r
        rows[i][i] = 1
    return rows


def in_lattice(r, lam, c):
    acc = 0
    for i in range(1, len(c)):
        acc += lam[i - 1] * c[i]
    return (c[0] - acc) % r == 0


def skew_weights(box):
    return [1.0 / float(max(hi - lo, 1)) ** 2 for (lo, hi) in box]


def _dotw(u, v, w):
    s = 0.0
    for i in range(len(u)):
        s += w[i] * u[i] * v[i]
    return s


def gram_schmidt(B, w):
    n = len(B)
    d = len(B[0])
    bs = []
    mu = [[0.0] * n for _ in range(n)]
    nn = []
    for i in range(n):
        v = [float(B[i][j]) for j in range(d)]
        for j in range(i):
            if nn[j] <= 0.0:
                mu[i][j] = 0.0
                continue
            mu[i][j] = _dotw([float(x) for x in B[i]], bs[j], w) / nn[j]
            for k in range(d):
                v[k] -= mu[i][j] * bs[j][k]
        bs.append(v)
        nn.append(_dotw(v, v, w))
    return bs, mu, nn


def lll(B, w, delta=0.99, maxsteps=20000):
    B = [list(map(int, row)) for row in B]
    n = len(B)
    if n <= 1:
        return B
    bs, mu, nn = gram_schmidt(B, w)
    k = 1
    steps = 0
    while k < n and steps < maxsteps:
        steps += 1
        changed = False
        for j in range(k - 1, -1, -1):
            q = int(math.floor(mu[k][j] + 0.5))
            if q != 0:
                B[k] = [B[k][i] - q * B[j][i] for i in range(len(B[k]))]
                changed = True
        if changed:
            bs, mu, nn = gram_schmidt(B, w)
        if nn[k] >= (delta - mu[k][k - 1] ** 2) * nn[k - 1]:
            k += 1
        else:
            B[k], B[k - 1] = B[k - 1], B[k]
            bs, mu, nn = gram_schmidt(B, w)
            k = max(k - 1, 1)
    return B


class EnumContext:
    def __init__(self, B, w):
        self.B = lll(B, w)
        self.w = w
        self.bs, self.mu, self.nn = gram_schmidt(self.B, w)
        self.n = len(self.B)
        self.d = len(self.B[0])
        self.nodes = 0

    def enumerate(self, shift, lo, hi):
        n, d = self.n, self.d
        B, mu, nn, bs, w = self.B, self.mu, self.nn, self.bs, self.w
        tgt = [(lo[i] + hi[i] - 1) * 0.5 - shift[i] for i in range(d)]
        rad2 = 0.0
        for i in range(d):
            half = (hi[i] - lo[i]) * 0.5
            rad2 += w[i] * half * half
        rad2 = rad2 * (1.0 + 1e-9) + 1e-9
        tau = []
        for i in range(n):
            tau.append(_dotw(tgt, bs[i], w) / nn[i] if nn[i] > 0 else 0.0)
        out = []
        z = [0] * n
        acc = [[0] * d for _ in range(n + 1)]

        def rec(i, rho):
            self.nodes += 1
            if i < 0:
                v = [shift[m] + acc[0][m] for m in range(d)]
                for m in range(d):
                    if not (lo[m] <= v[m] < hi[m]):
                        return
                out.append(tuple(v))
                return
            rem = rad2 - rho
            if rem < 0.0:
                return
            s = -tau[i]
            for j in range(i + 1, n):
                s += mu[j][i] * z[j]
            half = math.sqrt(rem / nn[i]) if nn[i] > 0 else 0.0
            centre = -s
            a = int(math.ceil(centre - half - 1e-12))
            b = int(math.floor(centre + half + 1e-12))
            row = B[i]
            for zi in range(a, b + 1):
                z[i] = zi
                dst = acc[i]
                src = acc[i + 1]
                for m in range(d):
                    dst[m] = src[m] + zi * row[m]
                rec(i - 1, rho + (zi + s) * (zi + s) * nn[i])
            z[i] = 0

        rec(n - 1, 0.0)
        return out


def normalised_gso(B, box):
    w = skew_weights(box[:len(B)])
    bs, mu, nn = gram_schmidt(B, w)
    return [math.sqrt(x) for x in nn]


def sieve_level(box, r):
    t = len(box)
    prod = 1
    for j in range(t):
        prod *= (box[j][1] - box[j][0])
        if prod >= r:
            return j
    return t - 1


def brute_sieve(r, lam, box):
    t = len(box)
    out = []
    idx = [box[i][0] for i in range(t)]
    while True:
        if in_lattice(r, lam, idx):
            out.append(tuple(idx))
        j = t - 1
        while j >= 0:
            idx[j] += 1
            if idx[j] < box[j][1]:
                break
            idx[j] = box[j][0]
            j -= 1
        if j < 0:
            break
    return out


def line_sieve(r, lam, box):
    t = len(box)
    lo0, hi0 = box[0]
    out = []
    if t == 1:
        c0 = lo0 + ((0 - lo0) % r)
        while c0 < hi0:
            out.append((c0,))
            c0 += r
        return out
    outer = [box[i] for i in range(1, t)]
    idx = [outer[i][0] for i in range(t - 1)]
    while True:
        acc = 0
        for i in range(t - 1):
            acc += lam[i] * idx[i]
        c0 = lo0 + ((acc - lo0) % r)
        while c0 < hi0:
            out.append(tuple([c0] + idx))
            c0 += r
        j = t - 2
        while j >= 0:
            idx[j] += 1
            if idx[j] < outer[j][1]:
                break
            idx[j] = outer[j][0]
            j -= 1
        if j < 0:
            break
    return out


def fk_reduce(r, lam, I):
    lam %= r
    if lam == 0:
        return (0, 1), (0, 1)
    xp, yp = lam, 1
    xm, ym = lam - r, 1
    while xp >= I and -xm >= I:
        if xp > -xm:
            k = xp // (-xm)
            xp += k * xm
            yp += k * ym
        else:
            k = (-xm) // xp
            xm += k * xp
            ym += k * yp
    if xp >= I and xm != 0:
        k = xp // (-xm)
        xp += k * xm
        yp += k * ym
    if -xm >= I and xp != 0:
        k = (-xm) // xp
        xm += k * xp
        ym += k * yp
    return (xm, ym), (xp, yp)


def fk_next(x, y, I, a, b):
    if x < I - b[0]:
        return x + b[0], y + b[1]
    if x >= -a[0]:
        return x + a[0], y + a[1]
    return x + a[0] + b[0], y + a[1] + b[1]


class PlaneSolver:
    def __init__(self, r, lam1, box0, box1):
        self.r = r
        self.lam1 = lam1 % r
        self.lo0, self.hi0 = box0
        self.lo1, self.hi1 = box1
        self.I = self.hi0 - self.lo0
        self.usable = (self.I <= r and self.lam1 != 0)
        if self.usable:
            self.a, self.b = fk_reduce(r, self.lam1, self.I)
            self.usable = (self.b[0] - self.a[0] >= self.I
                           and self.a[1] > 0 and self.b[1] > 0)
        B = [[r, 0], [self.lam1, 1]]
        w = [1.0 / float(max(self.I, 1)) ** 2,
             1.0 / float(max(self.hi1 - self.lo1, 1)) ** 2]
        self.ctx = EnumContext(B, w)

    def first(self, mu):
        window = max(1, min(self.hi1 - self.lo1, self.r // max(self.I, 1) + 1))
        lo1 = self.lo1
        while lo1 < self.hi1:
            hi1 = min(self.hi1, lo1 + window)
            pts = self.ctx.enumerate((mu % self.r, 0),
                                     (self.lo0, lo1), (self.hi0, hi1))
            if pts:
                return min(pts, key=lambda p: p[1])
            lo1 = hi1
            window *= 2
        return None

    def solve(self, mu):
        if not self.usable:
            return self.ctx.enumerate((mu % self.r, 0),
                                      (self.lo0, self.lo1),
                                      (self.hi0, self.hi1))
        p0 = self.first(mu)
        if p0 is None:
            return []
        out = [p0]
        x = p0[0] - self.lo0
        y = p0[1]
        while True:
            x, y = fk_next(x, y, self.I, self.a, self.b)
            if y >= self.hi1:
                break
            out.append((x + self.lo0, y))
        return out


def plane_sieve(r, lam, box):
    t = len(box)
    if t < 2:
        return line_sieve(r, lam, box)
    solver = PlaneSolver(r, lam[0], box[0], box[1])
    if t == 2:
        return solver.solve(0)
    outer = [box[i] for i in range(2, t)]
    idx = [outer[i][0] for i in range(t - 2)]
    out = []
    while True:
        acc = 0
        for i in range(t - 2):
            acc += lam[i + 1] * idx[i]
        for p in solver.solve(acc % r):
            out.append(tuple(list(p) + idx))
        j = t - 3
        while j >= 0:
            idx[j] += 1
            if idx[j] < outer[j][1]:
                break
            idx[j] = outer[j][0]
            j -= 1
        if j < 0:
            break
    return out


class FiberedSieve:
    def __init__(self, r, lam, box, level=None):
        self.r = r
        self.lam = list(lam)
        self.box = list(box)
        self.t = len(box)
        self.level = sieve_level(box, r) if level is None else level
        m = self.level + 1
        self.m = m
        self.lo = [self.box[i][0] for i in range(m)]
        self.hi = [self.box[i][1] for i in range(m)]
        self.ctx = None
        self.solver = None
        if m == 1:
            self.mode = "ap"
        elif m == 2:
            self.mode = "fk"
            self.solver = PlaneSolver(r, self.lam[0], self.box[0], self.box[1])
        else:
            self.mode = "enum"
            self.ctx = EnumContext(qlattice_basis(r, self.lam[:self.level]),
                                   skew_weights(self.box[:m]))
        self.slices = 0

    def nodes(self):
        return self.ctx.nodes if self.ctx is not None else 0

    def slice_solve(self, mu):
        if self.mode == "ap":
            lo0, hi0 = self.box[0]
            out = []
            c0 = lo0 + ((mu - lo0) % self.r)
            while c0 < hi0:
                out.append((c0,))
                c0 += self.r
            return out
        if self.mode == "fk":
            return self.solver.solve(mu % self.r)
        return self.ctx.enumerate([mu % self.r] + [0] * (self.m - 1),
                                  self.lo, self.hi)

    def run_count(self):
        t, m, r = self.t, self.m, self.r
        if m == t:
            self.slices = 1
            return len(self.slice_solve(0))
        outer = [self.box[i] for i in range(m, t)]
        idx = [outer[i][0] for i in range(t - m)]
        total = 0
        while True:
            acc = 0
            for i in range(t - m):
                acc += self.lam[m - 1 + i] * idx[i]
            self.slices += 1
            total += len(self.slice_solve(acc % r))
            j = t - m - 1
            while j >= 0:
                idx[j] += 1
                if idx[j] < outer[j][1]:
                    break
                idx[j] = outer[j][0]
                j -= 1
            if j < 0:
                break
        return total

    def run(self):
        t, m, r = self.t, self.m, self.r
        out = []
        if m == t:
            self.slices = 1
            return self.slice_solve(0)
        outer = [self.box[i] for i in range(m, t)]
        idx = [outer[i][0] for i in range(t - m)]
        while True:
            acc = 0
            for i in range(t - m):
                acc += self.lam[m - 1 + i] * idx[i]
            self.slices += 1
            for p in self.slice_solve(acc % r):
                out.append(tuple(list(p) + idx))
            j = t - m - 1
            while j >= 0:
                idx[j] += 1
                if idx[j] < outer[j][1]:
                    break
                idx[j] = outer[j][0]
                j -= 1
            if j < 0:
                break
        return out


def fibered_sieve(r, lam, box, level=None):
    return FiberedSieve(r, lam, box, level).run()


def section_lattices(r, lam, box):
    t = len(box)
    out = []
    for k in range(t):
        sub = qlattice_basis(r, lam[:k])
        w = skew_weights(box[:k + 1])
        out.append(lll(sub, w))
    return out


def _small_combinations(B, bound):
    n = len(B)
    d = len(B[0])
    res = []
    coeff = [0] * n

    def rec(i):
        if i == n:
            if any(coeff):
                v = [0] * d
                for j in range(n):
                    if coeff[j]:
                        for m in range(d):
                            v[m] += coeff[j] * B[j][m]
                res.append(tuple(v))
            return
        for c in range(-bound, bound + 1):
            coeff[i] = c
            rec(i + 1)
        coeff[i] = 0

    rec(0)
    return res


def transition_candidates(r, lam, box, bound=2):
    t = len(box)
    sides = box_sides(box)
    cand = []
    for k in range(t):
        Bk = lll(qlattice_basis(r, lam[:k]), skew_weights(box[:k + 1]))
        vecs = []
        for v in _small_combinations(Bk, bound):
            if v[k] <= 0:
                continue
            if v[k] >= sides[k]:
                continue
            ok = True
            for i in range(k):
                if abs(v[i]) >= sides[i]:
                    ok = False
                    break
            if ok:
                vecs.append(tuple(list(v) + [0] * (t - k - 1)))
        vecs = sorted(set(vecs), key=lambda u: (u[k], sum(abs(x) for x in u)))
        cand.append(vecs)
    return cand


def babai(ctx, target):
    n, d = ctx.n, ctx.d
    B, mu, nn, bs, w = ctx.B, ctx.mu, ctx.nn, ctx.bs, ctx.w
    tau = []
    for i in range(n):
        tau.append(_dotw([float(x) for x in target], bs[i], w) / nn[i]
                   if nn[i] > 0 else 0.0)
    z = [0] * n
    for i in range(n - 1, -1, -1):
        s = tau[i]
        for j in range(i + 1, n):
            s -= mu[j][i] * z[j]
        z[i] = int(math.floor(s + 0.5))
    v = [0] * d
    for i in range(n):
        if z[i]:
            for m in range(d):
                v[m] += z[i] * B[i][m]
    return v


class TransitionSieve:
    def __init__(self, r, lam, box, bound=2, fallback=True):
        self.r = r
        self.lam = list(lam)
        self.box = list(box)
        self.t = len(box)
        self.T = transition_candidates(r, lam, box, bound)
        self.fallback = fallback
        self.ctx = []
        for k in range(self.t):
            self.ctx.append(EnumContext(qlattice_basis(r, lam[:max(k, 0)]),
                                        skew_weights(box[:k + 1])))

    def _in_prefix(self, c, k):
        for i in range(k + 1):
            if not (self.box[i][0] <= c[i] < self.box[i][1]):
                return False
        return True

    def _step(self, c, k, sign):
        best = None
        for v in self.T[k]:
            cand = [c[i] + sign * v[i] for i in range(self.t)]
            if not (self.box[k][0] <= cand[k] < self.box[k][1]):
                continue
            if self._in_prefix(cand, k):
                if best is None or abs(cand[k] - c[k]) < abs(best[k] - c[k]):
                    best = cand
        if best is not None:
            return best
        if not self.fallback or k == 0:
            return None
        for v in self.T[k]:
            base = [c[i] + sign * v[i] for i in range(self.t)]
            tgt = [(self.box[i][0] + self.box[i][1] - 1) * 0.5 - base[i]
                   for i in range(k)]
            corr = babai(self.ctx[k - 1], tgt)
            cand = list(base)
            for i in range(k):
                cand[i] += corr[i]
            if (self.box[k][0] <= cand[k] < self.box[k][1]
                    and self._in_prefix(cand, k)
                    and in_lattice(self.r, self.lam, cand)):
                w = tuple(cand[i] - c[i] for i in range(self.t))
                if w[k] * sign > 0:
                    self.T[k].append(tuple(sign * x for x in w))
                    self.T[k].sort(key=lambda u: (u[k],
                                                  sum(abs(x) for x in u)))
                return cand
        return None

    def run(self):
        found = set()
        start = [0] * self.t
        if not self._in_prefix(start, self.t - 1):
            return []

        def walk(k, c):
            for sign in (1, -1):
                cur = list(c)
                if sign == -1:
                    nxt = self._step(cur, k, -1)
                    if nxt is None:
                        continue
                    cur = nxt
                while self._in_prefix(cur, self.t - 1) or True:
                    if self._in_prefix(cur, self.t - 1):
                        found.add(tuple(cur))
                    if k > 0:
                        walk(k - 1, cur)
                    nxt = self._step(cur, k, sign)
                    if nxt is None:
                        break
                    cur = nxt

        walk(self.t - 1, start)
        return sorted(found)


def space_sieve(r, lam, box, bound=2):
    return TransitionSieve(r, lam, box, bound).run()


def line_sieve_work(r, lam, box):
    w = 1
    for i in range(1, len(box)):
        w *= (box[i][1] - box[i][0])
    return w


def plane_sieve_work(r, lam, box):
    w = 1
    for i in range(2, len(box)):
        w *= (box[i][1] - box[i][0])
    return w


def fibered_sieve_work(r, lam, box, level=None):
    t = len(box)
    if level is None:
        level = sieve_level(box, r)
    w = 1
    for i in range(level + 1, t):
        w *= (box[i][1] - box[i][0])
    return w


def array_collection(ideals, box, sieve=None):
    vol = box_volume(box)
    store = bytearray(vol)
    sides = box_sides(box)
    if sieve is None:
        sieve = fibered_sieve
    for (r, lam) in ideals:
        for p in sieve(r, lam, box):
            pos = 0
            for i in range(len(box)):
                pos = pos * sides[i] + (p[i] - box[i][0])
            if store[pos] < 255:
                store[pos] += 1
    hits = sum(1 for v in store if v)
    return hits, vol


def blocks_of(box, split_from):
    t = len(box)
    if split_from >= t:
        yield list(box)
        return
    outer = [box[i] for i in range(split_from, t)]
    idx = [outer[i][0] for i in range(t - split_from)]
    while True:
        sub = [box[i] for i in range(split_from)]
        sub += [(idx[i], idx[i] + 1) for i in range(t - split_from)]
        yield sub
        j = t - split_from - 1
        while j >= 0:
            idx[j] += 1
            if idx[j] < outer[j][1]:
                break
            idx[j] = outer[j][0]
            j -= 1
        if j < 0:
            break


def blocked_collection(ideals, box, split_from, sieve=None):
    t = len(box)
    if sieve is None:
        sieve = fibered_sieve
    total = 0
    peak_cells = 0
    nblocks = 0
    for sub in blocks_of(box, split_from):
        vol = box_volume(sub)
        peak_cells = max(peak_cells, vol)
        nblocks += 1
        store = bytearray(vol)
        sides = box_sides(sub)
        for (r, lam) in ideals:
            for p in sieve(r, lam, sub):
                pos = 0
                for i in range(t):
                    pos = pos * sides[i] + (p[i] - sub[i][0])
                if store[pos] < 255:
                    store[pos] += 1
        total += sum(1 for v in store if v)
        del store
    return total, peak_cells, nblocks


def random_instance(t, side, r, rng):
    lam = [rng.randrange(0, r) for _ in range(t - 1)]
    box = [(-side, side) for _ in range(t - 1)] + [(0, side)]
    return lam, box


def expected_count(r, box):
    return box_volume(box) / float(r)


def primitive_points(points, MQ=None):
    out = []
    for p in points:
        v = p if MQ is None else tuple(
            sum(p[i] * MQ[i][j] for i in range(len(p)))
            for j in range(len(MQ[0])))
        if content(v) == 1:
            out.append(p)
    return out


def count_primitive_cuboid(lo, hi):
    t = len(lo)
    mx = max(max(abs(lo[i]), abs(hi[i])) for i in range(t))
    mu = mobius_upto(mx + 1)
    total = 0
    for m in range(1, mx + 1):
        if mu[m] == 0:
            continue
        prod = 1
        for i in range(t):
            a = -((-lo[i]) // m)
            b = (hi[i] - 1) // m
            cnt = b - a + 1
            if cnt <= 0:
                prod = 0
                break
            prod *= cnt
        if prod:
            total += mu[m] * prod
    return total


def mobius_upto(n):
    mu = [1] * (n + 1)
    primes = []
    is_comp = [False] * (n + 1)
    mu[0] = 0
    for i in range(2, n + 1):
        if not is_comp[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > n:
                break
            is_comp[i * p] = True
            if i % p == 0:
                mu[i * p] = 0
                break
            mu[i * p] = -mu[i]
    return mu


def zeta_value(k, terms=200000):
    s = 0.0
    for n in range(1, terms + 1):
        s += 1.0 / n ** k
    return s


def primitive_density(k, P=100000):
    val = Fraction(1)
    for p in primes_upto(P):
        val *= Fraction(p ** k - 1, p ** k)
    tail = Fraction(1, (k - 1) * (P - 1) ** (k - 1))
    return val * (1 - tail), val
