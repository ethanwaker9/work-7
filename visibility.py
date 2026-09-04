import math
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


def vgcd(vec):
    g = 0
    for c in vec:
        g = gcd(g, abs(c))
    return g


def classes_mod_p(S, p):
    return sorted(set(tuple(c % p for c in v) for v in S))


def s_of_p(S, p):
    return len(classes_mod_p(S, p))


def count_bruteforce(S, L):
    k = len(S[0])
    idx = [1] * k
    cnt = 0
    while True:
        ok = True
        for v in S:
            if vgcd([idx[i] - v[i] for i in range(k)]) != 1:
                ok = False
                break
        if ok:
            cnt += 1
        j = k - 1
        while j >= 0:
            idx[j] += 1
            if idx[j] <= L:
                break
            idx[j] = 1
            j -= 1
        if j < 0:
            break
    return cnt


def count_marking(S, L):
    k = len(S[0])
    A = max(abs(c) for v in S for c in v) if S else 0
    total = L ** k
    marked = bytearray(total)
    plist = primes_upto(L + A + 1)
    for p in plist:
        for cls in classes_mod_p(S, p):
            starts = []
            for i in range(k):
                r = cls[i] % p
                first = r if r >= 1 else p
                if first > L:
                    starts = None
                    break
                starts.append(first)
            if starts is None:
                continue
            coords = [list(range(first, L + 1, p)) for first in starts]
            def rec(i, base):
                if i == k:
                    marked[base] = 1
                    return
                mult = L ** (k - 1 - i)
                for x in coords[i]:
                    rec(i + 1, base + (x - 1) * mult)
            rec(0, 0)
    inbox = set()
    for v in S:
        if all(1 <= c <= L for c in v):
            pos = 0
            for i in range(k):
                pos = pos * L + (v[i] - 1)
            inbox.add(pos)
    cnt = 0
    for pos in range(total):
        if not marked[pos] and pos not in inbox:
            cnt += 1
    return cnt


def count_1d(L, m, r):
    r = r % m
    if r == 0:
        return L // m
    if r > L:
        return 0
    return (L - r) // m + 1


def finite_sieve_count(kdim, L, EX):
    primes = {}
    for (p, cls) in EX:
        primes.setdefault(p, set()).add(tuple(c % p for c in cls))
    plist = sorted(primes.keys())
    total = 0
    stack = [(0, 1, (0,) * kdim, 1)]
    while stack:
        (idx, m, res, sign) = stack.pop()
        if idx == len(plist):
            term = 1
            for i in range(kdim):
                term *= count_1d(L, m, res[i])
                if term == 0:
                    break
            total += sign * term
            continue
        p = plist[idx]
        stack.append((idx + 1, m, res, sign))
        inv = pow(m, -1, p)
        for cls in primes[p]:
            newres = tuple(res[i] + m * (((cls[i] - res[i]) * inv) % p)
                           for i in range(kdim))
            stack.append((idx + 1, m * p, newres, -sign))
    return total


def _blocked_factor_lists(L, shifts, block=None):
    if block is None:
        block = max(int(L ** 0.5) + 1, 1024)
    A = max(abs(a) for a in shifts) if shifts else 0
    plist = primes_upto(int((L + A + 1) ** 0.5) + 1)
    lo = 1
    while lo <= L:
        hi = min(lo + block - 1, L)
        n = hi - lo + 1
        vals = [[x - a for x in range(lo, hi + 1)] for a in shifts]
        fac = [[[] for _ in range(n)] for _ in shifts]
        for j, a in enumerate(shifts):
            vj = vals[j]
            fj = fac[j]
            for p in plist:
                start = (a - lo) % p
                for idx in range(start, n, p):
                    if vj[idx] != 0 and vj[idx] % p == 0:
                        fj[idx].append(p)
                        while vj[idx] % p == 0:
                            vj[idx] //= p
            for idx in range(n):
                rem = abs(vj[idx])
                if rem > 1:
                    fj[idx].append(rem)
        yield (lo, hi, fac)
        lo = hi + 1


def count_exact(S, L):
    k = len(S[0])
    return _fiber(k, L, [tuple(v) for v in S], [])


def _fiber(kdim, L, VS, EX):
    if not VS:
        return finite_sieve_count(kdim, L, EX)
    if kdim == 1:
        cnt = 0
        for x in range(1, L + 1):
            ok = True
            for v in VS:
                if abs(x - v[0]) != 1:
                    ok = False
                    break
            if ok:
                for (p, cls) in EX:
                    if (x - cls[0]) % p == 0:
                        ok = False
                        break
            if ok:
                cnt += 1
        return cnt
    total = 0
    shifts = [v[0] for v in VS]
    for (lo, hi, fac) in _blocked_factor_lists(L, shifts):
        for x1 in range(lo, hi + 1):
            idx = x1 - lo
            VSp = []
            EXp = []
            for (p, cls) in EX:
                if (x1 - cls[0]) % p == 0:
                    EXp.append((p, cls[1:]))
            special = False
            for j, v in enumerate(VS):
                d = x1 - v[0]
                if d == 0:
                    VSp.append(v[1:])
                    special = True
                elif abs(d) != 1:
                    for p in fac[j][idx]:
                        EXp.append((p, v[1:]))
            if special:
                total += _fiber(kdim - 1, L, VSp, EXp)
            else:
                total += finite_sieve_count(kdim - 1, L, EXp)
    return total


def density_bounds(S, k, P=100000):
    lower = Fraction(1)
    for p in primes_upto(P):
        sp = s_of_p(S, p)
        if sp >= p ** k:
            return Fraction(0), Fraction(0)
        lower *= Fraction(p ** k - sp, p ** k)
    s = len(S)
    tail = Fraction(s, (k - 1) * (P - 1) ** (k - 1))
    lo = lower * (1 - tail) if tail < 1 else Fraction(0)
    return lo, lower


def certify_deficit(S, L, P=100000):
    k = len(S[0])
    N = count_exact(S, L)
    lo, hi = density_bounds(S, k, P)
    is_deficit = Fraction(N) < lo * L ** k
    return N, lo, hi, is_deficit


def n_shift(S, p, i, target):
    return sum(1 for w in classes_mod_p(S, p) if w[i] == target % p)


def _squarefree_with_factors(Dmax):
    out = []
    for d in range(1, Dmax + 1):
        x = d
        pr = []
        ok = True
        p = 2
        while p * p <= x:
            if x % p == 0:
                x //= p
                pr.append(p)
                if x % p == 0:
                    ok = False
                    break
            p += 1
        if ok and x > 1:
            pr.append(x)
        if ok:
            out.append((d, pr))
    return out


def B_criterion(S, k, u, Dmax=600):
    s = len(S)
    total = 0.0
    for (d, pr) in _squarefree_with_factors(Dmax):
        sd = 1
        for p in pr:
            sd *= s_of_p(S, p)
        um = u % d
        C = 0.0
        for v in S:
            for i in range(k):
                pos = 1 if v[i] >= 1 else 0
                J = min(um + pos, d)
                for j in range(J):
                    m = j + (0 if pos else 1)
                    prod = 1.0
                    for p in pr:
                        prod *= n_shift(S, p, i, v[i] + m)
                        if prod == 0.0:
                            break
                    C += prod
        C -= k * s * sd * (um / d)
        total += (-1) ** (len(pr) + 1) * d ** (1 - k) * C
    tail = 2.0 * k * s * (u + 2) * (s ** 2) * Dmax ** (2 - k)
    return total, tail


def jordan_scan_counts(k, Lmax):
    spf = list(range(Lmax + 1))
    for i in range(2, int(Lmax ** 0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, Lmax + 1, i):
                if spf[j] == j:
                    spf[j] = i
    binom = [math.comb(k, j) for j in range(k + 1)]
    counts = []
    running = 0
    for m in range(1, Lmax + 1):
        x = m
        ps = []
        while x > 1:
            p = spf[x]
            ps.append(p)
            while x % p == 0:
                x //= p
        Js = []
        for r in range(k):
            e = k - 1 - r
            if e == 0:
                Js.append(1 if m == 1 else 0)
                continue
            val = m ** e
            for p in ps:
                val = val // p ** e * (p ** e - 1)
            Js.append(val)
        shell = 0
        for r in range(k):
            shell += (-1) ** r * binom[r + 1] * Js[r]
        running += shell
        counts.append(running)
    return counts


def certificate_constants(S, k, z, t):
    P = Fraction(1)
    sigma = Fraction(0)
    Theta = Fraction(1)
    Theta2 = Fraction(1)
    Lam = 0
    for p in primes_upto(z):
        sp = s_of_p(S, p)
        P *= Fraction(p ** k - sp, p ** k)
        sigma += Fraction(sp, p ** k)
        Theta *= Fraction(p ** (k - 1) + sp, p ** (k - 1))
        Theta2 *= (Fraction(p ** (k - 2) + sp, p ** (k - 2))
                   if k >= 3 else Fraction(1))
        Lam += sp
    r = 2 * t + 2
    esig = Fraction(1)
    term = Fraction(1)
    for j in range(1, 40):
        term = term * sigma / j
        esig += term
    bonf = (sigma ** r) * esig / math.factorial(r)
    Gam = Fraction(0)
    term = Fraction(1)
    for j in range(0, 2 * t + 2):
        if j > 0:
            term = term * Lam / j
        Gam += term
    return P, bonf, Theta, Theta2, Gam


def certified_lower_bound(S, k, z, t, L):
    if k < 3 or L < 4 * k * k:
        return None
    s = len(S)
    A = max(abs(c) for v in S for c in v) if S else 0
    P, bonf, Theta, Theta2, Gam = certificate_constants(S, k, z, t)
    sqL = math.isqrt(L)
    return (P - bonf
            - Fraction(k) * Theta / L
            - Fraction(2 ** k) * Theta2 / L ** 2
            - Fraction(2 * s, (k - 1) * z ** (k - 1))
            - Fraction(2 ** k * s, (k - 1) * max(sqL - 1, 1) ** (k - 1))
            - Fraction(2) * Gam / L ** k
            - Fraction(s * (A + 1), L ** k))


def certify_sd_minimum(S, ratios, zlist=(20, 40, 80, 160, 320), tlist=(1, 2)):
    k = len(S[0])
    best = min(ratios, key=lambda pr: pr[1])
    q = best[1] if isinstance(best[1], Fraction) else Fraction(best[1])
    Lscan = max(L for (L, _) in ratios)
    bestcert = None
    for z in zlist:
        for t in tlist:
            lo, hi = 4 * k * k, 1
            while hi <= 10 ** 9:
                hi = max(4 * k * k, hi * 2)
                b = certified_lower_bound(S, k, z, t, hi)
                if b is not None and b > q:
                    break
            if hi > 10 ** 9:
                continue
            lo = 4 * k * k
            while lo + 1 < hi:
                mid = (lo + hi) // 2
                b = certified_lower_bound(S, k, z, t, mid)
                if b is not None and b > q:
                    hi = mid
                else:
                    lo = mid
            if bestcert is None or hi < bestcert[0]:
                bestcert = (hi, z, t)
    if bestcert is None:
        return best, None, None
    L1, z, t = bestcert
    verified = L1 <= Lscan
    return best, L1, (z, t, verified)


def sd_scan_exact(S, Lmax, P=100000):
    k = len(S[0])
    lo, hi = density_bounds(S, k, P)
    best = None
    deficits = []
    ratios = []
    for L in range(1, Lmax + 1):
        N = count_exact(S, L)
        r = Fraction(N, L ** k)
        ratios.append((L, N))
        if r < lo:
            deficits.append(L)
        if best is None or r < best[1]:
            best = (L, r)
    return best, deficits, ratios, (lo, hi)
