"""Recompute M-23 with correct labels, exact chance, and the missing control."""
import itertools, struct, sys
import numpy as np

V, C, L, NH, P = 65, 128, 4, 4, 64
hs = C // NH

def load(path):
    raw = open(path, 'rb').read()[64:]
    a = np.frombuffer(raw, dtype='<f4')
    sizes = [('wte',V*C),('wpe',P*C),('ln1w',L*C),('ln1b',L*C),('qkvw',L*3*C*C),('qkvb',L*3*C),
             ('attprojw',L*C*C),('attprojb',L*C),('ln2w',L*C),('ln2b',L*C),('fcw',L*4*C*C),
             ('fcb',L*4*C),('fcprojw',L*C*4*C),('fcprojb',L*C),('lnfw',C),('lnfb',C)]
    out, k = {}, 0
    for n, s in sizes:
        out[n] = a[k:k+s].astype(np.float64); k += s
    # The file may carry AdamW moments after the params (kCkptHasMoments).
    assert a.size in (k, 3 * k), (k, a.size)
    return out

def ln1_rows(p, layer):
    x = p['wte'].reshape(V, C)
    w = p['ln1w'].reshape(L, C)[layer]; b = p['ln1b'].reshape(L, C)[layer]
    mu = x.mean(1, keepdims=True); var = ((x-mu)**2).mean(1, keepdims=True)
    return (x-mu)/np.sqrt(var+1e-5)*w + b

def ov_table(p, layer, head, wv=None, wo=None):
    xn = ln1_rows(p, layer)
    WV = p['qkvw'].reshape(L,3*C,C)[layer][2*C+head*hs:2*C+(head+1)*hs] if wv is None else wv
    WO = p['attprojw'].reshape(L,C,C)[layer][:, head*hs:(head+1)*hs] if wo is None else wo
    y = (xn @ WV.T) @ WO.T                 # [V, C]
    t = y @ p['wte'].reshape(V, C).T       # [V, V]
    return t - t.mean(0, keepdims=True)    # column-centre

def cat(ch):
    if ch in 'aeiouAEIOU': return 'vowel'
    if ch.isalpha() and ch.isupper(): return 'upper'
    if ch.isalpha(): return 'lower'
    if ch in ' \n': return 'space'
    return 'punct'

vocab = list(open(sys.argv[2]).read())[:V] if len(sys.argv) > 2 else None

def purity(vecs, chars, k=4):
    tot = 0.0
    for v in vecs:
        idx = np.argsort(-v)[:k]
        cs = [cat(chars[i]) for i in idx]
        tot += max(cs.count(x) for x in set(cs))/k
    return tot/len(vecs)

def sides(p, chars, rng=None):
    """Returns (read-side purity, write-side purity) over all heads x 4 directions."""
    reads, writes = [], []
    for l in range(L):
        for h in range(NH):
            if rng is None:
                t = ov_table(p, l, h)
            else:
                wv = rng.normal(0, 0.02, (hs, C)); wo = rng.normal(0, 0.02, (C, hs))
                t = ov_table(p, l, h, wv, wo)
            u, s, vt = np.linalg.svd(t)
            for d in range(4):
                flip = 1.0 if u[np.argmax(np.abs(u[:, d])), d] >= 0 else -1.0
                reads.append(flip*u[:, d])     # left  = source axis = READS
                writes.append(flip*vt[d, :])   # right = target axis = WRITES
    return purity(reads, chars), purity(writes, chars)

chars = list(open(sys.argv[2]).read())[:V]
trained = load(sys.argv[1]); untrained = load(sys.argv[3])

tr_r, tr_w = sides(trained, chars)
un_r, un_w = sides(untrained, chars)

# Exact chance for the max-category-share statistic over 4 distinct characters.
tot = n = 0
for comb in itertools.combinations(range(V), 4):
    cs = [cat(chars[i]) for i in comb]
    tot += max(cs.count(x) for x in set(cs))/4; n += 1
chance = tot/n

# The control the first version was missing: TRAINED embeddings and ln1,
# RANDOM head weights. Isolates the head from the vocabulary geometry that all
# 16 heads share.
rng = np.random.default_rng(0)
null = np.array([sides(trained, chars, rng) for _ in range(12)])

print(f"  exact chance (max-category share, 4 distinct of {V}) : {chance:.4f}\n")
print(f"  {'':<34}{'reads':>10}{'writes':>10}")
print(f"  {'trained':<34}{tr_r:>10.4f}{tr_w:>10.4f}")
print(f"  {'untrained model (head+embeddings)':<34}{un_r:>10.4f}{un_w:>10.4f}")
print(f"  {'trained embeddings, RANDOM head':<34}{null[:,0].mean():>10.4f}{null[:,1].mean():>10.4f}")
print(f"  {'  (sd over 12 draws)':<34}{null[:,0].std():>10.4f}{null[:,1].std():>10.4f}")
p_r = (null[:,0] >= tr_r).mean(); p_w = (null[:,1] >= tr_w).mean()
print(f"\n  p(random head reaches trained)      reads {p_r:.3f}   writes {p_w:.3f}")
