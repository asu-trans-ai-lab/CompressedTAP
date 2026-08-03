#!/usr/bin/env python3
"""
Time-efficient path-column interface (binary CSR pool).

The CSV pool (link_ids as ';'-joined strings) is kept ONLY as a
human-readable interchange/debug view. All heavy operations use this
binary columnar format (single .npz, uncompressed for speed):

  indptr   int64  (n_paths+1)   CSR row pointer into `links`
  links    int32  (nnz)         kernel link_id sequence, concatenated
  o_zone   int32  (n_paths)
  d_zone   int32  (n_paths)
  x0       float64(n_paths)     nominal flow (baseline store; 0 for latent)
  cost     float64(n_paths)     base cost (NaN ok)
  source   int16  (n_paths) + source_names (list)  provenance

Speed principles:
  * CSV -> arrays via ONE str.cat + ONE C-level split (no per-row Python).
  * dedup via vectorized composite keys (od, len, first, last, str-hash) —
    no giant-string set operations, memory bounded per scenario.
  * B / A sparse matrices built directly from the CSR arrays (no loops).
"""
import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


# ---------------------------------------------------------------- parsing
def strings_to_csr(link_str: pd.Series):
    """';'-joined id strings -> (indptr int64, links int32). Vectorized."""
    s = link_str.astype(str).str.strip(';')
    counts = (s.str.count(';') + 1).to_numpy(dtype=np.int64).copy()
    counts[s.str.len().to_numpy() == 0] = 0
    indptr = np.zeros(len(s) + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    blob = s.str.cat(sep=';')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        links = np.fromstring(blob, dtype=np.int64, sep=';').astype(np.int32)
    assert len(links) == indptr[-1], 'CSR parse mismatch'
    return indptr, links


def row_keys(o, d, indptr, links, str_hash):
    """Composite dedup key arrays — collision-safe in practice."""
    n = len(o)
    lens = (indptr[1:] - indptr[:-1]).astype(np.int64)
    first = np.where(lens > 0, links[np.minimum(indptr[:-1], len(links) - 1)], -1)
    last = np.where(lens > 0, links[np.maximum(indptr[1:] - 1, 0)], -1)
    k = np.empty(n, dtype=[('o', np.int32), ('d', np.int32),
                           ('L', np.int32), ('f', np.int32),
                           ('l', np.int32), ('h', np.int64)])
    k['o'], k['d'] = o, d
    k['L'], k['f'], k['l'], k['h'] = lens, first, last, str_hash
    return k


class PoolBuilder:
    """Accumulate scenario route sets with streaming dedup."""

    def __init__(self):
        self.chunks = []          # per-scenario dicts of arrays
        self.seen = None          # structured key array of kept rows

    def add_route_csv(self, folder, source, chunksize=2_000_000):
        """Streams the route store in chunks (multi-GB regional/Philadelphia
        CSVs); each chunk dedups against everything seen so far."""
        f = os.path.join(folder, 'route_assignment.csv')
        if not os.path.exists(f):
            return 0
        hdr = pd.read_csv(f, nrows=0)
        want = {'o_zone_id', 'd_zone_id', 'link_ids', 'volume',
                'total_travel_time'}
        use = [c for c in hdr.columns if c.strip() in want]
        total = 0
        for r in pd.read_csv(f, usecols=use, low_memory=False,
                             chunksize=chunksize):
            r.columns = [c.strip() for c in r.columns]
            r = r[r.link_ids.astype(str).str.len() > 0]
            total += self._add_frame(r, source)
        return total

    def _add_frame(self, r, source):
        indptr, links = strings_to_csr(r.link_ids)
        h = r.link_ids.map(hash).to_numpy(dtype=np.int64)
        o = r.o_zone_id.to_numpy(dtype=np.int32)
        d = r.d_zone_id.to_numpy(dtype=np.int32)
        keys = row_keys(o, d, indptr, links, h)

        if self.seen is None:
            fresh = np.ones(len(keys), dtype=bool)
            # in-scenario dedup
            _, first_idx = np.unique(keys, return_index=True)
            fresh[:] = False
            fresh[first_idx] = True
        else:
            both = np.concatenate([self.seen, keys])
            _, first_idx = np.unique(both, return_index=True)
            fresh = np.zeros(len(both), dtype=bool)
            fresh[first_idx] = True
            fresh = fresh[len(self.seen):]
            # also dedup within this scenario against itself
            _, fi = np.unique(keys, return_index=True)
            self_first = np.zeros(len(keys), dtype=bool)
            self_first[fi] = True
            fresh &= self_first

        keep = np.where(fresh)[0]
        # slice CSR rows
        lens = indptr[1:] - indptr[:-1]
        new_indptr = np.zeros(len(keep) + 1, dtype=np.int64)
        np.cumsum(lens[keep], out=new_indptr[1:])
        pos = np.repeat(indptr[:-1][keep], lens[keep]) + \
            (np.arange(new_indptr[-1]) - np.repeat(new_indptr[:-1], lens[keep]))
        chunk = {
            'indptr': new_indptr, 'links': links[pos],
            'o': o[keep], 'd': d[keep],
            'x0': (r.volume.to_numpy(dtype=np.float64)[keep]
                   if 'volume' in r.columns else np.zeros(len(keep))),
            'cost': (r.total_travel_time.to_numpy(dtype=np.float64)[keep]
                     if 'total_travel_time' in r.columns
                     else np.full(len(keep), np.nan)),
            'source': source,
        }
        self.chunks.append(chunk)
        self.seen = (keys[keep] if self.seen is None
                     else np.concatenate([self.seen, keys[keep]]))
        return len(keep)

    def finalize(self):
        names = [c['source'] for c in self.chunks]
        n_tot = sum(len(c['o']) for c in self.chunks)
        nnz = sum(len(c['links']) for c in self.chunks)
        indptr = np.zeros(n_tot + 1, dtype=np.int64)
        links = np.empty(nnz, dtype=np.int32)
        o = np.empty(n_tot, np.int32)
        d = np.empty(n_tot, np.int32)
        x0 = np.empty(n_tot, np.float64)
        cost = np.empty(n_tot, np.float64)
        src = np.empty(n_tot, np.int16)
        p = q = 0
        for si, c in enumerate(self.chunks):
            k, z = len(c['o']), len(c['links'])
            indptr[p + 1:p + k + 1] = c['indptr'][1:] + q
            links[q:q + z] = c['links']
            o[p:p + k], d[p:p + k] = c['o'], c['d']
            x0[p:p + k], cost[p:p + k] = c['x0'], c['cost']
            src[p:p + k] = si
            p += k
            q += z
        return {'indptr': indptr, 'links': links, 'o_zone': o, 'd_zone': d,
                'x0': x0, 'cost': cost, 'source': src,
                'source_names': np.array(names)}


def save_pool(path_npz, pool):
    np.savez(path_npz, **pool)


def load_pool(path_npz):
    z = np.load(path_npz, allow_pickle=False)
    return {k: z[k] for k in z.files}


# ------------------------------------------------- problem construction
def build_problem_from_pool(dataset_dir, pool, baseline_source='s0_baseline',
                            x0_from_baseline_only=True):
    """Binary-pool equivalent of compressed_assignment.load_problem —
    B and A built directly from CSR arrays (no Python loops)."""
    import compressed_assignment as ca
    link = pd.read_csv(os.path.join(dataset_dir, 'link.csv'), low_memory=False)
    lut_size = int(link.link_id.max()) + 1
    lut = np.full(lut_size, -1, dtype=np.int64)
    lut[link.link_id.to_numpy()] = np.arange(len(link))
    cols = lut[pool['links']]
    assert (cols >= 0).all(), 'pool references unknown link ids'

    n = len(pool['o_zone'])
    m = len(link)
    B = csr_matrix((np.ones(len(cols)), cols, pool['indptr']), shape=(n, m))

    od = pool['o_zone'].astype(np.int64) * 100000 + pool['d_zone']
    od_unique, p2od = np.unique(od, return_inverse=True)
    n_od = len(od_unique)
    A = csr_matrix((np.ones(n), (p2od, np.arange(n))), shape=(n_od, n))

    dem = pd.read_csv(os.path.join(dataset_dir, 'demand.csv'))
    dem = dem.groupby(['o_zone_id', 'd_zone_id'], as_index=False).volume.sum()
    dk = dem.o_zone_id.to_numpy(np.int64) * 100000 + dem.d_zone_id.to_numpy()
    pos = np.searchsorted(od_unique, dk)
    ok = (pos < n_od)
    ok[ok] &= od_unique[pos[ok]] == dk[ok]
    dvec = np.zeros(n_od)
    dvec[pos[ok]] = dem.volume.to_numpy()[ok]

    x0 = pool['x0'].copy()
    if x0_from_baseline_only:
        names = list(pool['source_names'])
        bidx = [i for i, nm in enumerate(names) if nm == baseline_source]
        if bidx:  # all chunk indices of the baseline source (chunked builds)
            x0 = np.where(np.isin(pool['source'], bidx), x0, 0.0)
    x0 = np.nan_to_num(np.maximum(x0, 0.0))
    return {'B': B, 'A': A, 'd': dvec, 'x0': x0, 'p2od': p2od,
            'bpr': ca.BPR(link), 'n': n, 'm': m, 'n_od': n_od,
            'demand_coverage': float(dvec.sum() / dem.volume.sum()),
            'source': pool['source'],
            'source_names': list(pool['source_names'])}


def load_any(dataset_dir, pool_file, x0_from_baseline_only=True):
    """Problem loader preferring the binary pool. `pool_file` may be the CSV
    name — the .npz sibling is used when present (converted on first use)."""
    npz = pool_file[:-4] + '.npz' if pool_file.endswith('.csv') else pool_file
    p = os.path.join(dataset_dir, npz)
    if not os.path.exists(p):
        csv_pool_to_npz(dataset_dir, pool_file, npz)
    pool = load_pool(p)
    return build_problem_from_pool(dataset_dir, pool,
                                   x0_from_baseline_only=x0_from_baseline_only)


def csv_pool_to_npz(dataset_dir, csv_name, npz_name, chunksize=1_000_000):
    """One-time converter for existing CSV pools. Streams the CSV in chunks so
    memory stays bounded on large regional pools (1GB+ CSVs); each chunk's
    link strings are parsed to CSR independently, then concatenated."""
    path = os.path.join(dataset_dir, csv_name)
    ip_parts, lk_parts = [], []
    o_parts, d_parts, x0_parts, c_parts, s_parts = [], [], [], [], []
    name_to_id, names = {}, []
    base = 0
    for chunk in pd.read_csv(path, chunksize=chunksize):
        ip, lk = strings_to_csr(chunk.link_ids)
        ip_parts.append(ip[1:] + base)            # drop leading 0, offset
        base += ip[-1]
        lk_parts.append(lk.astype(np.int32))
        o_parts.append(chunk.o_zone_id.to_numpy(np.int32))
        d_parts.append(chunk.d_zone_id.to_numpy(np.int32))
        x0_parts.append(np.nan_to_num(chunk.volume_ref.to_numpy(np.float64)))
        c_parts.append(chunk.cost_base.to_numpy(np.float64))
        sid = np.empty(len(chunk), np.int16)
        for nm in chunk.source.unique():
            if nm not in name_to_id:
                name_to_id[nm] = len(names)
                names.append(nm)
        sid[:] = chunk.source.map(name_to_id).to_numpy()
        s_parts.append(sid)
    indptr = np.concatenate([[0]] + ip_parts).astype(np.int64)
    out = {'indptr': indptr,
           'links': np.concatenate(lk_parts),
           'o_zone': np.concatenate(o_parts),
           'd_zone': np.concatenate(d_parts),
           'x0': np.concatenate(x0_parts),
           'cost': np.concatenate(c_parts),
           'source': np.concatenate(s_parts),
           'source_names': np.array(names, dtype=str)}
    save_pool(os.path.join(dataset_dir, npz_name), out)
    return out
