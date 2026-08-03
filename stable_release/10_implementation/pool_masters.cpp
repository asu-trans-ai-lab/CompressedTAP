// pool_masters.cpp — C++ implementations of the pool masters for the
// systematic all-network comparison:
//   fw    Frank-Wolfe on the pool (AON + exact line search)
//   gp    GP per-OD Newton shifts (symmetric-difference denominators via an
//         O(nnz) per-OD stamp pass — no sparse-multiply overhead)
//   rsd   restricted simplicial decomposition (<=40 vertex columns,
//         projected-BB simplex master)
//   mlgp  Major-Latent GP (anchor-residual + explicit majors + softmax
//         latent bundle pi = softmax(theta0 + U z); needs mlgp_* exports)
// Reads the cpp_problem binary format of compressed_solver.cpp. Emits a
// per-iteration history CSV (elapsed_s,gap) and one SUMMARY line.
// build: g++ -O2 -fopenmp -o pool_masters pool_masters.cpp   (NO -march=native)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace std;
using clk = chrono::steady_clock;
static double secs(clk::time_point a, clk::time_point b) {
    return chrono::duration<double>(b - a).count();
}

template <typename T>
static vector<T> load_bin(const string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "missing %s\n", path.c_str()); exit(2); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    vector<T> v(sz / sizeof(T));
    if (fread(v.data(), 1, sz, f) != (size_t)sz) { exit(2); }
    fclose(f); return v;
}
static bool file_exists(const string& p) {
    FILE* f = fopen(p.c_str(), "rb"); if (f) { fclose(f); return true; }
    return false;
}
static long json_long(const string& js, const char* key) {
    size_t p = js.find(string("\"") + key + "\"");
    if (p == string::npos) return -1;
    p = js.find(':', p);
    return atol(js.c_str() + p + 1);
}

struct Prob {
    long n, m, n_od;
    vector<int64_t> Bp; vector<int32_t> Bi;   // path->links CSR
    vector<int32_t> p2od;
    vector<double> d, x0, t0, alpha, beta, cap;
    vector<double> v0;   // offset link flows (fixed single-path ODs), 0 if absent
    // per-OD path grouping
    vector<int64_t> Op; vector<int32_t> Oi;   // od -> path indices CSR
};

static Prob load(const string& dir) {
    Prob P;
    FILE* f = fopen((dir + "/meta.json").c_str(), "rb");
    if (!f) { fprintf(stderr, "no meta.json in %s\n", dir.c_str()); exit(2); }
    string js(8192, 0); js.resize(fread(&js[0], 1, 8192, f)); fclose(f);
    P.n = json_long(js, "n"); P.m = json_long(js, "m");
    P.n_od = json_long(js, "n_od");
    P.Bp = load_bin<int64_t>(dir + "/B_indptr.i64");
    P.Bi = load_bin<int32_t>(dir + "/B_indices.i32");
    P.p2od = load_bin<int32_t>(dir + "/p2od.i32");
    P.d = load_bin<double>(dir + "/dvec.f64");
    P.x0 = load_bin<double>(dir + "/x0.f64");
    P.t0 = load_bin<double>(dir + "/bpr_t0.f64");
    P.alpha = load_bin<double>(dir + "/bpr_alpha.f64");
    P.beta = load_bin<double>(dir + "/bpr_beta.f64");
    P.cap = load_bin<double>(dir + "/bpr_cap.f64");
    if (file_exists(dir + "/v0.f64")) {
        P.v0 = load_bin<double>(dir + "/v0.f64");
        double s0 = 0; for (double a : P.v0) s0 += a;
        fprintf(stderr, "v0 offset loaded (sum %.0f veh-links)\n", s0);
    } else P.v0.assign(P.m, 0.0);
    // group paths by OD
    vector<int64_t> cnt(P.n_od + 1, 0);
    for (long p = 0; p < P.n; ++p) cnt[P.p2od[p] + 1]++;
    P.Op.assign(P.n_od + 1, 0);
    for (long w = 0; w < P.n_od; ++w) P.Op[w + 1] = P.Op[w] + cnt[w + 1];
    P.Oi.assign(P.n, 0);
    vector<int64_t> cur(P.Op.begin(), P.Op.end() - 1);
    for (long p = 0; p < P.n; ++p) P.Oi[cur[P.p2od[p]]++] = (int32_t)p;
    return P;
}

// ------------------------------------------------------------- BPR helpers
static inline void bpr_t(const Prob& P, const vector<double>& v,
                         vector<double>& t) {
#pragma omp parallel for schedule(static)
    for (long a = 0; a < P.m; ++a) {
        double r = max(v[a], 0.0) / P.cap[a];
        t[a] = P.t0[a] * (1.0 + P.alpha[a] * pow(r, P.beta[a]));
    }
}
static inline void bpr_dt(const Prob& P, const vector<double>& v,
                          vector<double>& dt) {
#pragma omp parallel for schedule(static)
    for (long a = 0; a < P.m; ++a) {
        double r = max(v[a], 1e-10) / P.cap[a];
        dt[a] = P.t0[a] * P.alpha[a] * P.beta[a] / P.cap[a]
                * pow(r, P.beta[a] - 1.0);
    }
}
static double beckmann(const Prob& P, const vector<double>& v) {
    double s = 0.0;
#pragma omp parallel for reduction(+:s) schedule(static)
    for (long a = 0; a < P.m; ++a) {
        double vp = max(v[a], 0.0), r = vp / P.cap[a];
        s += P.t0[a] * vp + P.t0[a] * P.alpha[a] * P.cap[a]
             / (P.beta[a] + 1.0) * pow(r, P.beta[a] + 1.0) * 1.0;
    }
    return s;
}
static void link_flows(const Prob& P, const vector<double>& x,
                       vector<double>& v) {
    v = P.v0;                       // paper formulation: v = B'x + v0
    // serial scatter (deterministic); nnz-bound
    for (long p = 0; p < P.n; ++p) {
        double xp = x[p]; if (xp == 0.0) continue;
        for (int64_t k = P.Bp[p]; k < P.Bp[p + 1]; ++k) v[P.Bi[k]] += xp;
    }
}
static void path_costs(const Prob& P, const vector<double>& t,
                       vector<double>& c) {
#pragma omp parallel for schedule(dynamic, 4096)
    for (long p = 0; p < P.n; ++p) {
        double s = 0.0;
        for (int64_t k = P.Bp[p]; k < P.Bp[p + 1]; ++k) s += t[P.Bi[k]];
        c[p] = s;
    }
}
// per-OD argmin of c
static void od_argmin(const Prob& P, const vector<double>& c,
                      vector<int32_t>& pick) {
#pragma omp parallel for schedule(static)
    for (long w = 0; w < P.n_od; ++w) {
        int32_t best = -1; double bc = 1e300;
        for (int64_t k = P.Op[w]; k < P.Op[w + 1]; ++k) {
            int32_t p = P.Oi[k];
            if (c[p] < bc) { bc = c[p]; best = p; }
        }
        pick[w] = best;
    }
}
static double pool_gap(const Prob& P, const vector<double>& x,
                       const vector<double>& c, const vector<int32_t>& pick) {
    double cx = 0.0, cy = 0.0;
#pragma omp parallel for reduction(+:cx) schedule(static)
    for (long p = 0; p < P.n; ++p) cx += c[p] * x[p];
#pragma omp parallel for reduction(+:cy) schedule(static)
    for (long w = 0; w < P.n_od; ++w) cy += P.d[w] * c[pick[w]];
    return (cx - cy) / max(cy, 1e-12);
}

struct Hist { vector<double> t, g; };
static vector<double> g_vfinal;   // final link volumes of the last run
static void dump_hist(const string& path, const Hist& h) {
    FILE* f = fopen(path.c_str(), "wb");
    fprintf(f, "elapsed_s,gap\n");
    for (size_t i = 0; i < h.t.size(); ++i)
        fprintf(f, "%.6f,%.10e\n", h.t[i], h.g[i]);
    fclose(f);
}

// ------------------------------------------------------------------- FW
static void run_fw(const Prob& P, double tol, long iters, Hist& H,
                   double& obj, double& gap, long& it_out) {
    vector<double> x(P.n), v(P.m), t(P.m), c(P.n), dv(P.m);
    vector<int32_t> pick(P.n_od);
    // WARM start from the route-store nominal x0 — same start as GP and
    // ML-GP so the master comparison is start-matched (a uniform start
    // here raced GP's warm start and the curves were not comparable)
    {
        vector<double> s(P.n_od, 0.0);
        for (long p = 0; p < P.n; ++p)
            s[P.p2od[p]] += max(P.x0[p], 0.0);
        for (long w = 0; w < P.n_od; ++w) {
            long k0 = P.Op[w], k1 = P.Op[w + 1];
            if (s[w] > 1e-9) {
                double sc = P.d[w] / s[w];
                for (long k = k0; k < k1; ++k)
                    x[P.Oi[k]] = max(P.x0[P.Oi[k]], 0.0) * sc;
            } else {
                double f = P.d[w] / max<double>(k1 - k0, 1);
                for (long k = k0; k < k1; ++k) x[P.Oi[k]] = f;
            }
        }
    }
    link_flows(P, x, v);
    auto t0c = clk::now(); long it = 0; gap = 1e9;
    for (it = 1; it <= iters; ++it) {
        bpr_t(P, v, t);
        path_costs(P, t, c);
        od_argmin(P, c, pick);
        gap = pool_gap(P, x, c, pick);
        H.t.push_back(secs(t0c, clk::now())); H.g.push_back(gap);
        if (gap < tol) break;
        // dv = B'(y - x) with y = AON on picks
        fill(dv.begin(), dv.end(), 0.0);
        for (long w = 0; w < P.n_od; ++w) {
            int32_t p = pick[w];
            for (int64_t k = P.Bp[p]; k < P.Bp[p + 1]; ++k)
                dv[P.Bi[k]] += P.d[w];
        }
        for (long a = 0; a < P.m; ++a) dv[a] -= v[a];
        // exact line search (bisection on directional derivative)
        double lo = 0.0, hi = 1.0;
        vector<double> vt(P.m), tt(P.m);
        for (int b = 0; b < 40; ++b) {
            double mid = 0.5 * (lo + hi);
            for (long a = 0; a < P.m; ++a) vt[a] = v[a] + mid * dv[a];
            bpr_t(P, vt, tt);
            double dd = 0.0;
#pragma omp parallel for reduction(+:dd) schedule(static)
            for (long a = 0; a < P.m; ++a) dd += tt[a] * dv[a];
            if (dd > 0) hi = mid; else lo = mid;
        }
        double al = 0.5 * (lo + hi);
#pragma omp parallel for schedule(static)
        for (long p = 0; p < P.n; ++p) x[p] *= (1.0 - al);
        for (long w = 0; w < P.n_od; ++w) {
            int32_t p = pick[w]; x[p] += al * P.d[w];
        }
        for (long a = 0; a < P.m; ++a) v[a] += al * dv[a];
    }
    obj = beckmann(P, v); it_out = it; g_vfinal = v;
}

// ------------------------------------------------------------------- GP
// symmetric-difference denominators via per-OD anchor stamps: O(nnz)/iter
static void run_gp(const Prob& P, double tol, long iters, Hist& H,
                   double& obj, double& gap, long& it_out) {
    vector<double> x(P.n), v(P.m), t(P.m), dt(P.m), c(P.n);
    vector<int32_t> pick(P.n_od);
    vector<int32_t> stamp(P.m, -1); vector<double> adt(P.m, 0.0);
    // WARM start from the route-store nominal x0 (per-OD rescaled to
    // demand; uniform fallback where the store has no flow). A uniform
    // start over-concentrates on low-capacity links at metro scale and
    // the pair-Newton step cannot recover ((v/c)^beta overload regime).
    {
        vector<double> s(P.n_od, 0.0);
        for (long p = 0; p < P.n; ++p)
            s[P.p2od[p]] += max(P.x0[p], 0.0);
        for (long w = 0; w < P.n_od; ++w) {
            long k0 = P.Op[w], k1 = P.Op[w + 1];
            if (s[w] > 1e-9) {
                double sc = P.d[w] / s[w];
                for (long k = k0; k < k1; ++k)
                    x[P.Oi[k]] = max(P.x0[P.Oi[k]], 0.0) * sc;
            } else {
                double f = P.d[w] / max<double>(k1 - k0, 1);
                for (long k = k0; k < k1; ++k) x[P.Oi[k]] = f;
            }
        }
    }
    link_flows(P, x, v);
    auto t0c = clk::now(); long it = 0; gap = 1e9; double step = 1.0;
    vector<double> xn(P.n), vn(P.m);
    for (it = 1; it <= iters; ++it) {
        bpr_t(P, v, t); bpr_dt(P, v, dt);
        path_costs(P, t, c);
        od_argmin(P, c, pick);
        gap = pool_gap(P, x, c, pick);
        H.t.push_back(secs(t0c, clk::now())); H.g.push_back(gap);
        if (gap < tol) break;
        double f0 = beckmann(P, v);
        bool ok = false;
        for (int tries = 0; tries < 12 && !ok; ++tries) {
            xn = x;
            // per-OD: stamp anchor links with dt, shift each nonbasic;
            // stamps RESET after each OD (stale stamps across iterations
            // would corrupt cross-terms when anchors change)
            for (long w = 0; w < P.n_od; ++w) {
                int32_t pb = pick[w];
                double sb = 0.0;
                for (int64_t k = P.Bp[pb]; k < P.Bp[pb + 1]; ++k) {
                    stamp[P.Bi[k]] = (int32_t)w;
                    adt[P.Bi[k]] = dt[P.Bi[k]];
                    sb += dt[P.Bi[k]];
                }
                double moved = 0.0;
                for (int64_t k = P.Op[w]; k < P.Op[w + 1]; ++k) {
                    int32_t p = P.Oi[k];
                    if (p == pb || x[p] <= 0.0) continue;
                    double sp = 0.0, cross = 0.0;
                    for (int64_t kk = P.Bp[p]; kk < P.Bp[p + 1]; ++kk) {
                        int32_t a = P.Bi[kk];
                        sp += dt[a];
                        if (stamp[a] == (int32_t)w) cross += adt[a];
                    }
                    double den = max(sp + sb - 2.0 * cross, 1e-12);
                    double dlt = step * (c[p] - c[pb]) / den;
                    if (dlt <= 0.0) continue;
                    dlt = min(dlt, x[p]);
                    xn[p] = x[p] - dlt; moved += dlt;
                }
                xn[pb] = x[pb] + moved;
                for (int64_t k = P.Bp[pb]; k < P.Bp[pb + 1]; ++k)
                    stamp[P.Bi[k]] = -1;
            }
            link_flows(P, xn, vn);
            if (beckmann(P, vn) <= f0 + 1e-9 * max(1.0, fabs(f0))) ok = true;
            else step *= 0.5;
        }
        if (!ok) break;
        swap(x, xn); swap(v, vn);
        step = min(1.0, step * 1.6);
    }
    obj = beckmann(P, v); it_out = it; g_vfinal = v;
}

// ------------------------------------------------------------------ RSD
static void run_rsd(const Prob& P, double tol, long outers, int max_cols,
                    Hist& H, double& obj, double& gap, long& it_out) {
    vector<double> v(P.m), t(P.m), c(P.n);
    vector<int32_t> pick(P.n_od);
    vector<vector<double>> V;               // vertex link-flow columns
    vector<vector<int32_t>> picks;
    vector<double> lam;
    // first vertex: AON at free flow
    fill(v.begin(), v.end(), 0.0);
    bpr_t(P, v, t); path_costs(P, t, c); od_argmin(P, c, pick);
    {
        vector<double> col(P.m, 0.0);
        for (long w = 0; w < P.n_od; ++w) {
            int32_t p = pick[w];
            for (int64_t k = P.Bp[p]; k < P.Bp[p + 1]; ++k)
                col[P.Bi[k]] += P.d[w];
        }
        V.push_back(col); picks.push_back(pick); lam.assign(1, 1.0);
    }
    auto t0c = clk::now(); long it = 0; gap = 1e9;
    vector<double> g;                        // K-dim gradient
    for (it = 1; it <= outers; ++it) {
        // master: projected-BB on simplex
        int K = (int)V.size();
        g.assign(K, 0.0);
        for (int mi = 0; mi < 150; ++mi) {
            fill(v.begin(), v.end(), 0.0);
            for (int j = 0; j < K; ++j) {
                double lj = lam[j]; if (lj == 0.0) continue;
                const vector<double>& col = V[j];
                for (long a = 0; a < P.m; ++a) v[a] += lj * col[a];
            }
            bpr_t(P, v, t);
            for (int j = 0; j < K; ++j) {
                double s = 0.0;
                const vector<double>& col = V[j];
#pragma omp parallel for reduction(+:s) schedule(static)
                for (long a = 0; a < P.m; ++a) s += col[a] * t[a];
                g[j] = s;
            }
            double gn = 0.0; for (int j = 0; j < K; ++j) gn += g[j] * g[j];
            double stp = 1.0 / (sqrt(gn) + 1e-12);
            // project lam - stp*g onto simplex
            vector<double> w0(K);
            for (int j = 0; j < K; ++j) w0[j] = lam[j] - stp * g[j];
            vector<double> u(w0); sort(u.begin(), u.end(), greater<double>());
            double css = 0.0, thr = 0.0; int rho = 0;
            for (int j = 0; j < K; ++j) {
                css += u[j];
                if (u[j] > (css - 1.0) / (j + 1)) { rho = j; thr = (css - 1.0) / (j + 1); }
                else css -= u[j];
            }
            (void)rho;
            double delta = 0.0;
            for (int j = 0; j < K; ++j) {
                double nl = max(w0[j] - thr, 0.0);
                delta += fabs(nl - lam[j]); lam[j] = nl;
            }
            if (delta < 1e-12) break;
        }
        fill(v.begin(), v.end(), 0.0);
        for (int j = 0; j < K; ++j) {
            double lj = lam[j]; if (lj == 0.0) continue;
            const vector<double>& col = V[j];
            for (long a = 0; a < P.m; ++a) v[a] += lj * col[a];
        }
        bpr_t(P, v, t); path_costs(P, t, c); od_argmin(P, c, pick);
        // gap on reconstructed x implied by (lam, picks)
        double cy = 0.0, cx = 0.0;
        for (long w = 0; w < P.n_od; ++w) cy += P.d[w] * c[pick[w]];
        for (int j = 0; j < K; ++j) {
            if (lam[j] == 0.0) continue;
            double s = 0.0;
            const vector<int32_t>& pk = picks[j];
#pragma omp parallel for reduction(+:s) schedule(static)
            for (long w = 0; w < P.n_od; ++w) s += P.d[w] * c[pk[w]];
            cx += lam[j] * s;
        }
        gap = (cx - cy) / max(cy, 1e-12);
        H.t.push_back(secs(t0c, clk::now())); H.g.push_back(gap);
        if (gap < tol) break;
        // add vertex
        vector<double> col(P.m, 0.0);
        for (long w = 0; w < P.n_od; ++w) {
            int32_t p = pick[w];
            for (int64_t k = P.Bp[p]; k < P.Bp[p + 1]; ++k)
                col[P.Bi[k]] += P.d[w];
        }
        V.push_back(col); picks.push_back(pick);
        for (auto& l : lam) l *= (1.0 - 1e-3);
        lam.push_back(1e-3);
        if ((int)V.size() > max_cols) {
            int drop = 0; double bl = 1e300;
            for (int j = 0; j < (int)lam.size(); ++j)
                if (lam[j] < bl) { bl = lam[j]; drop = j; }
            V.erase(V.begin() + drop); picks.erase(picks.begin() + drop);
            lam.erase(lam.begin() + drop);
            double s = 0.0; for (double l : lam) s += l;
            for (auto& l : lam) l /= max(s, 1e-12);
        }
    }
    obj = beckmann(P, v); it_out = it; g_vfinal = v;
}

// ----------------------------------------------------------------- MLGP
static void run_mlgp(const string& dir, const Prob& P, double tol,
                     long iters, Hist& H, double& obj, double& gap,
                     long& it_out) {
    // exports from export_mlgp.py
    vector<int32_t> anchor = load_bin<int32_t>(dir + "/mlgp_anchor.i32");
    vector<uint8_t> major = load_bin<uint8_t>(dir + "/mlgp_major.u8");
    vector<int32_t> li = load_bin<int32_t>(dir + "/mlgp_latent_idx.i32");
    vector<double> U = load_bin<double>(dir + "/mlgp_U.f64");       // nL x r
    vector<double> th0 = load_bin<double>(dir + "/mlgp_theta0.f64");
    long nL = (long)li.size();
    long r = nL ? (long)(U.size() / nL) : 0;
    vector<int32_t> p2odL(nL);
    for (long i = 0; i < nL; ++i) p2odL[i] = P.p2od[li[i]];

    vector<double> fM(P.n, 0.0), mu(P.n_od, 0.0), z(r, 0.0);
    for (long p = 0; p < P.n; ++p) if (major[p]) fM[p] = max(P.x0[p], 0.0);
    for (long i = 0; i < nL; ++i) mu[p2odL[i]] += max(P.x0[li[i]], 0.0);
    for (long w = 0; w < P.n_od; ++w) mu[w] = min(mu[w], 0.9 * P.d[w]);
    {   // scale majors+mu into the demand
        vector<double> sM(P.n_od, 0.0);
        for (long p = 0; p < P.n; ++p) if (major[p]) sM[P.p2od[p]] += fM[p];
        for (long w = 0; w < P.n_od; ++w) {
            double tot = sM[w] + mu[w];
            if (tot > 0.98 * P.d[w]) {
                double s = 0.98 * P.d[w] / max(tot, 1e-12);
                mu[w] *= s;
                // scale this OD's majors
                for (int64_t k = P.Op[w]; k < P.Op[w + 1]; ++k)
                    if (major[P.Oi[k]]) fM[P.Oi[k]] *= s;
            }
        }
    }
    vector<double> pi(nL), theta(nL), x(P.n), v(P.m), t(P.m), dt(P.m),
        c(P.n), dbar(P.n_od), Smu(P.n_od), sM(P.n_od);
    vector<int32_t> pick(P.n_od);
    vector<int32_t> stamp(P.m, -1);
    auto softmax = [&](const vector<double>& zz) {
        // theta = th0 + U zz ; per-OD softmax
#pragma omp parallel for schedule(static)
        for (long i = 0; i < nL; ++i) {
            double s = th0[i];
            const double* Ui = &U[i * r];
            for (long j = 0; j < r; ++j) s += Ui[j] * zz[j];
            theta[i] = s;
        }
        vector<double> mx(P.n_od, -1e300), ss(P.n_od, 0.0);
        for (long i = 0; i < nL; ++i) mx[p2odL[i]] = max(mx[p2odL[i]], theta[i]);
        for (long i = 0; i < nL; ++i) {
            pi[i] = exp(theta[i] - mx[p2odL[i]]); ss[p2odL[i]] += pi[i];
        }
        for (long i = 0; i < nL; ++i) pi[i] /= max(ss[p2odL[i]], 1e-300);
    };
    auto assemble = [&]() {
        fill(x.begin(), x.end(), 0.0);
        for (long p = 0; p < P.n; ++p) if (major[p]) x[p] = fM[p];
        for (long i = 0; i < nL; ++i) x[li[i]] = mu[p2odL[i]] * pi[i];
        fill(sM.begin(), sM.end(), 0.0);
        for (long p = 0; p < P.n; ++p) sM[P.p2od[p]] += x[p];
        for (long w = 0; w < P.n_od; ++w)
            x[anchor[w]] += max(P.d[w] - sM[w], 0.0);
    };
    softmax(z); assemble(); link_flows(P, x, v);
    auto t0c = clk::now(); long it = 0; gap = 1e9; double step = 1.0;
    vector<double> fMn(P.n), mun(P.n_od), zn(r), gz(r);
    for (it = 1; it <= iters; ++it) {
        bpr_t(P, v, t); bpr_dt(P, v, dt);
        path_costs(P, t, c);
        od_argmin(P, c, pick);
        gap = pool_gap(P, x, c, pick);
        H.t.push_back(secs(t0c, clk::now())); H.g.push_back(gap);
        if (gap < tol) break;
        // anchor re-election among explicit (anchor U majors): ROLE SWAP
        for (long w = 0; w < P.n_od; ++w) {
            int32_t best = anchor[w]; double bc = c[best];
            for (int64_t k = P.Op[w]; k < P.Op[w + 1]; ++k) {
                int32_t p = P.Oi[k];
                if ((major[p] || p == anchor[w]) && c[p] < bc) {
                    bc = c[p]; best = p;
                }
            }
            if (best != anchor[w]) {
                fM[anchor[w]] = x[anchor[w]]; major[anchor[w]] = 1;
                fM[best] = 0.0; major[best] = 0;
                anchor[w] = best;
            }
        }
        double f0 = beckmann(P, v);
        bool ok = false;
        vector<double> piP(P.n, 0.0);
        for (long i = 0; i < nL; ++i) piP[li[i]] = pi[i];
        for (int tries = 0; tries < 12 && !ok; ++tries) {
            fMn = fM; mun = mu;
            // one stamped per-OD pass: majors get exact shifts AND the
            // latent bundle gets the EXACT expected symmetric-difference
            // denominator S_mu = sum_l pi_l (sp + sb - 2 cross). The
            // overlap-free overestimate was the Regional plateau suspect:
            // long metro paths share many anchor links, inflating S_mu
            // several-fold and starving the mu-steps.
            fill(dbar.begin(), dbar.end(), 0.0);
            fill(Smu.begin(), Smu.end(), 0.0);
            for (long w = 0; w < P.n_od; ++w) {
                int32_t pb = anchor[w];
                double sb = 0.0;
                for (int64_t k = P.Bp[pb]; k < P.Bp[pb + 1]; ++k) {
                    stamp[P.Bi[k]] = (int32_t)w; sb += dt[P.Bi[k]];
                }
                for (int64_t k = P.Op[w]; k < P.Op[w + 1]; ++k) {
                    int32_t p = P.Oi[k];
                    if (p == pb) continue;
                    double sp = 0.0, cross = 0.0;
                    for (int64_t kk = P.Bp[p]; kk < P.Bp[p + 1]; ++kk) {
                        int32_t a = P.Bi[kk];
                        sp += dt[a];
                        if (stamp[a] == (int32_t)w) cross += dt[a];
                    }
                    double den = max(sp + sb - 2.0 * cross, 1e-12);
                    if (major[p]) {
                        fMn[p] = max(0.0,
                                     fM[p] - step * (c[p] - c[pb]) / den);
                    } else {
                        dbar[w] += piP[p] * c[p];
                        Smu[w] += piP[p] * den;
                    }
                }
                for (int64_t k = P.Bp[pb]; k < P.Bp[pb + 1]; ++k)
                    stamp[P.Bi[k]] = -1;
                mun[w] = mu[w] - step * (dbar[w] - c[pb])
                         / max(Smu[w], 1e-12);
            }
            // z: softmax shape gradient, normalized step
            fill(gz.begin(), gz.end(), 0.0);
            for (long i = 0; i < nL; ++i) {
                double wgt = mu[p2odL[i]] * pi[i]
                             * (c[li[i]] - dbar[p2odL[i]]);
                const double* Ui = &U[i * r];
                for (long j = 0; j < r; ++j) gz[j] += Ui[j] * wgt;
            }
            double gn = 0.0; for (long j = 0; j < r; ++j) gn += gz[j] * gz[j];
            gn = sqrt(gn) + 1e-12;
            double zsc = 0.0; for (long j = 0; j < r; ++j) zsc += z[j] * z[j];
            zsc = sqrt(zsc) + 1.0;
            for (long j = 0; j < r; ++j) zn[j] = z[j] - step * zsc * gz[j] / gn;
            // caps: mu in [0, d - sum majors]
            fill(sM.begin(), sM.end(), 0.0);
            for (long p = 0; p < P.n; ++p) if (major[p]) sM[P.p2od[p]] += fMn[p];
            for (long w = 0; w < P.n_od; ++w)
                mun[w] = min(max(mun[w], 0.0), max(P.d[w] - sM[w], 0.0));
            // trial state
            vector<double> z_save = z, pi_save = pi, fM_save = fM,
                mu_save = mu, x_save = x, v_save = v;
            z = zn; softmax(z); fM = fMn; mu = mun;
            assemble(); link_flows(P, x, v);
            if (beckmann(P, v) <= f0 + 1e-9 * max(1.0, fabs(f0))) ok = true;
            else {
                z = z_save; pi = pi_save; fM = fM_save; mu = mu_save;
                x = x_save; v = v_save;
                step *= 0.5;
            }
        }
        if (!ok) break;
        step = min(1.0, step * 1.6);
    }
    obj = beckmann(P, v); it_out = it; g_vfinal = v;
}

// -------------------------------------------------------------- LATENT GP
// Network-wide feasible-column latent GP (production-spec P1/P2 network
// form): per OD one anchor path, a few explicit majors, ONE nonnegative
// blended atom over the remaining minors (weights from x0; feasibility by
// construction). Fresh-residual coordinate steps on sparse response
// columns, support-local true-descent acceptance, per-OD anchor-exhaustion
// pairwise swap, counted full-pool certificates with per-OD E1R refresh
// (promote the pool argmin as a new major when unrepresented). All the
// rules here were validated one-OD in latent_gp_prod.cpp.
static inline double beck_link(const Prob& P, long a, double v) {
    double vp = max(v, 0.0), r = vp / P.cap[a];
    return P.t0[a] * vp + P.t0[a] * P.alpha[a] * P.cap[a]
           / (P.beta[a] + 1.0) * pow(r, P.beta[a] + 1.0);
}
static void run_latgp(const Prob& P, double tol, long iters, Hist& H,
                      double& obj, double& gap, long& it_out) {
    const double RHO_M = 0.75;        // major coverage of non-anchor mass
    const int    MAJ0 = 6, MAJCAP = 24;
    // ---- warm start x from x0 (per-OD rescaled, uniform fallback)
    vector<double> x(P.n, 0.0);
    {
        vector<double> s(P.n_od, 0.0);
        for (long p = 0; p < P.n; ++p) s[P.p2od[p]] += max(P.x0[p], 0.0);
        for (long w = 0; w < P.n_od; ++w) {
            long k0 = P.Op[w], k1 = P.Op[w + 1];
            if (s[w] > 1e-9) {
                double sc = P.d[w] / s[w];
                for (long k = k0; k < k1; ++k)
                    x[P.Oi[k]] = max(P.x0[P.Oi[k]], 0.0) * sc;
            } else {
                double f = P.d[w] / max<double>(k1 - k0, 1);
                for (long k = k0; k < k1; ++k) x[P.Oi[k]] = f;
            }
        }
    }
    vector<double> v(P.m), cst(P.m), drv(P.m), c(P.n);
    link_flows(P, x, v);
    bpr_t(P, v, cst); bpr_dt(P, v, drv);
    // ---- per-OD split and sparse response columns
    vector<int32_t> aw(P.n_od);           // anchor path per OD
    vector<double> fanch(P.n_od);
    vector<int64_t> coff; vector<int32_t> clen;   // column SoA
    vector<int32_t> cids; vector<double> ccf, cflow;
    vector<int32_t> cod;                          // column -> OD
    vector<vector<int32_t>> odcols(P.n_od);
    vector<uint8_t> is_col(P.n, 0);
    vector<pair<int32_t, double>> tmp;
    auto push_response = [&](long w, const vector<pair<int32_t,double>>& pos,
                             double flow0) {
        // response = pos - a_anchor(w), merged sparse
        tmp.clear();
        for (auto& e : pos) tmp.push_back(e);
        int32_t ap = aw[w];
        for (int64_t k = P.Bp[ap]; k < P.Bp[ap + 1]; ++k)
            tmp.push_back({P.Bi[k], -1.0});
        sort(tmp.begin(), tmp.end());
        coff.push_back((int64_t)cids.size());
        size_t i = 0; long len = 0;
        while (i < tmp.size()) {
            int32_t a = tmp[i].first; double sc = 0;
            while (i < tmp.size() && tmp[i].first == a) sc += tmp[i++].second;
            if (fabs(sc) > 1e-12) { cids.push_back(a); ccf.push_back(sc); ++len; }
        }
        clen.push_back((int32_t)len);
        cflow.push_back(flow0);
        cod.push_back((int32_t)w);
        odcols[w].push_back((int32_t)(cflow.size() - 1));
    };
    {
        vector<int64_t> ord;
        vector<pair<int32_t,double>> pos;
        for (long w = 0; w < P.n_od; ++w) {
            long k0 = P.Op[w], k1 = P.Op[w + 1], kn = k1 - k0;
            // anchor = max-x path
            long best = P.Oi[k0]; double bx = -1;
            for (long k = k0; k < k1; ++k)
                if (x[P.Oi[k]] > bx) { bx = x[P.Oi[k]]; best = P.Oi[k]; }
            aw[w] = (int32_t)best; is_col[best] = 1;
            if (kn == 1) { fanch[w] = P.d[w]; continue; }
            // majors: largest-x paths covering RHO_M of non-anchor mass
            ord.clear();
            for (long k = k0; k < k1; ++k)
                if (P.Oi[k] != best) ord.push_back(P.Oi[k]);
            sort(ord.begin(), ord.end(),
                 [&](int64_t a2, int64_t b2) { return x[a2] > x[b2]; });
            double nonanc = P.d[w] - x[best], acc = 0;
            size_t nmaj = 0;
            for (; nmaj < ord.size() && (long)nmaj < MAJ0; ++nmaj) {
                if (nonanc > 1e-12 && acc >= RHO_M * nonanc) break;
                acc += x[ord[nmaj]];
            }
            for (size_t i2 = 0; i2 < nmaj; ++i2) {
                long p = ord[i2];
                pos.clear();
                for (int64_t k = P.Bp[p]; k < P.Bp[p + 1]; ++k)
                    pos.push_back({P.Bi[k], 1.0});
                push_response(w, pos, x[p]);
                is_col[p] = 1;
            }
            // atom over the remaining minors, weights from x (>=eps)
            double mu = 0, wsum = 0;
            for (size_t i2 = nmaj; i2 < ord.size(); ++i2) {
                mu += x[ord[i2]]; wsum += max(x[ord[i2]], 1e-12);
            }
            if (ord.size() > nmaj) {
                pos.clear();
                for (size_t i2 = nmaj; i2 < ord.size(); ++i2) {
                    long p = ord[i2];
                    double pw = max(x[p], 1e-12) / wsum;
                    for (int64_t k = P.Bp[p]; k < P.Bp[p + 1]; ++k)
                        pos.push_back({P.Bi[k], pw});
                }
                push_response(w, pos, mu);
            }
            fanch[w] = max(P.d[w] - x[best] - acc - mu, 0.0) + x[best];
            // note: acc+mu+x[best] may not equal d exactly after rescale;
            // fold any residual into the anchor (keeps Ax = d exact)
        }
    }
    long ncol = (long)cflow.size();
    fprintf(stderr, "latgp: %ld columns (%ld ODs), %.1fM support entries\n",
            ncol, P.n_od, cids.size() / 1e6);
    // ---- sweeps
    vector<int32_t> pick(P.n_od);
    auto t0c = clk::now(); long it = 0; gap = 1e9;
    long Rc = 10; double prev_cert = 1e300;
    vector<double> gj; vector<pair<int32_t,double>> mg;
    for (it = 1; it <= iters; ++it) {
        if ((it - 1) % Rc == 0) {
            // certificate: full-pool pricing (counted in the clock)
            bpr_t(P, v, cst);
            path_costs(P, cst, c);
            od_argmin(P, c, pick);
            double cx = 0;
            for (long w = 0; w < P.n_od; ++w) cx += P.d[w] * c[aw[w]];
            for (long j = 0; j < ncol; ++j) {
                if (cflow[j] == 0.0) continue;
                double s2 = 0;
                for (int64_t k2 = coff[j]; k2 < coff[j] + clen[j]; ++k2)
                    s2 += ccf[k2] * cst[cids[k2]];
                cx += cflow[j] * s2;
            }
            double cy = 0;
            for (long w = 0; w < P.n_od; ++w) cy += P.d[w] * c[pick[w]];
            gap = (cx - cy) / max(cy, 1e-12);
            H.t.push_back(secs(t0c, clk::now())); H.g.push_back(gap);
            if (gap < tol) break;
            // E1R refresh: promote unrepresented pool argmins as majors
            if (gap > 0.5 * prev_cert) {
                long promoted = 0;
                vector<pair<int32_t,double>> pos;
                for (long w = 0; w < P.n_od; ++w) {
                    int32_t p = pick[w];
                    if (is_col[p] || (long)odcols[w].size() >= MAJCAP)
                        continue;
                    pos.clear();
                    for (int64_t k = P.Bp[p]; k < P.Bp[p + 1]; ++k)
                        pos.push_back({P.Bi[k], 1.0});
                    push_response(w, pos, 0.0);
                    is_col[p] = 1; ++promoted;
                }
                ncol = (long)cflow.size();
                if (promoted)
                    fprintf(stderr, "  refresh it %ld: +%ld columns "
                            "(gap %.2e)\n", it, promoted, gap);
            }
            prev_cert = gap;
            Rc = gap > 1e-2 ? 10 : (gap > 1e-4 ? 5 : 2);
        }
        // one pass over all columns (fresh residuals, incremental updates)
        for (long j = 0; j < ncol; ++j) {
            long w = cod[j];
            double g = 0, S = 0;
            for (int64_t k2 = coff[j]; k2 < coff[j] + clen[j]; ++k2) {
                double r = ccf[k2]; long a = cids[k2];
                g += r * cst[a]; S += r * r * drv[a];
            }
            double d = -g / (S + 1e-12);
            if (d < -cflow[j]) d = -cflow[j];
            if (d > fanch[w]) d = fanch[w];
            if (d == 0.0) continue;
            int hv = 0;
            while (hv++ < 30) {
                double dF = 0;
                for (int64_t k2 = coff[j]; k2 < coff[j] + clen[j]; ++k2) {
                    long a = cids[k2];
                    dF += beck_link(P, a, v[a] + d * ccf[k2])
                          - beck_link(P, a, v[a]);
                }
                if (dF < -1e-13) break;
                d *= 0.5;
            }
            if (hv > 30 || d == 0.0) continue;
            cflow[j] += d; fanch[w] -= d;
            for (int64_t k2 = coff[j]; k2 < coff[j] + clen[j]; ++k2) {
                long a = cids[k2];
                v[a] += d * ccf[k2];
                double r = max(v[a], 0.0) / P.cap[a];
                cst[a] = P.t0[a] * (1.0 + P.alpha[a] * pow(r, P.beta[a]));
                double r2 = max(v[a], 1e-10) / P.cap[a];
                drv[a] = P.t0[a] * P.alpha[a] * P.beta[a] / P.cap[a]
                         * pow(r2, P.beta[a] - 1.0);
            }
        }
        // anchor-exhaustion pairwise swap, per OD with empty anchor
        for (long w = 0; w < P.n_od; ++w) {
            if (fanch[w] > 1e-12 || odcols[w].size() < 2) continue;
            long b = -1; double gb = 1e300;
            gj.assign(odcols[w].size(), 0.0);
            for (size_t i2 = 0; i2 < odcols[w].size(); ++i2) {
                long j = odcols[w][i2]; double s2 = 0;
                for (int64_t k2 = coff[j]; k2 < coff[j] + clen[j]; ++k2)
                    s2 += ccf[k2] * cst[cids[k2]];
                gj[i2] = s2;
                if (s2 < gb) { gb = s2; b = i2; }
            }
            if (b < 0) continue;
            long jb = odcols[w][b];
            for (size_t i2 = 0; i2 < odcols[w].size(); ++i2) {
                if ((long)i2 == b) continue;
                long j = odcols[w][i2];
                if (cflow[j] <= 0 || gj[i2] <= gb + 1e-12) continue;
                double Sj = 0, Sb = 0;
                for (int64_t k2 = coff[j]; k2 < coff[j] + clen[j]; ++k2)
                    Sj += ccf[k2] * ccf[k2] * drv[cids[k2]];
                for (int64_t k2 = coff[jb]; k2 < coff[jb] + clen[jb]; ++k2)
                    Sb += ccf[k2] * ccf[k2] * drv[cids[k2]];
                double del = (gj[i2] - gb) / (2.0 * (Sj + Sb) + 1e-12);
                if (del > cflow[j]) del = cflow[j];
                if (del <= 0) continue;
                mg.clear();
                for (int64_t k2 = coff[j]; k2 < coff[j] + clen[j]; ++k2)
                    mg.push_back({cids[k2], -ccf[k2]});
                for (int64_t k2 = coff[jb]; k2 < coff[jb] + clen[jb]; ++k2)
                    mg.push_back({cids[k2], ccf[k2]});
                sort(mg.begin(), mg.end());
                size_t wq = 0;
                for (size_t q = 0; q < mg.size();) {
                    int32_t a = mg[q].first; double s2 = 0;
                    while (q < mg.size() && mg[q].first == a)
                        s2 += mg[q++].second;
                    if (s2 != 0.0) mg[wq++] = {a, s2};
                }
                mg.resize(wq);
                int hv = 0; bool ok = false;
                while (hv++ < 30) {
                    double dF = 0;
                    for (auto& e : mg)
                        dF += beck_link(P, e.first, v[e.first] + del * e.second)
                              - beck_link(P, e.first, v[e.first]);
                    if (dF < -1e-13) { ok = true; break; }
                    del *= 0.5;
                }
                if (!ok) continue;
                cflow[j] -= del; cflow[jb] += del;
                for (auto& e : mg) {
                    long a = e.first;
                    v[a] += del * e.second;
                    double r = max(v[a], 0.0) / P.cap[a];
                    cst[a] = P.t0[a] * (1.0 + P.alpha[a] * pow(r, P.beta[a]));
                    double r2 = max(v[a], 1e-10) / P.cap[a];
                    drv[a] = P.t0[a] * P.alpha[a] * P.beta[a] / P.cap[a]
                             * pow(r2, P.beta[a] - 1.0);
                }
            }
        }
    }
    obj = beckmann(P, v); it_out = it; g_vfinal = v;
}


// ------------------------------------------------------- PE (pool)
// Strict path equilibration: per OD, one Newton swap from the current
// max-cost positive route to the min-cost route (symmetric-difference
// denominator), full pricing each iteration. Pool counterpart of the
// PE implementation in TAsK.
static void run_pe(const Prob& P, double tol, long iters, Hist& H,
                   double& obj, double& gap, long& it_out) {
    vector<double> x(P.n), v(P.m), t(P.m), dt(P.m), c(P.n);
    vector<int32_t> pick(P.n_od);
    {
        vector<double> s(P.n_od, 0.0);
        for (long p = 0; p < P.n; ++p) s[P.p2od[p]] += max(P.x0[p], 0.0);
        for (long w = 0; w < P.n_od; ++w) {
            long k0 = P.Op[w], k1 = P.Op[w + 1];
            if (s[w] > 1e-9) {
                double sc = P.d[w] / s[w];
                for (long k = k0; k < k1; ++k)
                    x[P.Oi[k]] = max(P.x0[P.Oi[k]], 0.0) * sc;
            } else {
                double f = P.d[w] / max<double>(k1 - k0, 1);
                for (long k = k0; k < k1; ++k) x[P.Oi[k]] = f;
            }
        }
    }
    link_flows(P, x, v);
    auto t0c = clk::now(); long it = 0; gap = 1e9;
    vector<int32_t> stamp(P.m, -1);
    for (it = 1; it <= iters; ++it) {
        bpr_t(P, v, t); bpr_dt(P, v, dt);
        path_costs(P, t, c);
        od_argmin(P, c, pick);
        gap = pool_gap(P, x, c, pick);
        H.t.push_back(secs(t0c, clk::now())); H.g.push_back(gap);
        if (gap < tol) break;
        for (long w = 0; w < P.n_od; ++w) {
            // FRESH per-OD costs from incrementally maintained times
            // (iteration-top prices go stale after earlier ODs move flow
            // and mispair max/min, starving late ODs)
            int32_t pmin = -1, pmax = -1;
            double cmin = 1e300, cmax = -1e300;
            for (int64_t k = P.Op[w]; k < P.Op[w + 1]; ++k) {
                int32_t p = P.Oi[k];
                double s2 = 0.0;
                for (int64_t k2 = P.Bp[p]; k2 < P.Bp[p + 1]; ++k2)
                    s2 += t[P.Bi[k2]];
                if (s2 < cmin) { cmin = s2; pmin = p; }
                if (x[p] > 0.0 && s2 > cmax) { cmax = s2; pmax = p; }
            }
            if (pmax < 0 || pmax == pmin) continue;
            double g = cmax - cmin;
            if (g <= 0.0) continue;
            double S = 0.0;
            for (int64_t k2 = P.Bp[pmax]; k2 < P.Bp[pmax + 1]; ++k2)
                stamp[P.Bi[k2]] = (int32_t)w;
            for (int64_t k2 = P.Bp[pmin]; k2 < P.Bp[pmin + 1]; ++k2) {
                long a = P.Bi[k2];
                if (stamp[a] == (int32_t)w) stamp[a] = -2;  // shared
                else S += dt[a];
            }
            for (int64_t k2 = P.Bp[pmax]; k2 < P.Bp[pmax + 1]; ++k2)
                if (stamp[P.Bi[k2]] == (int32_t)w) S += dt[P.Bi[k2]];
            double del = g / (S + 1e-12);
            if (del > x[pmax]) del = x[pmax];
            if (del <= 0.0) continue;
            // true-descent acceptance over the symmetric difference
            // (stale sweep-top prices overshoot otherwise; stamp==-2
            // marks links shared by both routes, which cancel)
            auto dF = [&](double dd) {
                double s2 = 0.0;
                for (int64_t k2 = P.Bp[pmax]; k2 < P.Bp[pmax + 1]; ++k2) {
                    long a = P.Bi[k2];
                    if (stamp[a] == -2) continue;
                    s2 += beck_link(P, a, v[a] - dd) - beck_link(P, a, v[a]);
                }
                for (int64_t k2 = P.Bp[pmin]; k2 < P.Bp[pmin + 1]; ++k2) {
                    long a = P.Bi[k2];
                    if (stamp[a] == -2) continue;
                    s2 += beck_link(P, a, v[a] + dd) - beck_link(P, a, v[a]);
                }
                return s2;
            };
            int hv = 0;
            while (dF(del) > -1e-12 && hv++ < 30) del *= 0.5;
            if (hv > 30 || del <= 0.0) continue;
            x[pmax] -= del; x[pmin] += del;
            for (int64_t k2 = P.Bp[pmax]; k2 < P.Bp[pmax + 1]; ++k2) {
                long a = P.Bi[k2];
                v[a] -= del;
                double r = max(v[a], 0.0) / P.cap[a];
                t[a] = P.t0[a] * (1.0 + P.alpha[a] * pow(r, P.beta[a]));
                double r2 = max(v[a], 1e-10) / P.cap[a];
                dt[a] = P.t0[a] * P.alpha[a] * P.beta[a] / P.cap[a]
                        * pow(r2, P.beta[a] - 1.0);
            }
            for (int64_t k2 = P.Bp[pmin]; k2 < P.Bp[pmin + 1]; ++k2) {
                long a = P.Bi[k2];
                v[a] += del;
                double r = max(v[a], 0.0) / P.cap[a];
                t[a] = P.t0[a] * (1.0 + P.alpha[a] * pow(r, P.beta[a]));
                double r2 = max(v[a], 1e-10) / P.cap[a];
                dt[a] = P.t0[a] * P.alpha[a] * P.beta[a] / P.cap[a]
                        * pow(r2, P.beta[a] - 1.0);
            }
        }
    }
    obj = beckmann(P, v); it_out = it; g_vfinal = v;
}

// -------------------------------------------- CFW / BFW on the pool
// Conjugate and bi-conjugate Frank-Wolfe with direction coefficients
// transcribed verbatim from TAsK (LinkFlowsCFW / LinkFlowsBFW,
// github.com/olga-perederieieva/TAsK); AON targets restricted to the pool.
static void run_xfw(const Prob& P, double tol, long iters, Hist& H,
                    double& obj, double& gap, long& it_out, bool bi) {
    const double ZF = 1e-12;
    vector<double> x(P.n), v(P.m), t(P.m), dt(P.m), c(P.n), dv(P.m);
    vector<double> yAuxV(P.m), sCFWV(P.m), sBFWV(P.m);
    vector<double> yAuxX(P.n), sCFWX(P.n), sBFWX(P.n);
    vector<int32_t> pick(P.n_od);
    {
        vector<double> s(P.n_od, 0.0);
        for (long p = 0; p < P.n; ++p) s[P.p2od[p]] += max(P.x0[p], 0.0);
        for (long w = 0; w < P.n_od; ++w) {
            long k0 = P.Op[w], k1 = P.Op[w + 1];
            if (s[w] > 1e-9) {
                double sc = P.d[w] / s[w];
                for (long k = k0; k < k1; ++k)
                    x[P.Oi[k]] = max(P.x0[P.Oi[k]], 0.0) * sc;
            } else {
                double f = P.d[w] / max<double>(k1 - k0, 1);
                for (long k = k0; k < k1; ++k) x[P.Oi[k]] = f;
            }
        }
    }
    link_flows(P, x, v);
    auto t0c = clk::now(); long it = 0; gap = 1e9;
    long nbCalls = 0; double stepPrev = 0.0, stepPrevPrev = 0.0;
    for (it = 1; it <= iters; ++it) {
        bpr_t(P, v, t); bpr_dt(P, v, dt);
        path_costs(P, t, c);
        od_argmin(P, c, pick);
        gap = pool_gap(P, x, c, pick);
        H.t.push_back(secs(t0c, clk::now())); H.g.push_back(gap);
        if (gap < tol) break;
        fill(yAuxV.begin(), yAuxV.end(), 0.0);
        fill(yAuxX.begin(), yAuxX.end(), 0.0);
        for (long w = 0; w < P.n_od; ++w) {
            int32_t p = pick[w];
            yAuxX[p] += P.d[w];
            for (int64_t k = P.Bp[p]; k < P.Bp[p + 1]; ++k)
                yAuxV[P.Bi[k]] += P.d[w];
        }
        if (nbCalls == 0) {
            sCFWV = yAuxV; sCFWX = yAuxX;
            fill(sBFWV.begin(), sBFWV.end(), 0.0);
            fill(sBFWX.begin(), sBFWX.end(), 0.0);
        } else if (!bi || nbCalls == 1) {
            if (bi) { sBFWV = sCFWV; sBFWX = sCFWX; }
            double num = 0.0, den = 0.0;
            for (long a = 0; a < P.m; ++a) {
                double dkfw = yAuxV[a] - v[a];
                double dbar = sCFWV[a] - v[a];
                num += dbar * dt[a] * dkfw;
                den += dbar * dt[a] * (dkfw - dbar);
            }
            double al = (den != 0.0) ? num / den : 0.0;
            if (al > 1.0 - ZF) al = 1.0 - ZF;
            if (al < 0.0) al = 0.0;
            for (long a = 0; a < P.m; ++a)
                sCFWV[a] = al * sCFWV[a] + (1.0 - al) * yAuxV[a];
            for (long p = 0; p < P.n; ++p)
                sCFWX[p] = al * sCFWX[p] + (1.0 - al) * yAuxX[p];
        } else {
            double beta0 = 1.0, beta1 = 0.0, beta2 = 0.0;
            if (fabs(1.0 - stepPrev) > ZF && fabs(1.0 - stepPrevPrev) > ZF) {
                double nk = 0, dk = 0, nnk = 0, ddk = 0;
                for (long a = 0; a < P.m; ++a) {
                    double der = dt[a];
                    double dir_fw = yAuxV[a] - v[a];
                    double dir_2 = stepPrev * sCFWV[a] - v[a]
                                   + (1 - stepPrev) * sBFWV[a];
                    double dir_1 = sCFWV[a] - v[a];
                    nk += dir_2 * der * dir_fw;
                    dk += dir_2 * der * (dir_2 - dir_1) / (1 - stepPrev);
                    nnk += dir_1 * der * dir_fw;
                    ddk += dir_1 * dir_1 * der;
                }
                if (fabs(dk) > ZF && fabs(ddk) > ZF) {
                    double mu = -nk / dk;
                    double eta = -nnk / ddk + (mu * stepPrev) / (1.0 - stepPrev);
                    beta0 = 1.0 / (1.0 + mu + eta);
                    if (beta0 >= 0.0 && beta0 <= 1.0 - ZF) {
                        beta1 = eta * beta0; beta2 = mu * beta0;
                        if (beta1 < 0.0 || beta1 > 1.0 - ZF ||
                            beta2 < 0.0 || beta2 > 1.0 - ZF)
                            { beta0 = 1.0; beta1 = 0.0; beta2 = 0.0; }
                    } else { beta0 = 1.0; beta1 = 0.0; beta2 = 0.0; }
                }
            }
            for (long a = 0; a < P.m; ++a) {
                double pv = beta0 * yAuxV[a] + beta1 * sCFWV[a]
                            + beta2 * sBFWV[a];
                sBFWV[a] = sCFWV[a]; sCFWV[a] = pv;
            }
            for (long p = 0; p < P.n; ++p) {
                double pv = beta0 * yAuxX[p] + beta1 * sCFWX[p]
                            + beta2 * sBFWX[p];
                sBFWX[p] = sCFWX[p]; sCFWX[p] = pv;
            }
        }
        ++nbCalls;
        for (long a = 0; a < P.m; ++a) dv[a] = sCFWV[a] - v[a];
        double lo = 0.0, hi = 1.0;
        vector<double> vt(P.m), tt(P.m);
        for (int b = 0; b < 40; ++b) {
            double mid = 0.5 * (lo + hi);
            for (long a = 0; a < P.m; ++a) vt[a] = v[a] + mid * dv[a];
            bpr_t(P, vt, tt);
            double dd = 0.0;
            for (long a = 0; a < P.m; ++a) dd += tt[a] * dv[a];
            if (dd > 0) hi = mid; else lo = mid;
        }
        double al2 = 0.5 * (lo + hi);
        for (long p = 0; p < P.n; ++p)
            x[p] = (1.0 - al2) * x[p] + al2 * sCFWX[p];
        for (long a = 0; a < P.m; ++a) v[a] += al2 * dv[a];
        stepPrevPrev = stepPrev; stepPrev = al2;
    }
    obj = beckmann(P, v); it_out = it; g_vfinal = v;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <problem_dir> fw|gp|rsd|mlgp|latgp "
                "[tol=1e-4] [iters] [hist_csv]\n", argv[0]);
        return 1;
    }
    string dir = argv[1], master = argv[2];
    double tol = argc > 3 ? atof(argv[3]) : 1e-4;
    long iters = argc > 4 ? atol(argv[4]) : 0;
    string hist = argc > 5 ? argv[5] : "";
    string vdump = argc > 6 ? argv[6] : "";
    auto tl0 = clk::now();
    Prob P = load(dir);
    double load_s = secs(tl0, clk::now());
    fprintf(stderr, "loaded n=%ld m=%ld n_od=%ld (%.1fs)\n",
            P.n, P.m, P.n_od, load_s);
    Hist H; double obj = 0, gap = 1e9; long it = 0;
    auto ts = clk::now();
    if (master == "fw")  run_fw(P, tol, iters ? iters : 4000, H, obj, gap, it);
    else if (master == "gp") run_gp(P, tol, iters ? iters : 500, H, obj, gap, it);
    else if (master == "rsd") run_rsd(P, tol, iters ? iters : 300, 40, H, obj, gap, it);
    else if (master == "mlgp") run_mlgp(dir, P, tol, iters ? iters : 2000, H, obj, gap, it);
    else if (master == "latgp") run_latgp(P, tol, iters ? iters : 2000, H, obj, gap, it);
    else if (master == "pe") run_pe(P, tol, iters ? iters : 4000, H, obj, gap, it);
    else if (master == "cfw") run_xfw(P, tol, iters ? iters : 4000, H, obj, gap, it, false);
    else if (master == "bfw") run_xfw(P, tol, iters ? iters : 4000, H, obj, gap, it, true);
    else { fprintf(stderr, "unknown master\n"); return 1; }
    double solve_s = secs(ts, clk::now());
    if (!hist.empty()) dump_hist(hist, H);
    if (!vdump.empty() && !g_vfinal.empty()) {
        FILE* fo = fopen(vdump.c_str(), "wb");
        if (fo) { fwrite(g_vfinal.data(), sizeof(double), g_vfinal.size(), fo); fclose(fo); }
    }
    printf("SUMMARY,%s,%s,%.3f,%.6e,%ld,%.2f\n",
           master.c_str(), dir.c_str(), solve_s, gap, it, obj);
    return 0;
}
