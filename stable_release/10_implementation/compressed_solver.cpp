// ============================================================================
// compressed_solver.cpp — same-algorithm C++ comparison of FULL vs COMPRESSED
// path-based traffic assignment (SPG-ALM), with C2 constraint regimes.
//
// Reads the binary problem dump written by python/export_problem.py.
// Inner solver: nonmonotone spectral projected gradient (Barzilai-Borwein,
// Grippo window) — identical machinery for FULL and COMPRESSED so the timing
// ratio is free of solver-choice and language confounds. OpenMP kernels.
//
// build:  g++ -O3 -fopenmp -march=native -o compressed_solver compressed_solver.cpp
// run:    compressed_solver <problem_dir> full|soft|screen|hard [max_outer]
// ============================================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

using std::vector;
using std::string;
using clk = std::chrono::steady_clock;

static double secs(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

template <typename T>
static vector<T> load_bin(const string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    vector<T> v(sz / sizeof(T));
    if (fread(v.data(), 1, sz, f) != (size_t)sz) { fprintf(stderr, "short read %s\n", path.c_str()); exit(1); }
    fclose(f);
    return v;
}

static long meta_int(const string& js, const char* key, bool required = true,
                     long dflt = 0) {
    string pat = string("\"") + key + "\":";
    size_t p = js.find(pat);
    if (p == string::npos) {
        if (required) { fprintf(stderr, "meta key %s missing\n", key); exit(1); }
        return dflt;
    }
    return atol(js.c_str() + p + pat.size());
}

// ---------------------------------------------------------------- problem
struct Problem {
    long n, m, n_od, nnz, r, n_major, n_minor;
    vector<int64_t> Bp;      // CSR indptr (paths)
    vector<int32_t> Bi;      // link indices
    vector<int32_t> p2od;
    vector<double> d, x0;
    vector<uint8_t> major;
    vector<double> t0, alpha, beta, cap;
    vector<double> v0;       // constant background link flow (singleton ODs); optional blob
    // compressed blocks
    vector<double> U, D, M, x0m, d_eff, v_base;   // row-major
    vector<int32_t> majid;   // path ids of majors (order = y order)
    vector<int32_t> minid;   // path ids of minors (order = U rows)
    // svd-newton blocks (full-space Woodbury)
    vector<double> Pmat, Qmat, sB;   // P: n x r, Q: m x r (row-major), sB: r
};

static Problem load(const string& dir) {
    Problem P;
    FILE* f = fopen((dir + "/meta.json").c_str(), "rb");
    if (!f) { fprintf(stderr, "no meta.json in %s\n", dir.c_str()); exit(1); }
    string js(8192, 0);
    js.resize(fread(&js[0], 1, 8192, f));
    fclose(f);
    P.n = meta_int(js, "n"); P.m = meta_int(js, "m");
    P.n_od = meta_int(js, "n_od"); P.nnz = meta_int(js, "nnz");
    P.r = meta_int(js, "r");
    P.n_major = meta_int(js, "n_major", false, 0);
    P.n_minor = meta_int(js, "n_minor", false, 0);
    P.Bp = load_bin<int64_t>(dir + "/B_indptr.i64");
    P.Bi = load_bin<int32_t>(dir + "/B_indices.i32");
    P.p2od = load_bin<int32_t>(dir + "/p2od.i32");
    P.d = load_bin<double>(dir + "/dvec.f64");   // 'd.f64' would collide with
                                                 // 'D.f64' on Windows (case-
                                                 // insensitive filesystem)
    P.x0 = load_bin<double>(dir + "/x0.f64");
    P.t0 = load_bin<double>(dir + "/bpr_t0.f64");
    P.alpha = load_bin<double>(dir + "/bpr_alpha.f64");
    P.beta = load_bin<double>(dir + "/bpr_beta.f64");
    P.cap = load_bin<double>(dir + "/bpr_cap.f64");
    // optional singleton background: v = v0 + B'x everywhere it matters. The compressed
    // blocks carry it inside v_base (the exporter bakes it in); the full model and every
    // from-scratch reconstruction must add it explicitly.
    if (FILE* f = fopen((dir + "/v0.f64").c_str(), "rb")) {
        fclose(f);
        P.v0 = load_bin<double>(dir + "/v0.f64");
    }
    bool has_compressed = false;
    if (FILE* f = fopen((dir + "/U.f64").c_str(), "rb")) {
        fclose(f);
        has_compressed = true;
        P.major = load_bin<uint8_t>(dir + "/major.u8");
        P.U = load_bin<double>(dir + "/U.f64");
        P.D = load_bin<double>(dir + "/D.f64");
        P.M = load_bin<double>(dir + "/M.f64");
        P.x0m = load_bin<double>(dir + "/x0m.f64");
        P.d_eff = load_bin<double>(dir + "/d_eff.f64");
        P.v_base = load_bin<double>(dir + "/v_base.f64");
    }
    // svd-newton factors (present only in svdn_problem_* dumps)
    if (FILE* f = fopen((dir + "/P.f64").c_str(), "rb")) {
        fclose(f);
        P.Pmat = load_bin<double>(dir + "/P.f64");
        P.Qmat = load_bin<double>(dir + "/Q.f64");
        P.sB = load_bin<double>(dir + "/sB.f64");
    } else if (has_compressed) {
        P.majid.reserve(P.n_major); P.minid.reserve(P.n_minor);
        for (long p = 0; p < P.n; ++p)
            (P.major[p] ? P.majid : P.minid).push_back((int32_t)p);
    }
    return P;
}

// dense symmetric-PD solve of (r x r) M k = b via Cholesky (r small)
static void chol_solve(vector<double> M, long r, const vector<double>& b,
                       vector<double>& k) {
    // in-place lower Cholesky
    for (long j = 0; j < r; ++j) {
        double d = M[j * r + j];
        for (long p = 0; p < j; ++p) d -= M[j * r + p] * M[j * r + p];
        d = std::sqrt(std::max(d, 1e-30));
        M[j * r + j] = d;
        for (long i = j + 1; i < r; ++i) {
            double s = M[i * r + j];
            for (long p = 0; p < j; ++p) s -= M[i * r + p] * M[j * r + p];
            M[i * r + j] = s / d;
        }
    }
    k = b;                                    // forward solve L y = b
    for (long i = 0; i < r; ++i) {
        double s = k[i];
        for (long p = 0; p < i; ++p) s -= M[i * r + p] * k[p];
        k[i] = s / M[i * r + i];
    }
    for (long i = r - 1; i >= 0; --i) {       // back solve L' k = y
        double s = k[i];
        for (long p = i + 1; p < r; ++p) s -= M[p * r + i] * k[p];
        k[i] = s / M[i * r + i];
    }
}

// inverse of an r x r SPD matrix via Cholesky (r small)
static void chol_inverse(const vector<double>& A, long r, vector<double>& Inv) {
    Inv.assign(r * r, 0.0);
    vector<double> col(r), sol(r);
    for (long j = 0; j < r; ++j) {
        std::fill(col.begin(), col.end(), 0.0);
        col[j] = 1.0;
        vector<double> M = A;                 // chol_solve overwrites M
        chol_solve(M, r, col, sol);
        for (long i = 0; i < r; ++i) Inv[i * r + j] = sol[i];
    }
}

// ---------------------------------------------------------------- kernels
struct Bpr {
    const Problem& P;
    explicit Bpr(const Problem& p) : P(p) {}
    void times(const vector<double>& v, vector<double>& t) const {
        #pragma omp parallel for schedule(static)
        for (long a = 0; a < P.m; ++a) {
            double vr = std::max(v[a], 0.0) / P.cap[a];
            t[a] = P.t0[a] * (1.0 + P.alpha[a] * std::pow(vr, P.beta[a]));
        }
    }
    void deriv(const vector<double>& v, vector<double>& W) const {
        #pragma omp parallel for schedule(static)
        for (long a = 0; a < P.m; ++a) {
            double vr = std::max(v[a], 1e-10) / P.cap[a];
            W[a] = P.t0[a] * P.alpha[a] * P.beta[a] / P.cap[a]
                   * std::pow(vr, P.beta[a] - 1.0);
        }
    }
    double beckmann(const vector<double>& v) const {
        // linear term on RAW v, max(v,0) on the power term only: keeps d/dv == times() for
        // ALL v, so objective and gradient stay consistent where the signed compressed basis
        // probes v<0 (with t0*max(v,0) the objective is flat there while times() returns t0,
        // which is the line-search inconsistency fixed on the Python side on 2026-07-15).
        // Feasible points (v>=0) are numerically unchanged.
        double s = 0;
        #pragma omp parallel for reduction(+:s) schedule(static)
        for (long a = 0; a < P.m; ++a) {
            double vp = std::max(v[a], 0.0), vr = vp / P.cap[a];
            s += P.t0[a] * v[a] + P.t0[a] * P.alpha[a] * P.cap[a] / (P.beta[a] + 1.0)
                 * std::pow(vr, P.beta[a] + 1.0);
        }
        return s;
    }
};

// v = B' x over a subset of paths (ids) with coefficient vector xsub (same order)
static void scatter_links(const Problem& P, const vector<int32_t>& ids,
                          const vector<double>& xsub, vector<double>& v) {
    std::fill(v.begin(), v.end(), 0.0);
    #pragma omp parallel
    {
        vector<double> loc(P.m, 0.0);
        #pragma omp for schedule(static)
        for (long k = 0; k < (long)ids.size(); ++k) {
            double xv = xsub[k];
            if (xv == 0.0) continue;
            long p = ids[k];
            for (int64_t q = P.Bp[p]; q < P.Bp[p + 1]; ++q) loc[P.Bi[q]] += xv;
        }
        #pragma omp critical
        for (long a = 0; a < P.m; ++a) v[a] += loc[a];
    }
}

// gather: out[k] = sum_{links of path ids[k]} t[link]
static void gather_paths(const Problem& P, const vector<int32_t>& ids,
                         const vector<double>& t, vector<double>& out) {
    #pragma omp parallel for schedule(static)
    for (long k = 0; k < (long)ids.size(); ++k) {
        long p = ids[k];
        double s = 0;
        for (int64_t q = P.Bp[p]; q < P.Bp[p + 1]; ++q) s += t[P.Bi[q]];
        out[k] = s;
    }
}

// od scatter: r[od] += coef * xsub over subset
static void od_accumulate(const Problem& P, const vector<int32_t>& ids,
                          const vector<double>& xsub, vector<double>& rod) {
    #pragma omp parallel
    {
        vector<double> loc(P.n_od, 0.0);
        #pragma omp for schedule(static)
        for (long k = 0; k < (long)ids.size(); ++k)
            loc[P.p2od[ids[k]]] += xsub[k];
        #pragma omp critical
        for (long w = 0; w < P.n_od; ++w) rod[w] += loc[w];
    }
}

static double nrm2(const vector<double>& v) {
    double s = 0;
    #pragma omp parallel for reduction(+:s) schedule(static)
    for (long i = 0; i < (long)v.size(); ++i) s += v[i] * v[i];
    return std::sqrt(s);
}

// dense GEMV helpers (row-major A: rows x r)
static void gemv(const vector<double>& A, long rows, long r,
                 const vector<double>& x, vector<double>& y, bool trans) {
    if (!trans) {                       // y(rows) = A x(r)
        #pragma omp parallel for schedule(static)
        for (long i = 0; i < rows; ++i) {
            const double* a = &A[i * r];
            double s = 0;
            for (long j = 0; j < r; ++j) s += a[j] * x[j];
            y[i] = s;
        }
    } else {                            // y(r) = A' x(rows)
        std::fill(y.begin(), y.end(), 0.0);
        #pragma omp parallel
        {
            vector<double> loc(r, 0.0);
            #pragma omp for schedule(static)
            for (long i = 0; i < rows; ++i) {
                const double* a = &A[i * r];
                double xv = x[i];
                if (xv == 0.0) continue;
                for (long j = 0; j < r; ++j) loc[j] += a[j] * xv;
            }
            #pragma omp critical
            for (long j = 0; j < r; ++j) y[j] += loc[j];
        }
    }
}

// ------------------------------------------------- projected L-BFGS core
// Two-loop recursion (memory 8) + projection-arc Armijo (Bertsekas 1982):
//   accept when F(P(w + a d)) <= F(w) + c1 * g'(P(w + a d) - w).
// Matches the inner-solver family used on the Python side (L-BFGS-B), which
// SPG could not (linear convergence stalled on the ill-conditioned ALM
// subproblems — see README TODO note).
struct SpgResult { long iters; double fval; };

template <typename FG, typename PJ, typename CB>
SpgResult plbfgs_cb(vector<double>& w, FG fg, PJ project, long max_iter,
                    double pg_tol, CB cb) {
    const int M = 8;
    const long nvar = (long)w.size();
    vector<vector<double>> S, Y;
    vector<double> g(nvar), w_new(nvar), g_new(nvar), dir(nvar), q(nvar);
    double f = fg(w, g);
    long it = 0;
    for (; it < max_iter; ++it) {
        // stationarity: ||w - P(w - g)||
        #pragma omp parallel for schedule(static)
        for (long i = 0; i < nvar; ++i) w_new[i] = w[i] - g[i];
        project(w_new);
        // ABSOLUTE projected-gradient max-norm, matching the Python reference's L-BFGS-B
        // gtol semantics. The former relative test sqrt(pg2)/(1+||g||) deflates as the ALM
        // penalty inflates ||g||, so inner solves quit while the outer residual stalls ---
        // the documented full-mode stall (Sioux gap stuck at 2.7e-2; grid floors missed).
        double pg_inf = 0;
        #pragma omp parallel for reduction(max:pg_inf) schedule(static)
        for (long i = 0; i < nvar; ++i) {
            double dd = std::fabs(w[i] - w_new[i]);
            if (dd > pg_inf) pg_inf = dd;
        }
        if (getenv("CS_DEBUG") && it < 3)
            fprintf(stderr, "[dbg] it=%ld pg_inf=%.3e tol=%.4e f=%.6e\n",
                    it, pg_inf, pg_tol, f);
        if (pg_inf < pg_tol) break;

        // two-loop recursion
        q = g;
        int k = (int)S.size();
        vector<double> alpha_c(k), rho_c(k);
        for (int j = k - 1; j >= 0; --j) {
            double sy = 0, sq = 0;
            #pragma omp parallel for reduction(+:sy, sq) schedule(static)
            for (long i = 0; i < nvar; ++i) {
                sy += S[j][i] * Y[j][i];
                sq += S[j][i] * q[i];
            }
            rho_c[j] = 1.0 / std::max(sy, 1e-16);
            alpha_c[j] = rho_c[j] * sq;
            #pragma omp parallel for schedule(static)
            for (long i = 0; i < nvar; ++i) q[i] -= alpha_c[j] * Y[j][i];
        }
        double gamma = 1.0;
        if (k > 0) {
            double sy = 0, yy = 0;
            #pragma omp parallel for reduction(+:sy, yy) schedule(static)
            for (long i = 0; i < nvar; ++i) {
                sy += S[k-1][i] * Y[k-1][i];
                yy += Y[k-1][i] * Y[k-1][i];
            }
            gamma = sy / std::max(yy, 1e-16);
        }
        #pragma omp parallel for schedule(static)
        for (long i = 0; i < nvar; ++i) q[i] *= gamma;
        for (int j = 0; j < k; ++j) {
            double yq = 0;
            #pragma omp parallel for reduction(+:yq) schedule(static)
            for (long i = 0; i < nvar; ++i) yq += Y[j][i] * q[i];
            double beta = rho_c[j] * yq;
            #pragma omp parallel for schedule(static)
            for (long i = 0; i < nvar; ++i) q[i] += (alpha_c[j] - beta) * S[j][i];
        }
        #pragma omp parallel for schedule(static)
        for (long i = 0; i < nvar; ++i) dir[i] = -q[i];

        // projection-arc Armijo
        double a = 1.0, f_try = f;
        bool ok = false;
        for (int ls = 0; ls < 40; ++ls) {
            #pragma omp parallel for schedule(static)
            for (long i = 0; i < nvar; ++i) w_new[i] = w[i] + a * dir[i];
            project(w_new);
            double arc = 0;
            #pragma omp parallel for reduction(+:arc) schedule(static)
            for (long i = 0; i < nvar; ++i) arc += g[i] * (w_new[i] - w[i]);
            f_try = fg(w_new, g_new);
            if ((f_try <= f + 1e-4 * arc && arc < 0) || a < 1e-14) { ok = arc < 0; break; }
            a *= 0.5;
        }
        if (getenv("CS_DEBUG") && it < 3)
            fprintf(stderr, "[dbg] it=%ld ls done a=%.3e ok=%d f_try=%.6e\n",
                    it, a, (int)ok, f_try);
        if (!ok) {                        // fall back to projected gradient step
            #pragma omp parallel for schedule(static)
            for (long i = 0; i < nvar; ++i) w_new[i] = w[i] - g[i];
            project(w_new);
            f_try = fg(w_new, g_new);
            if (getenv("CS_DEBUG"))
                fprintf(stderr, "[dbg] it=%ld fallback f=%.6e f_try=%.6e\n",
                        it, f, f_try);
            if (f_try > f) break;         // no descent available: stationary
        }
        // curvature update
        vector<double> s(nvar), y(nvar);
        double sy = 0, ss = 0;
        #pragma omp parallel for reduction(+:sy, ss) schedule(static)
        for (long i = 0; i < nvar; ++i) {
            s[i] = w_new[i] - w[i];
            y[i] = g_new[i] - g[i];
            sy += s[i] * y[i];
            ss += s[i] * s[i];
        }
        if (sy > 1e-10 * std::sqrt(ss) * nrm2(y)) {
            S.push_back(std::move(s));
            Y.push_back(std::move(y));
            if ((int)S.size() > M) { S.erase(S.begin()); Y.erase(Y.begin()); }
        }
        w.swap(w_new);
        g.swap(g_new);
        f = f_try;
        cb(it, w);
    }
    return {it, f};
}

template <typename FG, typename PJ>
SpgResult plbfgs(vector<double>& w, FG fg, PJ project, long max_iter, double pg_tol) {
    return plbfgs_cb(w, fg, project, max_iter, pg_tol,
                     [](long, const vector<double>&) {});
}

template <typename FG, typename PJ, typename CB>
SpgResult spg_cb(vector<double>& w, FG fg, PJ project, long max_iter,
                 double pg_tol, CB cb) {
    const int Mwin = 8;
    vector<double> g(w.size()), w_new(w.size()), g_new(w.size());
    double f = fg(w, g);
    vector<double> hist(Mwin, f);
    double step = 1.0;
    long it = 0;
    for (; it < max_iter; ++it) {
        // projected gradient stationarity
        double pgn = 0, gn = 0;
        #pragma omp parallel for reduction(+:pgn, gn) schedule(static)
        for (long i = 0; i < (long)w.size(); ++i) {
            double wi = w[i] - g[i];
            double pj = wi;             // projection applied component-wise below
            pgn += 0;                   // placeholder (computed after project)
            gn += g[i] * g[i];
        }
        // compute projected gradient properly
        w_new = w;
        for (long i = 0; i < (long)w.size(); ++i) w_new[i] = w[i] - g[i];
        project(w_new);
        double pg_inf = 0;
        #pragma omp parallel for reduction(max:pg_inf) schedule(static)
        for (long i = 0; i < (long)w.size(); ++i) {
            double dd = std::fabs(w[i] - w_new[i]);
            if (dd > pg_inf) pg_inf = dd;
        }
        if (pg_inf < pg_tol) break;   // absolute max-norm (see plbfgs_cb note)

        double fmax = *std::max_element(hist.begin(), hist.end());
        double alpha = step;
        double f_try = 0;
        for (int ls = 0; ls < 30; ++ls) {
            #pragma omp parallel for schedule(static)
            for (long i = 0; i < (long)w.size(); ++i)
                w_new[i] = w[i] - alpha * g[i];
            project(w_new);
            double dg = 0;
            #pragma omp parallel for reduction(+:dg) schedule(static)
            for (long i = 0; i < (long)w.size(); ++i)
                dg += g[i] * (w_new[i] - w[i]);
            f_try = fg(w_new, g_new);
            if (f_try <= fmax + 1e-4 * dg || alpha < 1e-14) break;
            alpha *= 0.5;
        }
        // BB step for next iteration
        double sty = 0, sts = 0;
        #pragma omp parallel for reduction(+:sty, sts) schedule(static)
        for (long i = 0; i < (long)w.size(); ++i) {
            double si = w_new[i] - w[i], yi = g_new[i] - g[i];
            sty += si * yi;
            sts += si * si;
        }
        step = (sty > 1e-16) ? std::min(std::max(sts / sty, 1e-6), 1e6) : 1.0;
        w.swap(w_new);
        g.swap(g_new);
        f = f_try;
        hist[it % Mwin] = f;
        cb(it, w);
    }
    return {it, f};
}

template <typename FG, typename PJ>
SpgResult spg(vector<double>& w, FG fg, PJ project, long max_iter, double pg_tol) {
    return spg_cb(w, fg, project, max_iter, pg_tol,
                  [](long, const vector<double>&) {});
}

// ---------------------------------------------------------------- main
// truncate the compressed blocks to the first r_use columns (singular-value
// order): repack row-major U/D/M from stride r to stride r_use
static void truncate_rank(Problem& P, long r_use) {
    if (r_use >= P.r) return;
    auto repack = [&](vector<double>& A, long rows) {
        vector<double> B((size_t)rows * r_use);
        #pragma omp parallel for schedule(static)
        for (long i = 0; i < rows; ++i)
            memcpy(&B[i * r_use], &A[i * P.r], r_use * sizeof(double));
        A.swap(B);
    };
    repack(P.U, P.n_minor);
    repack(P.D, P.m);
    repack(P.M, P.n_od);
    P.r = r_use;
}

// ---- support-locality restriction (validated: cpp_alm_latent_validation)
// Compression pays only when the solver also avoids full-network work:
// restrict every per-link array to the union of links touched by the pool
// (Bi) or carrying base/decoder mass. Links outside the union hold zero
// volume in every mode (v_base and D rows vanish there), so objectives
// and exported volumes are unchanged exactly. Disable: SUPPORT_LOCAL=0.
static long g_m_full = 0;
static vector<int32_t> g_act;            // active -> full link id

static void restrict_to_support(Problem& P) {
    const char* env = getenv("SUPPORT_LOCAL");
    if (env && env[0] == '0') { printf("support-local: disabled\n"); return; }
    vector<uint8_t> on(P.m, 0);
    for (int64_t q = 0; q < P.nnz; ++q) on[P.Bi[q]] = 1;
    for (long a = 0; a < P.m; ++a) {
        if (on[a]) continue;
        if (!P.v_base.empty() && P.v_base[a] != 0.0) { on[a] = 1; continue; }
        if (!P.v0.empty() && P.v0[a] != 0.0) { on[a] = 1; continue; }
        if (!P.D.empty()) {
            const double* da = &P.D[a * P.r];
            for (long j = 0; j < P.r; ++j) if (da[j] != 0.0) { on[a] = 1; break; }
        }
        if (!on[a] && !P.Qmat.empty()) {
            const double* qa = &P.Qmat[a * P.r];
            for (long j = 0; j < P.r; ++j) if (qa[j] != 0.0) { on[a] = 1; break; }
        }
    }
    long mA = 0;
    vector<int32_t> to_act(P.m, -1);
    for (long a = 0; a < P.m; ++a) if (on[a]) to_act[a] = (int32_t)mA++;
    if (mA == P.m) { printf("support-local: all %ld links active\n", P.m); return; }
    g_m_full = P.m;
    g_act.resize(mA);
    for (long a = 0; a < P.m; ++a) if (on[a]) g_act[to_act[a]] = (int32_t)a;
    for (int64_t q = 0; q < P.nnz; ++q) P.Bi[q] = to_act[P.Bi[q]];
    auto pick = [&](vector<double>& A, long stride) {
        if (A.empty()) return;
        vector<double> B((size_t)mA * stride);
        for (long i = 0; i < mA; ++i)
            std::copy(&A[(size_t)g_act[i] * stride],
                      &A[(size_t)g_act[i] * stride] + stride,
                      &B[(size_t)i * stride]);
        A.swap(B);
    };
    pick(P.t0, 1); pick(P.alpha, 1); pick(P.beta, 1); pick(P.cap, 1);
    pick(P.v_base, 1); pick(P.v0, 1); pick(P.D, P.r); pick(P.Qmat, P.r);
    printf("support-local: m %ld -> %ld active (%.1f%%)\n",
           P.m, mA, 100.0 * mA / P.m);
    P.m = mA;
}

// add the singleton background into a freshly scattered link-flow vector
static void add_bg(const Problem& P, vector<double>& v) {
    if (P.v0.empty()) return;
    #pragma omp parallel for schedule(static)
    for (long a = 0; a < P.m; ++a) v[a] += P.v0[a];
}

// in-pool AON relative gap of a (feasible) path-flow vector — the common certificate the
// Python harness reports (pool_relgap), computed on total volumes v0 + B'x.
static double pool_gap(const Problem& P, const Bpr& bpr,
                       const vector<int32_t>& allids, const vector<double>& x) {
    vector<double> vh(P.m), th(P.m), ch(P.n);
    scatter_links(P, allids, x, vh);
    add_bg(P, vh);
    bpr.times(vh, th);
    gather_paths(P, allids, th, ch);
    vector<double> best(P.n_od, 1e300);
    double cx = 0, cy = 0;
    for (long p = 0; p < P.n; ++p) {
        cx += ch[p] * x[p];
        if (ch[p] < best[P.p2od[p]]) best[P.p2od[p]] = ch[p];
    }
    for (long q = 0; q < P.n_od; ++q) cy += P.d[q] * best[q];
    return (cx - cy) / std::max(cy, 1e-12);
}

// export restricted v scattered back to the full link index space
static void dump_v(const string& path, const vector<double>& v) {
    FILE* fo = fopen(path.c_str(), "wb");
    if (!fo) return;
    if (g_m_full) {
        vector<double> vf(g_m_full, 0.0);
        for (long i = 0; i < (long)v.size(); ++i) vf[g_act[i]] = v[i];
        fwrite(vf.data(), sizeof(double), vf.size(), fo);
    } else {
        fwrite(v.data(), sizeof(double), v.size(), fo);
    }
    fclose(fo);
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc < 3) { fprintf(stderr, "usage: %s <dir> full|soft|screen|hard [max_outer] [rank_r] [vdump] [max_inner]\n", argv[0]); return 1; }
    string dir = argv[1], mode = argv[2];
    long max_outer = argc > 3 ? atol(argv[3]) : 40;
    long r_use = argc > 4 ? atol(argv[4]) : 0;
    string vdump = argc > 5 ? argv[5] : "";
    long max_inner = argc > 6 ? atol(argv[6]) : 300;
    auto t_load0 = clk::now();
    Problem P = load(dir);
    if (mode == "svdnewton" && !P.v0.empty()) {
        fprintf(stderr, "svdnewton does not carry the singleton background v0; "
                        "use full|soft|screen|hard on this instance\n");
        return 1;
    }
    if (r_use > 0) truncate_rank(P, r_use);
    if (mode != "kernels") restrict_to_support(P);   // kernels: full link space for parity
    Bpr bpr(P);
    auto t_load1 = clk::now();
    printf("loaded n=%ld m=%ld n_od=%ld nnz=%ld r=%ld majors=%ld minors=%ld (%.1fs)\n",
           P.n, P.m, P.n_od, P.nnz, P.r, P.n_major, P.n_minor, secs(t_load0, t_load1));
#ifdef _OPENMP
    printf("openmp threads: %d\n", omp_get_max_threads());
#endif

    vector<int32_t> allids(P.n);
    for (long p = 0; p < P.n; ++p) allids[p] = (int32_t)p;

    // ---------------- kernel-parity mode (protocol gate C1) --------------------------------
    // usage: compressed_solver <dir> kernels 0 0 <x_test.f64>
    // Evaluates the mathematical primitives at the frozen test vector x — v = v0 + B'x,
    // BPR travel times t(v), Beckmann objective, path-cost gradient B t(v), and the in-pool
    // relative gap — and writes <x>.v.f64 / .t.f64 / .g.f64 plus a KERNELS line, so the
    // Python reference implementation can compare every component to protocol tolerances
    // BEFORE any optimizer output is trusted. No optimization happens in this mode.
    if (mode == "kernels") {
        if (vdump.empty()) { fprintf(stderr, "kernels mode needs <x_test.f64>\n"); return 1; }
        vector<double> x = load_bin<double>(vdump);
        if ((long)x.size() != P.n) {
            fprintf(stderr, "x size %zu != n %ld\n", x.size(), P.n); return 1;
        }
        vector<double> v(P.m), tl(P.m), g(P.n);
        scatter_links(P, allids, x, v);
        add_bg(P, v);
        bpr.times(v, tl);
        gather_paths(P, allids, tl, g);
        auto wf = [&](const string& suf, const vector<double>& a) {
            FILE* fo = fopen((vdump + suf).c_str(), "wb");
            fwrite(a.data(), sizeof(double), a.size(), fo); fclose(fo);
        };
        wf(".v.f64", v); wf(".t.f64", tl); wf(".g.f64", g);
        printf("KERNELS obj=%.17e gap=%.17e\n", bpr.beckmann(v),
               pool_gap(P, bpr, allids, x));
        return 0;
    }

    const double tol_cons = 1e-4;
    double rho = 100.0, prev_cons = 1e300;
    vector<double> lam(P.n_od, 0.0);
    auto t0c = clk::now();

    // ---------------- R0: exact anchor elimination + capped-simplex SPG (gate C2) ----------
    // usage: compressed_solver <dir> r0 [max_iter] [-] [vdump] [-]
    // Anchor a(w) = per-OD argmax of the nominal flow (matches the Python reference). The OD
    // equalities are eliminated exactly (x_a = d_w - sum of the free flows), and the free
    // variables live in the capped simplex {u >= 0, 1'u <= d_w}, enforced by the EXACT
    // OD-wise projection at every iterate (sorting argument; O(K log K) only when the cap
    // binds). No multipliers, no penalties: demand conservation and anchor nonnegativity
    // hold exactly throughout, which is the manuscript's stated mechanism. Inner engine is
    // the nonmonotone spectral projected gradient (BB step, Grippo window) above.
    if (mode == "r0") {
        long max_iter = max_outer > 100 ? max_outer : 20000;
        vector<int32_t> anchor(P.n_od, -1);
        for (long p = 0; p < P.n; ++p) {
            long w = P.p2od[p];
            if (anchor[w] < 0 || P.x0[p] > P.x0[anchor[w]]) anchor[w] = (int32_t)p;
        }
        vector<uint8_t> is_anc(P.n, 0);
        for (long w = 0; w < P.n_od; ++w) is_anc[anchor[w]] = 1;
        // free ids grouped by OD: contiguous segments so the projection is segment-local
        vector<int64_t> seg(P.n_od + 1, 0);
        for (long p = 0; p < P.n; ++p) if (!is_anc[p]) seg[P.p2od[p] + 1]++;
        for (long w = 0; w < P.n_od; ++w) seg[w + 1] += seg[w];
        long nf = P.n - P.n_od;
        vector<int32_t> fids(nf);
        {
            vector<int64_t> pos(seg.begin(), seg.end() - 1);
            for (long p = 0; p < P.n; ++p)
                if (!is_anc[p]) fids[pos[P.p2od[p]]++] = (int32_t)p;
        }
        auto project = [&](vector<double>& u) {
            #pragma omp parallel for schedule(dynamic, 64)
            for (long w = 0; w < P.n_od; ++w) {
                long a = seg[w], b = seg[w + 1];
                if (a == b) continue;
                double dcap = P.d[w], s = 0;
                for (long k = a; k < b; ++k) { u[k] = std::max(u[k], 0.0); s += u[k]; }
                if (s <= dcap) continue;
                vector<double> t(u.begin() + a, u.begin() + b);
                std::sort(t.begin(), t.end(), std::greater<double>());
                double cs = 0, theta = 0;
                for (long k = 0; k < (long)t.size(); ++k) {
                    cs += t[k];
                    double th = (cs - dcap) / (k + 1);
                    if (t[k] - th > 0) theta = th; else break;
                }
                for (long k = a; k < b; ++k) u[k] = std::max(u[k] - theta, 0.0);
            }
        };
        vector<double> y(nf);
        for (long k = 0; k < nf; ++k) y[k] = std::max(P.x0[fids[k]], 1e-3);
        project(y);                                   // feasible from the first iterate

        vector<int32_t> aid(anchor.begin(), anchor.end());
        vector<double> x(P.n), v(P.m), tl(P.m), cfree(nf), canc(P.n_od);
        auto assemble = [&](const vector<double>& u) {
            for (long k = 0; k < nf; ++k) x[fids[k]] = u[k];
            for (long w = 0; w < P.n_od; ++w) {
                double s = 0;
                for (long k = seg[w]; k < seg[w + 1]; ++k) s += u[k];
                x[anchor[w]] = P.d[w] - s;
            }
        };
        auto fg = [&](const vector<double>& u, vector<double>& g) {
            assemble(u);
            scatter_links(P, allids, x, v);
            add_bg(P, v);
            bpr.times(v, tl);
            gather_paths(P, fids, tl, cfree);
            gather_paths(P, aid, tl, canc);
            #pragma omp parallel for schedule(static)
            for (long k = 0; k < nf; ++k)
                g[k] = cfree[k] - canc[P.p2od[fids[k]]];   // reduced gradient
            return bpr.beckmann(v);
        };
        // optional argv[4] overrides the stationarity tolerance (absolute pg max-norm) so
        // timed runs can stop in the same accuracy class as r1/hard instead of polishing to
        // machine precision; default keeps the certified 1e-9 behavior.
        double pg_tol = (argc > 4 && argv[4][0] && atof(argv[4]) > 0) ? atof(argv[4]) : 1e-9;
        // matched-accuracy trajectory: HIST_CSV=<path> logs (elapsed_s, pool_gap) every 20
        // SPG iterations, so T_M(eps) can be read off for any eps both models reach.
        FILE* hf = nullptr;
        if (const char* hcsv = getenv("HIST_CSV")) {
            hf = fopen(hcsv, "w");
            if (hf) fprintf(hf, "elapsed_s,gap\n");
        }
        auto cb = [&](long it, const vector<double>& u) {
            if (!hf || it % 20 != 0) return;
            assemble(u);
            fprintf(hf, "%.6f,%.10e\n", secs(t0c, clk::now()),
                    pool_gap(P, bpr, allids, x));
            fflush(hf);
        };
        SpgResult res = spg_cb(y, fg, project, max_iter, pg_tol, cb);
        if (hf) fclose(hf);
        assemble(y);
        scatter_links(P, allids, x, v);
        add_bg(P, v);
        vector<double> rr(P.n_od, 0.0);
        od_accumulate(P, allids, x, rr);
        double cons = 0, min_anchor = 1e300;
        long at_bound = 0;
        for (long w = 0; w < P.n_od; ++w) {
            double e = rr[w] - P.d[w];
            cons += e * e;
            double xa = x[anchor[w]];
            min_anchor = std::min(min_anchor, xa);
            if (xa < 1e-9) ++at_bound;
        }
        printf("RESULT mode=r0 r_used=0 time_s=%.2f inner_iters=%ld obj_feasible=%.4f "
               "cons=%.3e pool_gap=%.4e min_anchor=%.3e anchors_at_bound=%ld\n",
               secs(t0c, clk::now()), res.iters, bpr.beckmann(v), std::sqrt(cons),
               pool_gap(P, bpr, allids, x), min_anchor, at_bound);
        if (!vdump.empty()) dump_v(vdump, v);
        return 0;
    }

    // ---------------- R1: anchor elimination + rank-r minor decoder (gate C3) --------------
    // usage: compressed_solver <dir> r1 [max_outer] [rank_r] [vdump] [max_inner]
    // Variables w = (y, z): y = free MAJOR path flows (majors minus anchors, box y >= 0),
    // z = latent minor coordinates, minors reconstructed as x0m + U z. Anchor flows are
    // eliminated exactly: x_a = d_w - sum(free majors of w) - sum(minor recon of w), so OD
    // conservation holds identically at every iterate. Minor nonnegativity (mu) and the
    // anchor bound (nu) are handled by multiplier-ALM, matching the Python reference
    // solve_R1 for parity; the inner engine is the projected L-BFGS above.
    if (mode == "r1") {
        if (P.U.empty()) { fprintf(stderr, "r1 needs the compressed blocks\n"); return 1; }
        long r = P.r, nm = P.n_minor;
        vector<int32_t> anchor(P.n_od, -1);
        for (long p = 0; p < P.n; ++p) {
            long w = P.p2od[p];
            if (anchor[w] < 0 || P.x0[p] > P.x0[anchor[w]]) anchor[w] = (int32_t)p;
        }
        vector<uint8_t> is_anc(P.n, 0);
        for (long w = 0; w < P.n_od; ++w) is_anc[anchor[w]] = 1;
        vector<int32_t> fids;                       // free majors (majors minus anchors)
        for (long k = 0; k < (long)P.majid.size(); ++k)
            if (!is_anc[P.majid[k]]) fids.push_back(P.majid[k]);
        long ny = fids.size();
        vector<int32_t> aid(anchor.begin(), anchor.end());

        vector<double> w_(ny + r, 0.0);
        for (long k = 0; k < ny; ++k) w_[k] = std::max(P.x0[fids[k]], 1e-3);
        vector<double> mu(nm, 0.0), nu(P.n_od, 0.0);
        const double c2 = 1e3;
        vector<double> x(P.n), v(P.m), tl(P.m), um(nm), cf(ny), cm(nm), canc(P.n_od),
                       xa(P.n_od), phi(nm), psi(P.n_od), gz(std::max(r, 1L));
        auto assemble = [&](const vector<double>& wv) {
            const double* y = wv.data();
            gemv(P.U, nm, r, vector<double>(wv.begin() + ny, wv.end()), um, false);
            for (long i = 0; i < nm; ++i) um[i] += P.x0m[i];
            std::fill(x.begin(), x.end(), 0.0);
            for (long k = 0; k < ny; ++k) x[fids[k]] = y[k];
            for (long i = 0; i < nm; ++i) x[P.minid[i]] = um[i];
            std::fill(xa.begin(), xa.end(), 0.0);
            for (long k = 0; k < ny; ++k) xa[P.p2od[fids[k]]] += y[k];
            for (long i = 0; i < nm; ++i) xa[P.p2od[P.minid[i]]] += um[i];
            for (long q = 0; q < P.n_od; ++q) xa[q] = P.d[q] - xa[q];
            for (long q = 0; q < P.n_od; ++q) x[anchor[q]] = xa[q];
        };
        auto fg = [&](const vector<double>& wv, vector<double>& g) {
            assemble(wv);
            scatter_links(P, allids, x, v);
            add_bg(P, v);
            double f = bpr.beckmann(v);
            bpr.times(v, tl);
            gather_paths(P, fids, tl, cf);
            gather_paths(P, P.minid, tl, cm);
            gather_paths(P, aid, tl, canc);
            double fpen = 0;
            for (long i = 0; i < nm; ++i) {
                phi[i] = std::max(0.0, mu[i] - c2 * um[i]);
                fpen += phi[i] * phi[i] - mu[i] * mu[i];
            }
            for (long q = 0; q < P.n_od; ++q) {
                psi[q] = std::max(0.0, nu[q] - c2 * xa[q]);
                fpen += psi[q] * psi[q] - nu[q] * nu[q];
            }
            f += fpen / (2.0 * c2);
            // reduced gradient wrt y: (c_p - c_a(w)) + psi_w  (anchor bound: dxa/dy = -1)
            #pragma omp parallel for schedule(static)
            for (long k = 0; k < ny; ++k) {
                long q = P.p2od[fids[k]];
                g[k] = cf[k] - canc[q] + psi[q];
            }
            // wrt z: U^T [ (c_minor - c_a(w)) - phi + psi_w ]
            for (long i = 0; i < nm; ++i) {
                long q = P.p2od[P.minid[i]];
                cm[i] = cm[i] - canc[q] - phi[i] + psi[q];
            }
            gemv(P.U, nm, r, cm, gz, true);
            for (long j = 0; j < r; ++j) g[ny + j] = gz[j];
            return f;
        };
        auto proj = [&](vector<double>& wv) {
            #pragma omp parallel for schedule(static)
            for (long k = 0; k < ny; ++k) wv[k] = std::max(wv[k], 0.0);
        };
        double eta = 0.05;
        long tot_it = 0;
        double viol = 1e300;
        FILE* hf = nullptr;
        if (const char* hcsv = getenv("HIST_CSV")) {
            hf = fopen(hcsv, "w");
            if (hf) fprintf(hf, "elapsed_s,gap\n");
        }
        auto checkpoint = [&]() {
            if (!hf) return;
            // gap of the feasibility-projected current point (clip + per-OD rescale)
            vector<double> xh(x), odsum(P.n_od, 0.0);
            for (long p = 0; p < P.n; ++p) xh[p] = std::max(xh[p], 0.0);
            od_accumulate(P, allids, xh, odsum);
            for (long p = 0; p < P.n; ++p) {
                double s = odsum[P.p2od[p]];
                xh[p] *= (s > 1e-12) ? P.d[P.p2od[p]] / s : 1.0;
            }
            fprintf(hf, "%.6f,%.10e\n", secs(t0c, clk::now()),
                    pool_gap(P, bpr, allids, xh));
            fflush(hf);
        };
        for (long outer = 0; outer < max_outer; ++outer) {
            SpgResult res = plbfgs(w_, fg, proj, max_inner, eta);
            tot_it += res.iters;
            assemble(w_);
            checkpoint();
            double vm = 0, va = 0;
            for (long i = 0; i < nm; ++i) vm += std::min(um[i], 0.0) * std::min(um[i], 0.0);
            for (long q = 0; q < P.n_od; ++q) va += std::min(xa[q], 0.0) * std::min(xa[q], 0.0);
            viol = std::sqrt(vm) + std::sqrt(va);
            printf("  outer %ld iters %ld viol %.2e t %.1fs\n", outer + 1, res.iters, viol,
                   secs(t0c, clk::now()));
            fflush(stdout);
            for (long i = 0; i < nm; ++i) mu[i] = std::max(0.0, mu[i] - c2 * um[i]);
            for (long q = 0; q < P.n_od; ++q) nu[q] = std::max(0.0, nu[q] - c2 * xa[q]);
            if (viol < 1e-3 && eta <= 1e-5) break;
            eta = std::max(eta * 0.5, 1e-6);
        }
        if (hf) fclose(hf);
        // feasible projection (clip + per-OD rescale, minors leave the subspace) + report
        assemble(w_);
        {
            vector<double> odsum(P.n_od, 0.0);
            #pragma omp parallel for schedule(static)
            for (long p = 0; p < P.n; ++p) x[p] = std::max(x[p], 0.0);
            od_accumulate(P, allids, x, odsum);
            #pragma omp parallel for schedule(static)
            for (long p = 0; p < P.n; ++p) {
                double s = odsum[P.p2od[p]];
                x[p] *= (s > 1e-12) ? P.d[P.p2od[p]] / s : 1.0;
            }
        }
        scatter_links(P, allids, x, v);
        add_bg(P, v);
        printf("RESULT mode=r1 r_used=%ld time_s=%.2f inner_iters=%ld obj_feasible=%.4f "
               "cons=%.3e pool_gap=%.4e viol=%.3e n_var=%ld\n",
               r, secs(t0c, clk::now()), tot_it, bpr.beckmann(v), 0.0,
               pool_gap(P, bpr, allids, x), viol, ny + r);
        if (!vdump.empty()) dump_v(vdump, v);
        return 0;
    }

    if (mode == "full") {
        // x >= 0 over all paths; start: per-OD rescaled max(x0,1e-3)
        vector<double> x(P.n);
        {
            vector<double> odsum(P.n_od, 0.0);
            for (long p = 0; p < P.n; ++p) x[p] = std::max(P.x0[p], 1e-3);
            od_accumulate(P, allids, x, odsum);
            for (long p = 0; p < P.n; ++p) {
                double s = odsum[P.p2od[p]];
                x[p] *= (s > 1e-12) ? P.d[P.p2od[p]] / s : 1.0;
            }
        }
        vector<double> v(P.m), tl(P.m), gpath(P.n), rr(P.n_od);
        long tot_it = 0;
        double eta = 0.05, cons = 0;
        FILE* hf = nullptr;
        if (const char* hcsv = getenv("HIST_CSV")) {
            hf = fopen(hcsv, "w");
            if (hf) fprintf(hf, "elapsed_s,gap\n");
        }
        auto hist_cb = [&](long it, const vector<double>& wv) {
            if (!hf || it % 20 != 0) return;
            vector<double> xh(wv), odsum(P.n_od, 0.0);
            for (long p = 0; p < P.n; ++p) xh[p] = std::max(xh[p], 0.0);
            od_accumulate(P, allids, xh, odsum);
            for (long p = 0; p < P.n; ++p) {
                double s = odsum[P.p2od[p]];
                xh[p] *= (s > 1e-12) ? P.d[P.p2od[p]] / s : 1.0;
            }
            fprintf(hf, "%.6f,%.10e\n", secs(t0c, clk::now()),
                    pool_gap(P, bpr, allids, xh));
            fflush(hf);
        };
        for (long outer = 0; outer < max_outer; ++outer) {
            auto fg = [&](const vector<double>& w, vector<double>& g) {
                scatter_links(P, allids, w, v);
                add_bg(P, v);
                std::fill(rr.begin(), rr.end(), 0.0);
                od_accumulate(P, allids, w, rr);
                #pragma omp parallel for schedule(static)
                for (long q = 0; q < P.n_od; ++q) rr[q] -= P.d[q];
                bpr.times(v, tl);
                gather_paths(P, allids, tl, g);
                #pragma omp parallel for schedule(static)
                for (long p = 0; p < P.n; ++p)
                    g[p] += lam[P.p2od[p]] + rho * rr[P.p2od[p]];
                double pen = 0, lr = 0;
                #pragma omp parallel for reduction(+:pen, lr) schedule(static)
                for (long q = 0; q < P.n_od; ++q) {
                    pen += rr[q] * rr[q];
                    lr += lam[q] * rr[q];
                }
                return bpr.beckmann(v) + lr + 0.5 * rho * pen;
            };
            auto proj = [&](vector<double>& w) {
                #pragma omp parallel for schedule(static)
                for (long i = 0; i < (long)w.size(); ++i) w[i] = std::max(w[i], 0.0);
            };
            SpgResult rres = plbfgs_cb(x, fg, proj, max_inner, eta, hist_cb);
            tot_it += rres.iters;
            std::fill(rr.begin(), rr.end(), 0.0);
            od_accumulate(P, allids, x, rr);
            for (long q = 0; q < P.n_od; ++q) rr[q] -= P.d[q];
            cons = nrm2(rr);
            printf("  outer %ld iters %ld cons %.5f t %.1fs\n",
                   outer + 1, rres.iters, cons, secs(t0c, clk::now()));
            fflush(stdout);
            // require the inexact-ALM forcing tolerance to have tightened —
            // a feasible warm start can satisfy cons at outer 1 while the
            // subproblem is solved only to the loose initial eta
            if (cons < tol_cons && eta <= 1e-5) break;
            for (long q = 0; q < P.n_od; ++q) lam[q] += rho * rr[q];
            if (cons > 0.25 * prev_cons) rho = std::min(rho * 2.0, 1e6);
            prev_cons = cons;
            eta = std::max(eta * 0.5, 1e-6);
        }
        if (hf) fclose(hf);
        // feasible projection + report
        vector<double> odsum(P.n_od, 0.0);
        od_accumulate(P, allids, x, odsum);
        // raw-x dump (added 2026-07-24, purely additive): UNPROJECTED path flows for the
        // v3 delta_F/Gap_F/R2 metrics. Full mode needs it too (the tau=0 reference).
        if (const char* xr = getenv("VDUMP_XRAW")) {
            FILE* fx = fopen(xr, "wb");
            if (fx) { fwrite(x.data(), sizeof(double), x.size(), fx); fclose(fx); }
        }
        #pragma omp parallel for schedule(static)
        for (long p = 0; p < P.n; ++p) {
            double s = odsum[P.p2od[p]];
            x[p] *= (s > 1e-12) ? P.d[P.p2od[p]] / s : 1.0;
        }
        scatter_links(P, allids, x, v);
        add_bg(P, v);
        printf("RESULT mode=full r_used=0 time_s=%.1f inner_iters=%ld obj_feasible=%.4f "
               "cons=%.6f pool_gap=%.4e\n",
               secs(t0c, clk::now()), tot_it, bpr.beckmann(v), cons,
               pool_gap(P, bpr, allids, x));
        if (!vdump.empty()) dump_v(vdump, v);
        return 0;
    }

    // ---------------- SVD-Newton: full x, Woodbury Hessian inverse ----------
    if (mode == "svdnewton") {
        if (P.Pmat.empty()) { fprintf(stderr, "no SVD factors in %s (use svdn_problem_*)\n", dir.c_str()); return 1; }
        long r = P.r;
        vector<double> x(P.n);
        {   // feasible start: per-OD rescaled max(x0,1e-3)
            vector<double> odsum(P.n_od, 0.0);
            for (long p = 0; p < P.n; ++p) x[p] = std::max(P.x0[p], 1e-3);
            od_accumulate(P, allids, x, odsum);
            for (long p = 0; p < P.n; ++p) {
                double s = odsum[P.p2od[p]];
                x[p] *= (s > 1e-12) ? P.d[P.p2od[p]] / s : 1.0;
            }
        }
        vector<double> v(P.m), tl(P.m), W(P.m), g(P.n), rr(P.n_od);
        vector<double> Ug(r), kk(r), PUg(P.n), Pk(P.n), Qw(r * r), C(r * r);
        double rho = 100.0, prev = 1e300, eta = 0.05, cons = 0;
        long tot_it = 0;
        for (long outer = 0; outer < max_outer; ++outer) {
            for (long inner = 0; inner < 30; ++inner) {
                scatter_links(P, allids, x, v);
                bpr.times(v, tl);
                bpr.deriv(v, W);
                std::fill(rr.begin(), rr.end(), 0.0);
                od_accumulate(P, allids, x, rr);
                #pragma omp parallel for schedule(static)
                for (long q = 0; q < P.n_od; ++q) rr[q] -= P.d[q];
                gather_paths(P, allids, tl, g);
                #pragma omp parallel for schedule(static)
                for (long p = 0; p < P.n; ++p)
                    g[p] += lam[P.p2od[p]] + rho * rr[P.p2od[p]];
                // Qw = Q' diag(W) Q  (r x r)
                std::fill(Qw.begin(), Qw.end(), 0.0);
                #pragma omp parallel
                {
                    vector<double> loc(r * r, 0.0);
                    #pragma omp for schedule(static)
                    for (long a = 0; a < P.m; ++a) {
                        const double* qa = &P.Qmat[a * r];
                        double wa = W[a];
                        for (long j = 0; j < r; ++j) {
                            double t = wa * qa[j];
                            for (long l = 0; l < r; ++l) loc[j * r + l] += t * qa[l];
                        }
                    }
                    #pragma omp critical
                    for (long i = 0; i < r * r; ++i) Qw[i] += loc[i];
                }
                // C = diag(s) Qw diag(s) + rho I
                for (long j = 0; j < r; ++j)
                    for (long l = 0; l < r; ++l)
                        C[j * r + l] = P.sB[j] * Qw[j * r + l] * P.sB[l]
                                       + (j == l ? rho : 0.0);
                // Ug = P' g
                std::fill(Ug.begin(), Ug.end(), 0.0);
                #pragma omp parallel
                {
                    vector<double> loc(r, 0.0);
                    #pragma omp for schedule(static)
                    for (long p = 0; p < P.n; ++p) {
                        const double* pp = &P.Pmat[p * r];
                        double gp = g[p];
                        for (long j = 0; j < r; ++j) loc[j] += pp[j] * gp;
                    }
                    #pragma omp critical
                    for (long j = 0; j < r; ++j) Ug[j] += loc[j];
                }
                chol_solve(C, r, Ug, kk);
                // PUg = P (P'g),  Pk = P kk
                #pragma omp parallel for schedule(static)
                for (long p = 0; p < P.n; ++p) {
                    const double* pp = &P.Pmat[p * r];
                    double a = 0, b = 0;
                    for (long j = 0; j < r; ++j) { a += pp[j] * Ug[j]; b += pp[j] * kk[j]; }
                    PUg[p] = a; Pk[p] = b;
                }
                // dx = -H^{-1} g = -[(g - PUg)/rho + Pk]
                // arc-Armijo projection line search
                double L_curr = bpr.beckmann(v);
                { double lr = 0, pq = 0;
                  for (long q = 0; q < P.n_od; ++q) { lr += lam[q]*rr[q]; pq += rr[q]*rr[q]; }
                  L_curr += lr + 0.5 * rho * pq; }
                double alpha = 1.0; bool ok = false; double pg_meas = 0;
                vector<double> x_new(P.n), v_new(P.m), rr_new(P.n_od);
                for (int ls = 0; ls < 30; ++ls) {
                    #pragma omp parallel for schedule(static)
                    for (long p = 0; p < P.n; ++p) {
                        double dx = -((g[p] - PUg[p]) / rho + Pk[p]);
                        x_new[p] = std::max(x[p] + alpha * dx, 0.0);
                    }
                    scatter_links(P, allids, x_new, v_new);
                    std::fill(rr_new.begin(), rr_new.end(), 0.0);
                    od_accumulate(P, allids, x_new, rr_new);
                    for (long q = 0; q < P.n_od; ++q) rr_new[q] -= P.d[q];
                    double L_new = bpr.beckmann(v_new);
                    double lr = 0, pq = 0, arc = 0;
                    for (long q = 0; q < P.n_od; ++q) { lr += lam[q]*rr_new[q]; pq += rr_new[q]*rr_new[q]; }
                    L_new += lr + 0.5 * rho * pq;
                    #pragma omp parallel for reduction(+:arc) schedule(static)
                    for (long p = 0; p < P.n; ++p) arc += g[p] * (x_new[p] - x[p]);
                    if ((L_new <= L_curr + 1e-4 * arc && arc < 0) || alpha < 1e-14) { ok = arc < 0; pg_meas = 0; break; }
                    alpha *= 0.5;
                }
                double xch = 0, xn = 0;
                #pragma omp parallel for reduction(+:xch, xn) schedule(static)
                for (long p = 0; p < P.n; ++p) { double dd = x_new[p]-x[p]; xch += dd*dd; xn += x[p]*x[p]; }
                x.swap(x_new);
                tot_it++;
                if (!ok || std::sqrt(xch)/(1.0+std::sqrt(xn)) < eta) break;
            }
            std::fill(rr.begin(), rr.end(), 0.0);
            od_accumulate(P, allids, x, rr);
            for (long q = 0; q < P.n_od; ++q) rr[q] -= P.d[q];
            cons = nrm2(rr);
            printf("  outer %ld tot %ld cons %.5f rho %.1e t %.1fs\n",
                   outer + 1, tot_it, cons, rho, secs(t0c, clk::now()));
            fflush(stdout);
            if (cons < tol_cons && eta <= 1e-5) break;
            for (long q = 0; q < P.n_od; ++q) lam[q] += rho * rr[q];
            if (cons > 0.25 * prev) rho = std::min(rho * 2.0, 1e6);
            prev = cons;
            eta = std::max(eta * 0.5, 1e-6);
        }
        vector<double> odsum(P.n_od, 0.0);
        od_accumulate(P, allids, x, odsum);
        #pragma omp parallel for schedule(static)
        for (long p = 0; p < P.n; ++p) {
            double s = odsum[P.p2od[p]];
            x[p] *= (s > 1e-12) ? P.d[P.p2od[p]] / s : 1.0;
        }
        scatter_links(P, allids, x, v);
        printf("RESULT mode=svdnewton r_used=%ld time_s=%.1f inner_iters=%ld obj_feasible=%.4f cons=%.6f\n",
               r, secs(t0c, clk::now()), tot_it, bpr.beckmann(v), cons);
        if (!vdump.empty()) dump_v(vdump, v);
        return 0;
    }

    // ---------- ALM-Newton / ALM-SVD-Newton: full x, ALM for BOTH constraints
    // OD via lambda/rho(=c1, A'A=I); nonnegativity x>=0 via mu/c2 penalty
    // (Eq.3), NOT projection. H = B W B' + rho I + c2 diag(active).
    if (mode == "almnewton" || mode == "almsvdn") {
        bool use_svd = (mode == "almsvdn");
        if (use_svd && P.Pmat.empty()) { fprintf(stderr, "no SVD factors\n"); return 1; }
        long r = P.r;
        const double c2 = 1e3;
        vector<double> x(P.n), mu(P.n, 0.0);
        {   // feasible start
            vector<double> odsum(P.n_od, 0.0);
            for (long p = 0; p < P.n; ++p) x[p] = std::max(P.x0[p], 1e-3);
            od_accumulate(P, allids, x, odsum);
            for (long p = 0; p < P.n; ++p) {
                double s = odsum[P.p2od[p]];
                x[p] *= (s > 1e-12) ? P.d[P.p2od[p]] / s : 1.0;
            }
        }
        vector<double> v(P.m), tl(P.m), W(P.m), g(P.n), rr(P.n_od), Dg(P.n);
        vector<double> C(use_svd ? r * r : 0), Cinv(use_svd ? r * r : 0),
            Bmat(use_svd ? r * r : 0), K(use_svd ? r * r : 0),
            Ug(use_svd ? r : 0), kk(use_svd ? r : 0);
        double rho = 100.0, prev = 1e300, eta = 0.05, cons = 0;
        long tot_it = 0;

        auto merit = [&](const vector<double>& xx, const vector<double>& vv,
                         const vector<double>& rrn) {
            double f = bpr.beckmann(vv), lr = 0, pq = 0, pen = 0;
            for (long q = 0; q < P.n_od; ++q) { lr += lam[q]*rrn[q]; pq += rrn[q]*rrn[q]; }
            #pragma omp parallel for reduction(+:pen) schedule(static)
            for (long p = 0; p < P.n; ++p) {
                double m = std::max(0.0, mu[p] - c2 * xx[p]);
                pen += m * m - mu[p] * mu[p];
            }
            return f + lr + 0.5 * rho * pq + 0.5 / c2 * pen;
        };

        for (long outer = 0; outer < max_outer; ++outer) {
            for (long inner = 0; inner < 30; ++inner) {
                scatter_links(P, allids, x, v);
                bpr.times(v, tl);
                bpr.deriv(v, W);
                std::fill(rr.begin(), rr.end(), 0.0);
                od_accumulate(P, allids, x, rr);
                for (long q = 0; q < P.n_od; ++q) rr[q] -= P.d[q];
                gather_paths(P, allids, tl, g);
                #pragma omp parallel for schedule(static)
                for (long p = 0; p < P.n; ++p) {
                    g[p] += lam[P.p2od[p]] + rho * rr[P.p2od[p]];
                    double phi = std::max(0.0, mu[p] - c2 * x[p]);   // nonneg ALM
                    g[p] -= phi;
                    Dg[p] = rho + (mu[p] - c2 * x[p] > 0 ? c2 : 0.0);
                }
                double gn = nrm2(g), xn0 = nrm2(x);
                if (gn / (1.0 + xn0) < eta && inner > 0) break;

                vector<double> dx(P.n);
                if (use_svd) {
                    // C = diag(s) Q'WQ diag(s) + eps I
                    std::fill(C.begin(), C.end(), 0.0);
                    #pragma omp parallel
                    {
                        vector<double> loc(r * r, 0.0);
                        #pragma omp for schedule(static)
                        for (long a = 0; a < P.m; ++a) {
                            const double* qa = &P.Qmat[a * r];
                            double wa = W[a];
                            for (long j = 0; j < r; ++j) { double t = wa*qa[j];
                                for (long l = 0; l < r; ++l) loc[j*r+l] += t*qa[l]; }
                        }
                        #pragma omp critical
                        for (long i = 0; i < r*r; ++i) C[i] += loc[i];
                    }
                    for (long j = 0; j < r; ++j)
                        for (long l = 0; l < r; ++l)
                            C[j*r+l] = P.sB[j]*C[j*r+l]*P.sB[l] + (j==l ? 1e-8 : 0.0);
                    chol_inverse(C, r, Cinv);
                    // Bmat = P' diag(1/Dg) P ; Ug = P'(g/Dg)
                    std::fill(Bmat.begin(), Bmat.end(), 0.0);
                    std::fill(Ug.begin(), Ug.end(), 0.0);
                    #pragma omp parallel
                    {
                        vector<double> lb(r*r, 0.0), lu(r, 0.0);
                        #pragma omp for schedule(static)
                        for (long p = 0; p < P.n; ++p) {
                            const double* pp = &P.Pmat[p*r];
                            double di = 1.0/Dg[p], gd = g[p]*di;
                            for (long j = 0; j < r; ++j) { double pj = pp[j];
                                lu[j] += pj*gd;
                                double t = pj*di;
                                for (long l = 0; l < r; ++l) lb[j*r+l] += t*pp[l]; }
                        }
                        #pragma omp critical
                        { for (long i=0;i<r*r;++i) Bmat[i]+=lb[i];
                          for (long j=0;j<r;++j) Ug[j]+=lu[j]; }
                    }
                    for (long i = 0; i < r*r; ++i) K[i] = Cinv[i] + Bmat[i];
                    chol_solve(K, r, Ug, kk);        // w = K^{-1} P'(g/Dg)
                    // dx = -(1/Dg) .* (g - P kk)
                    #pragma omp parallel for schedule(static)
                    for (long p = 0; p < P.n; ++p) {
                        const double* pp = &P.Pmat[p*r];
                        double pw = 0; for (long j=0;j<r;++j) pw += pp[j]*kk[j];
                        dx[p] = -(g[p] - pw) / Dg[p];
                    }
                } else {
                    // ALM-Newton: CG on (BWB' + Dg) dx = -g, precond 1/Dg
                    vector<double> rC(P.n), z(P.n), pC(P.n), Ap(P.n), Btp(P.m);
                    for (long p = 0; p < P.n; ++p) { dx[p] = 0; rC[p] = -g[p]; }
                    for (long p = 0; p < P.n; ++p) { z[p] = rC[p]/Dg[p]; pC[p] = z[p]; }
                    double rz = 0; for (long p=0;p<P.n;++p) rz += rC[p]*z[p];
                    for (int cgit = 0; cgit < 200; ++cgit) {
                        scatter_links(P, allids, pC, Btp);       // B' pC
                        #pragma omp parallel for schedule(static)
                        for (long a=0;a<P.m;++a) Btp[a] *= W[a];
                        gather_paths(P, allids, Btp, Ap);         // B (W B' pC)
                        #pragma omp parallel for schedule(static)
                        for (long p=0;p<P.n;++p) Ap[p] += Dg[p]*pC[p];
                        double pAp=0; for (long p=0;p<P.n;++p) pAp += pC[p]*Ap[p];
                        double alpha = rz / std::max(pAp, 1e-30);
                        #pragma omp parallel for schedule(static)
                        for (long p=0;p<P.n;++p){ dx[p]+=alpha*pC[p]; rC[p]-=alpha*Ap[p]; }
                        double rn=nrm2(rC); if (rn < 1e-6*(1.0+gn)) break;
                        for (long p=0;p<P.n;++p) z[p]=rC[p]/Dg[p];
                        double rz2=0; for (long p=0;p<P.n;++p) rz2+=rC[p]*z[p];
                        double beta=rz2/std::max(rz,1e-30); rz=rz2;
                        #pragma omp parallel for schedule(static)
                        for (long p=0;p<P.n;++p) pC[p]=z[p]+beta*pC[p];
                    }
                }
                // arc line search (no projection)
                vector<double> x_new(P.n), v_new(P.m), rr_new(P.n_od);
                double L_curr = merit(x, v, rr), gTdx = 0;
                for (long p=0;p<P.n;++p) gTdx += g[p]*dx[p];
                double alpha = 1.0; bool ok = false;
                for (int ls = 0; ls < 30; ++ls) {
                    #pragma omp parallel for schedule(static)
                    for (long p=0;p<P.n;++p) x_new[p]=x[p]+alpha*dx[p];
                    scatter_links(P, allids, x_new, v_new);
                    std::fill(rr_new.begin(), rr_new.end(), 0.0);
                    od_accumulate(P, allids, x_new, rr_new);
                    for (long q=0;q<P.n_od;++q) rr_new[q]-=P.d[q];
                    if (merit(x_new, v_new, rr_new) <= L_curr + 1e-4*alpha*gTdx
                        || alpha < 1e-13) { ok = gTdx < 0; break; }
                    alpha *= 0.5;
                }
                x.swap(x_new);
                tot_it++;
                if (!ok) break;
            }
            std::fill(rr.begin(), rr.end(), 0.0);
            od_accumulate(P, allids, x, rr);
            for (long q=0;q<P.n_od;++q) rr[q]-=P.d[q];
            cons = nrm2(rr);
            double negx = 0;
            for (long p=0;p<P.n;++p){ double m=std::min(x[p],0.0); negx+=m*m; }
            negx = std::sqrt(negx);
            printf("  outer %ld tot %ld cons %.5f negx %.2f rho %.1e t %.1fs\n",
                   outer+1, tot_it, cons, negx, rho, secs(t0c, clk::now()));
            fflush(stdout);
            for (long q=0;q<P.n_od;++q) lam[q]+=rho*rr[q];
            #pragma omp parallel for schedule(static)
            for (long p=0;p<P.n;++p) mu[p]=std::max(0.0, mu[p]-c2*x[p]);
            if (cons < tol_cons && negx < 1e-3 && eta <= 1e-5) break;
            if (cons > 0.25*prev) rho = std::min(rho*2.0, 1e6);
            prev = cons;
            eta = std::max(eta*0.5, 1e-6);
        }
        vector<double> odsum(P.n_od, 0.0);
        od_accumulate(P, allids, x, odsum);
        #pragma omp parallel for schedule(static)
        for (long p=0;p<P.n;++p){ double s=odsum[P.p2od[p]];
            x[p]=std::max(x[p],0.0)*((s>1e-12)?P.d[P.p2od[p]]/s:1.0); }
        scatter_links(P, allids, x, v);
        printf("RESULT mode=%s r_used=%ld time_s=%.1f inner_iters=%ld obj_feasible=%.4f cons=%.6f\n",
               mode.c_str(), use_svd ? r : 0L, secs(t0c, clk::now()), tot_it,
               bpr.beckmann(v), cons);
        if (!vdump.empty()) dump_v(vdump, v);
        return 0;
    }

    // ---------------- compressed modes: variables w = [y (majors); z (r)]
    if (P.r == 0 || P.U.empty()) {
        fprintf(stderr, "mode %s needs compressed blocks (U/D/M...) but the "
                "dump has r=0 — export them first\n", mode.c_str());
        return 1;
    }
    long s_dim = P.n_major, r = P.r, nm = P.n_minor;
    vector<double> w(s_dim + r, 0.0);
    for (long k = 0; k < s_dim; ++k) w[k] = std::max(P.x0[P.majid[k]], 1e-3);
    vector<double> mu;                       // hard-mode multipliers
    if (mode == "hard") mu.assign(nm, 0.0);
    vector<uint8_t> inW(nm, 0);              // screen working set
    const double c2 = 1e3, mu_soft = 1e3;

    vector<double> v(P.m), tl(P.m), rr(P.n_od), u(nm), zvec(r), gv(P.m);
    vector<double> ysub(s_dim), gy(s_dim), gz(r), tmp_r(r), Mz(P.n_od), Dz(P.m);
    double cons = 0, eta = 0.05;
    long tot_it = 0;
    double row_work_M = 0;                   // dense minor-row work (millions)

    // active hinges (production-spec rule "never scan minors in-loop"):
    // the nonneg hinge is evaluated only on rows within a safety band of
    // their floor (plus rows with mu>0 or in the screen working set),
    // rebuilt from the full minor pass at every outer boundary; all
    // termination checks (cons, R+, viol) remain full-space.
    // Measured A/B (port_ab.log 2026-07-11): Sketch band is 96% dense
    // (no saving); Regional band 14% saves 3x row work but the inexact
    // hinge doubles inner iterations (987s vs 555s) — net loss on both.
    // Therefore OPT-IN: enable with ACTIVE_HINGE=1.
    const char* ah_env = getenv("ACTIVE_HINGE");
    const bool use_act = (ah_env && ah_env[0] == '1') && mode != "recover";
    vector<int32_t> act;
    auto rebuild_act = [&](const vector<double>& ufull) {
        act.clear();
        for (long i = 0; i < nm; ++i) {
            double ui = P.x0m[i] + ufull[i];
            bool in = ui < 0.05 * (P.x0m[i] + 1.0);
            if (mode == "hard" && mu[i] > 0.0) in = true;
            if (mode == "screen") in = inW[i];
            if (in) act.push_back((int32_t)i);
        }
    };
    if (use_act) {                            // z = 0 at start -> u = 0
        std::fill(u.begin(), u.end(), 0.0);
        rebuild_act(u);
        printf("active-hinge: %ld / %ld minor rows in initial band (%.2f%%)\n",
               (long)act.size(), nm, 100.0 * act.size() / std::max(nm, 1L));
    }

    for (long outer = 0; outer < max_outer; ++outer) {
        auto fg = [&](const vector<double>& wv, vector<double>& g) {
            for (long j = 0; j < r; ++j) zvec[j] = wv[s_dim + j];
            for (long k = 0; k < s_dim; ++k) ysub[k] = wv[k];
            gemv(P.D, P.m, r, zvec, Dz, false);
            scatter_links(P, P.majid, ysub, v);
            #pragma omp parallel for schedule(static)
            for (long a = 0; a < P.m; ++a) v[a] += P.v_base[a] + Dz[a];
            std::fill(rr.begin(), rr.end(), 0.0);
            od_accumulate(P, P.majid, ysub, rr);
            gemv(P.M, P.n_od, r, zvec, Mz, false);
            #pragma omp parallel for schedule(static)
            for (long q = 0; q < P.n_od; ++q) rr[q] += Mz[q] - P.d_eff[q];
            bpr.times(v, tl);
            // gy
            gather_paths(P, P.majid, tl, gy);
            #pragma omp parallel for schedule(static)
            for (long k = 0; k < s_dim; ++k)
                gy[k] += lam[P.p2od[P.majid[k]]] + rho * rr[P.p2od[P.majid[k]]];
            // gz = D' t + M'(lam + rho rr)
            gemv(P.D, P.m, r, tl, gz, true);
            vector<double> lr(P.n_od);
            #pragma omp parallel for schedule(static)
            for (long q = 0; q < P.n_od; ++q) lr[q] = lam[q] + rho * rr[q];
            gemv(P.M, P.n_od, r, lr, tmp_r, true);
            for (long j = 0; j < r; ++j) gz[j] += tmp_r[j];
            double f = bpr.beckmann(v);
            double pq = 0, lrs = 0;
            #pragma omp parallel for reduction(+:pq, lrs) schedule(static)
            for (long q = 0; q < P.n_od; ++q) {
                pq += rr[q] * rr[q];
                lrs += lam[q] * rr[q];
            }
            f += lrs + 0.5 * rho * pq;
            // minor nonnegativity penalty per regime
            if (mode != "recover" && use_act) {
                // fused active-row pass: u_i, hinge, and gz scatter in one
                // sweep over the banded rows only (validated port)
                double fpen = 0;
                std::fill(tmp_r.begin(), tmp_r.end(), 0.0);
                row_work_M += 2.0 * act.size() / 1e6;
                for (long a2 = 0; a2 < (long)act.size(); ++a2) {
                    long i = act[a2];
                    const double* Ui = &P.U[i * r];
                    double ui = P.x0m[i];
                    for (long j = 0; j < r; ++j) ui += Ui[j] * zvec[j];
                    double wi;
                    if (mode == "hard") {
                        double phi = std::max(0.0, mu[i] - c2 * ui);
                        fpen += (phi * phi - mu[i] * mu[i]) / (2 * c2);
                        wi = -phi;
                    } else {
                        double neg = std::min(ui, 0.0);
                        fpen += 0.5 * mu_soft * neg * neg;
                        wi = mu_soft * neg;
                    }
                    if (wi != 0.0)
                        for (long j = 0; j < r; ++j) tmp_r[j] += wi * Ui[j];
                }
                for (long j = 0; j < r; ++j) gz[j] += tmp_r[j];
                f += fpen;
            } else if (mode != "recover") {
                gemv(P.U, nm, r, zvec, u, false);
                row_work_M += nm / 1e6;
                double fpen = 0;
                vector<double> wrow(nm, 0.0);
                if (mode == "hard") {
                    #pragma omp parallel for reduction(+:fpen) schedule(static)
                    for (long i = 0; i < nm; ++i) {
                        double ui = P.x0m[i] + u[i];
                        double phi = std::max(0.0, mu[i] - c2 * ui);
                        fpen += (phi * phi - mu[i] * mu[i]) / (2 * c2);
                        wrow[i] = -phi;
                    }
                } else {  // soft / screen
                    #pragma omp parallel for reduction(+:fpen) schedule(static)
                    for (long i = 0; i < nm; ++i) {
                        if (mode[0] == 's' && mode[1] == 'c' && !inW[i]) { wrow[i] = 0; continue; }
                        double ui = P.x0m[i] + u[i];
                        double neg = std::min(ui, 0.0);
                        fpen += 0.5 * mu_soft * neg * neg;
                        wrow[i] = mu_soft * neg;
                    }
                }
                gemv(P.U, nm, r, wrow, tmp_r, true);
                row_work_M += nm / 1e6;
                for (long j = 0; j < r; ++j) gz[j] += tmp_r[j];
                f += fpen;
            }
            for (long k = 0; k < s_dim; ++k) g[k] = gy[k];
            for (long j = 0; j < r; ++j) g[s_dim + j] = gz[j];
            return f;
        };
        auto proj = [&](vector<double>& wv) {
            #pragma omp parallel for schedule(static)
            for (long k = 0; k < s_dim; ++k) wv[k] = std::max(wv[k], 0.0);
        };
        SpgResult rres = plbfgs(w, fg, proj, max_inner, eta);
        tot_it += rres.iters;

        // in-pool AON gap history of the feasible-projected iterate, on
        // the SAME certificate as pool_masters (for the one-figure
        // ALM-vs-GP-vs-latgp comparison). HIST_CSV env enables it.
        if (const char* hcsv = getenv("HIST_CSV")) {
            static FILE* hf = nullptr;
            if (!hf) { hf = fopen(hcsv, "w");
                       if (hf) fprintf(hf, "elapsed_s,gap\n"); }
            if (hf) {
                vector<double> xh(P.n, 0.0), uh(P.n_minor);
                for (long k = 0; k < s_dim; ++k) xh[P.majid[k]] = w[k];
                vector<double> zh(r);
                for (long j = 0; j < r; ++j) zh[j] = w[s_dim + j];
                gemv(P.U, P.n_minor, r, zh, uh, false);
                for (long i = 0; i < P.n_minor; ++i)
                    xh[P.minid[i]] = std::max(P.x0m[i] + uh[i], 0.0);
                vector<double> odsum(P.n_od, 0.0);
                od_accumulate(P, allids, xh, odsum);
                for (long p = 0; p < P.n; ++p) {
                    double s = odsum[P.p2od[p]];
                    xh[p] *= (s > 1e-12) ? P.d[P.p2od[p]] / s : 1.0;
                }
                vector<double> vh(P.m), th(P.m), ch(P.n);
                scatter_links(P, allids, xh, vh);
                add_bg(P, vh);
                bpr.times(vh, th);
                gather_paths(P, allids, th, ch);
                vector<double> best(P.n_od, 1e300);
                double cx = 0, cy = 0;
                for (long p = 0; p < P.n; ++p) {
                    cx += ch[p] * xh[p];
                    if (ch[p] < best[P.p2od[p]]) best[P.p2od[p]] = ch[p];
                }
                for (long q = 0; q < P.n_od; ++q) cy += P.d[q] * best[q];
                fprintf(hf, "%.6f,%.10e\n", secs(t0c, clk::now()),
                        (cx - cy) / std::max(cy, 1e-12));
                fflush(hf);
            }
        }

        for (long j = 0; j < r; ++j) zvec[j] = w[s_dim + j];
        for (long k = 0; k < s_dim; ++k) ysub[k] = w[k];
        std::fill(rr.begin(), rr.end(), 0.0);
        od_accumulate(P, P.majid, ysub, rr);
        gemv(P.M, P.n_od, r, zvec, Mz, false);
        for (long q = 0; q < P.n_od; ++q) rr[q] += Mz[q] - P.d_eff[q];
        cons = nrm2(rr);
        gemv(P.U, nm, r, zvec, u, false);
        double Rp = 0;
        long nviol = 0;
        for (long i = 0; i < nm; ++i) {
            double ui = P.x0m[i] + u[i];
            if (ui < 0) { Rp += ui * ui; ++nviol; }
        }
        Rp = std::sqrt(Rp);
        printf("  outer %ld iters %ld cons %.5f R+ %.3f viol %ld act %.1f%% t %.1fs\n",
               outer + 1, rres.iters, cons, Rp, nviol,
               use_act ? 100.0 * act.size() / std::max(nm, 1L) : 100.0,
               secs(t0c, clk::now()));
        fflush(stdout);
        if (mode == "hard")
            for (long i = 0; i < nm; ++i)
                mu[i] = std::max(0.0, mu[i] - c2 * (P.x0m[i] + u[i]));
        if (mode == "screen")
            for (long i = 0; i < nm; ++i)
                if (P.x0m[i] + u[i] < -1e-6) inW[i] = 1;
        if (use_act) rebuild_act(u);   // band refresh from the full pass
        bool feas_ok = (mode == "recover") || Rp < 1e-3 ||
                       (mode == "screen" && nviol == 0);
        if (cons < tol_cons && feas_ok && eta <= 1e-5) break;
        for (long q = 0; q < P.n_od; ++q) lam[q] += rho * rr[q];
        if (cons > 0.25 * prev_cons) rho = std::min(rho * 2.0, 1e6);
        prev_cons = cons;
        eta = std::max(eta * 0.5, 1e-6);
    }

    // reconstruct x, clip minors, per-OD rescale, evaluate
    vector<double> x(P.n, 0.0);
    for (long k = 0; k < s_dim; ++k) x[P.majid[k]] = w[k];
    for (long j = 0; j < r; ++j) zvec[j] = w[s_dim + j];
    gemv(P.U, nm, r, zvec, u, false);
    for (long i = 0; i < nm; ++i) x[P.minid[i]] = std::max(P.x0m[i] + u[i], 0.0);
    vector<double> odsum(P.n_od, 0.0);
    od_accumulate(P, allids, x, odsum);
    double sx = 0, sd = 0, ssum = 0, zero_od = 0;
    for (long p = 0; p < P.n; ++p) sx += x[p];
    for (long q = 0; q < P.n_od; ++q) {
        sd += P.d[q];
        ssum += odsum[q];
        if (odsum[q] <= 1e-12 && P.d[q] > 0) zero_od += P.d[q];
    }
    printf("  recon: sum_x=%.1f sum_odsum=%.1f sum_d=%.1f demand_on_zero_ods=%.1f\n",
           sx, ssum, sd, zero_od);
    // raw-x dump (added 2026-07-24, purely additive): the UNPROJECTED reconstructed path
    // flows, so Python can compute the v3 delta_F/Gap_F/R2 through the SAME v3_metrics used
    // for the Python solves. Written only when VDUMP_XRAW names a path; default off.
    if (const char* xr = getenv("VDUMP_XRAW")) {
        FILE* fx = fopen(xr, "wb");
        if (fx) { fwrite(x.data(), sizeof(double), x.size(), fx); fclose(fx); }
    }
    #pragma omp parallel for schedule(static)
    for (long p = 0; p < P.n; ++p) {
        double s = odsum[P.p2od[p]];
        x[p] *= (s > 1e-12) ? P.d[P.p2od[p]] / s : 1.0;
    }
    scatter_links(P, allids, x, v);
    add_bg(P, v);
    printf("RESULT mode=%s r_used=%ld time_s=%.1f inner_iters=%ld obj_feasible=%.4f "
           "cons=%.6f row_work_M=%.0f pool_gap=%.4e\n",
           mode.c_str(), P.r, secs(t0c, clk::now()), tot_it, bpr.beckmann(v), cons, row_work_M,
           pool_gap(P, bpr, allids, x));
    if (!vdump.empty()) dump_v(vdump, v);
    return 0;
}
