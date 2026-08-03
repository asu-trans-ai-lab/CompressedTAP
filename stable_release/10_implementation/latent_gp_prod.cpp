// Phase-1 production Latent GP (CPP_PRODUCTION_SPEC): sparse response
// columns, one-pass gradient/curvature kernel, incremental link-cost
// updates on column support only, quadratic-prediction acceptance (no
// line search), adaptive certificate schedule. Single thread.
// Fairness: the FULL-GP baseline in this binary uses the SAME kernels.
//
// usage: latent_gp_prod <pool_dir> [rho_M] [out_csv]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <ctime>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>

struct Link { double bg, t0, cap, al, be; };
static std::vector<Link> L;
static std::vector<double> vol, cst, drv;

static inline double bpr(double v, const Link& l) {
    double x = (l.bg + v) / l.cap;
    return l.t0 * (1.0 + l.al * x * x * x * x);
}
static inline double bprd(double v, const Link& l) {
    double x = (l.bg + v) / l.cap;
    return l.t0 * l.al * 4.0 * x * x * x / l.cap;
}
static inline double beck_link(double v, const Link& l) {
    double x = (l.bg + v) / l.cap;
    return l.t0 * ((l.bg + v) + l.al * l.cap / 5.0 * x * x * x * x * x);
}

struct Col { uint32_t off, len; double lo, hi, flow; };
static std::vector<uint32_t> ids;   // link ids, all columns contiguous
static std::vector<double> cf;      // coefficients

static FILE* g_hist = nullptr;       // per-iteration history (stage,it,t,gap)
static const char* g_stage = "";

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <pool> [rhoM] [csv]\n",
                            argv[0]); return 1; }
    std::string dir = argv[1];
    double rhoM = argc > 2 ? atof(argv[2]) : 0.75;
    if (argc > 3) {
        g_hist = fopen(argv[3], "w");
        if (g_hist) fprintf(g_hist, "stage,iter,time_s,gap\n");
    }
    // ---- load pool (links.csv background,t0,capacity,alpha,beta,toll)
    {
        std::ifstream f(dir + "/links.csv");
        std::string ln; std::getline(f, ln);
        while (std::getline(f, ln)) {
            Link l; double toll;
            sscanf(ln.c_str(), "%lf,%lf,%lf,%lf,%lf,%lf",
                   &l.bg, &l.t0, &l.cap, &l.al, &l.be, &toll);
            if (l.t0 < 1e-4) l.t0 = 1e-4;
            L.push_back(l);
        }
    }
    double q; long m, K;
    { std::ifstream f(dir + "/meta.txt"); int o, d;
      f >> o >> d >> q >> m >> K; }
    double qmul = argc > 4 ? atof(argv[4]) : 1.0;   // demand overload factor
    q *= qmul;
    if (qmul != 1.0) printf("demand overload: q x%.2f = %.1f\n", qmul, q);
    std::vector<std::vector<uint32_t>> paths;
    {
        std::ifstream f(dir + "/paths.txt");
        std::string ln;
        while (std::getline(f, ln)) {
            if (ln.empty()) continue;
            std::vector<uint32_t> p; std::stringstream ss(ln);
            std::string tok;
            while (std::getline(ss, tok, ';'))
                p.push_back((uint32_t)atoi(tok.c_str()));
            paths.push_back(p);
        }
    }
    K = (long)paths.size(); m = (long)L.size();
    vol.assign(m, 0.0); cst.resize(m); drv.resize(m);
    for (long a = 0; a < m; ++a) { cst[a] = bpr(0, L[a]);
                                   drv[a] = bprd(0, L[a]); }
    // ---- oracle split: deep full-GP (same kernels) to build atom
    auto path_cost = [&](const std::vector<uint32_t>& p) {
        double s = 0; for (uint32_t a : p) s += cst[a]; return s; };
    // full-GP state
    std::vector<double> x(K, 0.0); x[0] = q;
    for (uint32_t a : paths[0]) { vol[a] += q; }
    for (long a = 0; a < m; ++a) { cst[a] = bpr(vol[a], L[a]);
                                   drv[a] = bprd(vol[a], L[a]); }
    auto full_sweep = [&](double tol, int iters, double* t_out,
                          double marks[3]) {
        clock_t t0 = clock();
        marks[0] = marks[1] = marks[2] = -1;
        for (int it = 0; it < iters; ++it) {
            // price all paths (this IS the K*Lbar work)
            int anc = 0; double cb = 1e300;
            static std::vector<double> c; c.resize(K);
            for (long k = 0; k < K; ++k) {
                double s = 0; for (uint32_t a : paths[k]) s += cst[a];
                c[k] = s; if (s < cb) { cb = s; anc = (int)k; }
            }
            double cx = 0; for (long k = 0; k < K; ++k) cx += c[k] * x[k];
            double gap = (cx - q * cb) / (q * cb);
            double el = double(clock() - t0) / CLOCKS_PER_SEC;
            if (g_hist) fprintf(g_hist, "%s,%d,%.6f,%.6e\n",
                                g_stage, it, el, gap);
            const double th[3] = {1e-4, 3e-5, 1e-5};
            for (int i = 0; i < 3; ++i)
                if (marks[i] < 0 && gap <= th[i]) marks[i] = el;
            if (gap <= tol) break;
            for (long k = 0; k < K; ++k) {
                if (k == anc || x[k] <= 0) continue;
                // fresh residuals: g and S from CURRENT costs (stale c[]
                // from the sweep-top pricing overshoots once the anchor
                // loads up mid-sweep and the sweep stalls at a gap floor)
                double sk = 0, sa = 0, S = 0;
                for (uint32_t a : paths[k])   { sk += cst[a]; S += drv[a]; }
                for (uint32_t a : paths[anc]) { sa += cst[a]; S += drv[a]; }
                double g = sk - sa;
                double d = -g / (S + 1e-12);
                if (d < -x[k]) d = -x[k];
                if (d > x[anc]) d = x[anc];
                if (d == 0) continue;
                x[k] += d; x[anc] -= d;
                for (uint32_t a : paths[k]) { vol[a] += d;
                    cst[a] = bpr(vol[a], L[a]);
                    drv[a] = bprd(vol[a], L[a]); }
                for (uint32_t a : paths[anc]) { vol[a] -= d;
                    cst[a] = bpr(vol[a], L[a]);
                    drv[a] = bprd(vol[a], L[a]); }
            }
        }
        *t_out = double(clock() - t0) / CLOCKS_PER_SEC;
    };
    double t_oracle, mk_o[3];
    g_stage = "oracle";
    full_sweep(1e-7, 20000, &t_oracle, mk_o);
    {   // report the oracle's actual terminal gap (diagnostic)
        double cb = 1e300, cx = 0;
        for (long k = 0; k < K; ++k) {
            double s = 0; for (uint32_t a : paths[k]) s += cst[a];
            if (s < cb) cb = s; cx += s * x[k];
        }
        printf("oracle full-GP: %.2fs gap %.2e (t1e-4 %.2fs)\n",
               t_oracle, (cx - q * cb) / (q * cb), mk_o[0]);
    }
    // split: anchor = min-cost positive; majors = rhoM of non-anchor
    int anchor = 0; { double cb = 1e300;
        for (long k = 0; k < K; ++k) if (x[k] > 1e-9) {
            double s = 0; for (uint32_t a : paths[k]) s += cst[a];
            if (s < cb) { cb = s; anchor = (int)k; } } }
    std::vector<long> order(K);
    for (long k = 0; k < K; ++k) order[k] = k;
    std::sort(order.begin(), order.end(),
              [&](long a, long b) { return x[a] > x[b]; });
    std::vector<char> is_major(K, 0);
    double nonanc = q - x[anchor], acc = 0;
    std::vector<long> majors;
    for (long k : order) {
        if (k == anchor) continue;
        if (acc >= rhoM * nonanc) break;
        is_major[k] = 1; majors.push_back(k); acc += x[k];
    }
    // atom = oracle minor flows (sparse merge)
    std::vector<double> abar(m, 0.0);
    double mu_star = 0;
    for (long k = 0; k < K; ++k)
        if (!is_major[k] && k != anchor && x[k] > 0) {
            mu_star += x[k];
            for (uint32_t a : paths[k]) abar[a] += x[k];
        }
    for (long a = 0; a < m; ++a) abar[a] /= std::max(mu_star, 1e-12);
    // ---- build SPARSE response columns (majors + one atom)
    std::vector<Col> cols;
    auto push_path_col = [&](long k) {
        Col c; c.off = (uint32_t)ids.size();
        // r = a_k - a_anchor via sorted merge
        std::vector<std::pair<uint32_t, double>> tmp;
        for (uint32_t a : paths[k]) tmp.push_back({a, 1.0});
        for (uint32_t a : paths[anchor]) tmp.push_back({a, -1.0});
        std::sort(tmp.begin(), tmp.end());
        for (size_t i = 0; i < tmp.size();) {
            uint32_t a = tmp[i].first; double s = 0;
            while (i < tmp.size() && tmp[i].first == a) s += tmp[i++].second;
            if (s != 0.0) { ids.push_back(a); cf.push_back(s); }
        }
        c.len = (uint32_t)(ids.size() - c.off);
        c.lo = 0; c.hi = q; c.flow = 0; cols.push_back(c);
    };
    for (long k : majors) push_path_col(k);
    if (mu_star > 1e-9) {   // atom response column (skip when the oracle
                            // equilibrium has no minor mass: abar=0 would
                            // otherwise become a flow-deleting column)
        Col c; c.off = (uint32_t)ids.size();
        std::vector<std::pair<uint32_t, double>> tmp;
        for (long a = 0; a < m; ++a)
            if (abar[a] > 1e-12) tmp.push_back({(uint32_t)a, abar[a]});
        for (uint32_t a : paths[anchor]) tmp.push_back({a, -1.0});
        std::sort(tmp.begin(), tmp.end());
        for (size_t i = 0; i < tmp.size();) {
            uint32_t a = tmp[i].first; double s = 0;
            while (i < tmp.size() && tmp[i].first == a) s += tmp[i++].second;
            if (std::fabs(s) > 1e-12) { ids.push_back(a); cf.push_back(s); }
        }
        c.len = (uint32_t)(ids.size() - c.off);
        c.lo = 0; c.hi = q; c.flow = 0; cols.push_back(c);
    }
    long ncol = (long)cols.size();
    printf("columns: %ld majors + 1 atom; atom support %u; mu*/q %.3f\n",
           ncol - 1, cols.back().len, mu_star / q);
    // ---- restart: all demand on anchor
    std::fill(vol.begin(), vol.end(), 0.0);
    for (uint32_t a : paths[anchor]) vol[a] += q;
    for (long a = 0; a < m; ++a) { cst[a] = bpr(vol[a], L[a]);
                                   drv[a] = bprd(vol[a], L[a]); }
    double f_anchor = q;
    // ---- production latent sweeps
    clock_t t0 = clock();
    double marks[3] = {-1, -1, -1};
    int Rc = 100;
    double gap = 1.0, prev_cert = 1e300;
    int n_refresh = 0;
    std::vector<char> promoted(K, 0);
    promoted[anchor] = 1;
    for (long k : majors) promoted[k] = 1;
    for (int it = 0; it < 200000; ++it) {
        if (it % Rc == 0) {
            // certificate: full-pool pricing (counted)
            double cb = 1e300; long kb = 0;
            for (long k = 0; k < K; ++k) {
                double s = 0; for (uint32_t a : paths[k]) s += cst[a];
                if (s < cb) { cb = s; kb = k; }
            }
            double cx = f_anchor * path_cost(paths[anchor]);
            for (long j = 0; j < ncol; ++j) {
                if (cols[j].flow == 0) continue;
                double s = 0;
                for (uint32_t p = cols[j].off;
                     p < cols[j].off + cols[j].len; ++p)
                    s += cf[p] * cst[ids[p]];
                cx += cols[j].flow * (s + path_cost(paths[anchor]) * 0);
            }
            // exact cx: anchor cost * total + sum flow * r'c
            cx = q * path_cost(paths[anchor]);
            for (long j = 0; j < ncol; ++j) {
                double s = 0;
                for (uint32_t p = cols[j].off;
                     p < cols[j].off + cols[j].len; ++p)
                    s += cf[p] * cst[ids[p]];
                cx += cols[j].flow * s;
            }
            gap = (cx - q * cb) / (q * cb);
            double el = double(clock() - t0) / CLOCKS_PER_SEC;
            if (g_hist) fprintf(g_hist, "latent,%d,%.6f,%.6e\n",
                                it, el, gap);
            const double th[3] = {1e-4, 3e-5, 1e-5};
            for (int i = 0; i < 3; ++i)
                if (marks[i] < 0 && gap <= th[i]) marks[i] = el;
            if (gap <= 1e-5) break;
            // E1R refresh (validated law: refresh removes the G_R->0 /
            // G_UE-plateau operating point): when the certificate stalls,
            // promote the pool's current min-cost path as a new explicit
            // response column. Construction cost is inside the clock.
            if (gap > 0.98 * prev_cert && !promoted[kb] &&
                n_refresh < 64) {
                push_path_col(kb);
                promoted[kb] = 1;
                ncol = (long)cols.size();
                ++n_refresh;
            }
            prev_cert = gap;
            Rc = gap > 1e-2 ? 100 : (gap > 1e-4 ? 25 : 1);
        }
        // one sweep over columns: one-pass kernel + incremental update
        for (long j = 0; j < ncol; ++j) {
            Col& c = cols[j];
            double g = 0, S = 0;
            for (uint32_t p = c.off; p < c.off + c.len; ++p) {
                uint32_t a = ids[p]; double r = cf[p];
                g += r * cst[a]; S += r * r * drv[a];
            }
            double d = -g / (S + 1e-12);
            if (d < -c.flow) d = -c.flow;
            if (d > f_anchor) d = f_anchor;
            if (d == 0.0) continue;
            // acceptance on the TRUE objective delta over the column's
            // support only (per-link Beckmann integrals, spec rule).
            // The quadratic model alone is blind when S ~ 0 on
            // uncongested links: the clipped step then overshoots and
            // two overlapping columns limit-cycle.
            auto dF_supp = [&](double dd) {
                double s = 0;
                for (uint32_t p = c.off; p < c.off + c.len; ++p) {
                    uint32_t a = ids[p];
                    s += beck_link(vol[a] + dd * cf[p], L[a])
                         - beck_link(vol[a], L[a]);
                }
                return s;
            };
            int hv = 0;
            while (dF_supp(d) > -1e-12 && hv++ < 30) d *= 0.5;
            if (hv > 30 || d == 0.0) continue;
            c.flow += d; f_anchor -= d;
            for (uint32_t p = c.off; p < c.off + c.len; ++p) {
                uint32_t a = ids[p];
                vol[a] += d * cf[p];
                cst[a] = bpr(vol[a], L[a]);
                drv[a] = bprd(vol[a], L[a]);
            }
        }
        if (f_anchor <= 1e-12) {
            // anchor-exhaustion rebalance (Newton-swapping form): with
            // f_anchor = 0 every anchor-mediated coordinate is blocked
            // even though column-to-column descent exists. Exchange each
            // column against the cheapest column b along r_j -> r_b.
            long b = -1; double gb = 1e300;
            std::vector<double> gj(ncol);
            for (long j = 0; j < ncol; ++j) {
                double s = 0;
                for (uint32_t p = cols[j].off;
                     p < cols[j].off + cols[j].len; ++p)
                    s += cf[p] * cst[ids[p]];
                gj[j] = s;
                if (s < gb) { gb = s; b = j; }
            }
            for (long j = 0; b >= 0 && j < ncol; ++j) {
                if (j == b || cols[j].flow <= 0) continue;
                if (gj[j] <= gb + 1e-12) continue;
                double Sj = 0, Sb = 0;
                for (uint32_t p = cols[j].off;
                     p < cols[j].off + cols[j].len; ++p)
                    Sj += cf[p] * cf[p] * drv[ids[p]];
                for (uint32_t p = cols[b].off;
                     p < cols[b].off + cols[b].len; ++p)
                    Sb += cf[p] * cf[p] * drv[ids[p]];
                double del = (gj[j] - gb) / (2.0 * (Sj + Sb) + 1e-12);
                if (del > cols[j].flow) del = cols[j].flow;
                if (del <= 0) continue;
                // net per-link change over the merged support
                std::vector<std::pair<uint32_t, double>> mg;
                for (uint32_t p = cols[j].off;
                     p < cols[j].off + cols[j].len; ++p)
                    mg.push_back({ids[p], -cf[p]});
                for (uint32_t p = cols[b].off;
                     p < cols[b].off + cols[b].len; ++p)
                    mg.push_back({ids[p], cf[p]});
                std::sort(mg.begin(), mg.end());
                size_t w = 0;
                for (size_t i2 = 0; i2 < mg.size();) {
                    uint32_t a = mg[i2].first; double s = 0;
                    while (i2 < mg.size() && mg[i2].first == a)
                        s += mg[i2++].second;
                    if (s != 0.0) mg[w++] = {a, s};
                }
                mg.resize(w);
                auto dF2 = [&](double dd) {
                    double s = 0;
                    for (auto& e : mg)
                        s += beck_link(vol[e.first] + dd * e.second,
                                       L[e.first])
                             - beck_link(vol[e.first], L[e.first]);
                    return s;
                };
                int hv = 0;
                while (dF2(del) > -1e-12 && hv++ < 30) del *= 0.5;
                if (hv > 30) continue;
                cols[j].flow -= del; cols[b].flow += del;
                for (auto& e : mg) {
                    vol[e.first] += del * e.second;
                    cst[e.first] = bpr(vol[e.first], L[e.first]);
                    drv[e.first] = bprd(vol[e.first], L[e.first]);
                }
                gj[b] = 0;  // recompute lazily next sweep
            }
        }
    }
    double t_lat = double(clock() - t0) / CLOCKS_PER_SEC;
    {   // reduced-space stationarity: g_j per column at exit
        double gmax_pos = 0, gmin = 1e300;
        for (long j = 0; j < ncol; ++j) {
            double s = 0;
            for (uint32_t p = cols[j].off; p < cols[j].off + cols[j].len; ++p)
                s += cf[p] * cst[ids[p]];
            if (cols[j].flow > 1e-9 && std::fabs(s) > gmax_pos)
                gmax_pos = std::fabs(s);
            if (s < gmin) gmin = s;
        }
        printf("PROD-LATENT: t(1e-4)=%.4fs t(3e-5)=%.4fs t(1e-5)=%.4fs "
               "total %.3fs gap %.2e G_R[|g|max@flow %.2e, gmin %.2e]\n",
               marks[0], marks[1], marks[2], t_lat, gap, gmax_pos, gmin);
    }
    // ---- matched full-GP from same restart (same kernels)
    std::fill(x.begin(), x.end(), 0.0); x[anchor] = q;
    std::fill(vol.begin(), vol.end(), 0.0);
    for (uint32_t a : paths[anchor]) vol[a] += q;
    for (long a = 0; a < m; ++a) { cst[a] = bpr(vol[a], L[a]);
                                   drv[a] = bprd(vol[a], L[a]); }
    double t_full, mk_f[3];
    g_stage = "gp";
    full_sweep(1e-5, 20000, &t_full, mk_f);
    {   // terminal gap of the matched run (diagnostic)
        double cb = 1e300, cx = 0;
        for (long k = 0; k < K; ++k) {
            double s = 0; for (uint32_t a : paths[k]) s += cst[a];
            if (s < cb) cb = s; cx += s * x[k];
        }
        printf("FULL-GP: t(1e-4)=%.4fs t(3e-5)=%.4fs t(1e-5)=%.4fs "
               "total %.3fs gap %.2e\n", mk_f[0], mk_f[1], mk_f[2],
               t_full, (cx - q * cb) / (q * cb));
    }
    printf("SPEEDUP S(1e-4)=%.2fx S(1e-5)=%.2fx\n",
           mk_f[0] / std::max(marks[0], 1e-9),
           mk_f[2] / std::max(marks[2], 1e-9));
    return 0;
}
