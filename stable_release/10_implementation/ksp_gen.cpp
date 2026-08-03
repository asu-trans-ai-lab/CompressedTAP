// ============================================================================
// ksp_gen.cpp — the project's customized penalty-KSP pool generator, compiled.
//
// Same deterministic algorithm as sf_rich_pools.py / gen_sf_pool.py: per OD,
// free-flow shortest path, then K_extra rounds of multiplying the last path's
// link costs by `penalty` and re-solving; the union of found paths is emitted.
// Python is fine for Sioux (528 ODs); Chicago Regional (~300k ODs x K rounds
// of Dijkstra on 13k nodes) needs this compiled version (author directive).
//
// build:  g++ -O2 -static -o ksp_gen.exe ksp_gen.cpp
// run:    ksp_gen <dataset_dir> <K_extra> [penalty=1.4] [out_csv]
//         reads <dir>/link.csv + <dir>/demand.csv (GMNS);
//         writes  <out_csv | dir/path_pool_KSP<K>.csv>  (ksp rows only; merge
//         with the base pool that carries volume_ref in Python afterwards).
// Progress: one line per 1000 origins. Single-threaded, deterministic
// (ties broken by node index, matching scipy's behavior closely but not
// guaranteed identically — validate pool STATISTICS, not byte equality).
// ============================================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <chrono>
#include <algorithm>

using std::vector;
using std::string;

struct CsvIdx {
    std::unordered_map<string, int> col;
    vector<string> split(const string& ln) {
        vector<string> f; string cur;
        for (char c : ln) {
            if (c == ',') { f.push_back(cur); cur.clear(); }
            else if (c != '\r') cur.push_back(c);
        }
        f.push_back(cur);
        return f;
    }
};

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <dataset_dir> <K_extra> [penalty] [out]\n",
                            argv[0]); return 1; }
    string dir = argv[1];
    int K_extra = atoi(argv[2]);
    double penalty = argc > 3 ? atof(argv[3]) : 1.4;
    string out = argc > 4 ? argv[4]
                          : dir + "/path_pool_KSP" + std::to_string(K_extra) + ".csv";
    auto t0 = std::chrono::steady_clock::now();

    // ---- link.csv
    CsvIdx cx;
    FILE* f = fopen((dir + "/link.csv").c_str(), "r");
    if (!f) { fprintf(stderr, "no link.csv\n"); return 1; }
    char buf[1 << 16];
    fgets(buf, sizeof buf, f);
    {
        auto h = cx.split(buf);
        for (size_t i = 0; i < h.size(); ++i) cx.col[h[i]] = (int)i;
    }
    auto want = [&](const char* a, const char* b) {
        if (cx.col.count(a)) return cx.col[a];
        if (b && cx.col.count(b)) return cx.col[b];
        return -1;
    };
    int c_id = want("link_id", nullptr), c_fr = want("from_node_id", nullptr),
        c_to = want("to_node_id", nullptr), c_fftt = want("vdf_fftt", nullptr),
        c_len = want("vdf_length_mi", "length"), c_spd = want("vdf_free_speed_mph",
                                                              "free_speed");
    vector<long> lid; vector<long> frN, toN; vector<double> fftt;
    while (fgets(buf, sizeof buf, f)) {
        auto r = cx.split(buf);
        if ((int)r.size() <= c_to) continue;
        lid.push_back(atol(r[c_id].c_str()));
        frN.push_back(atol(r[c_fr].c_str()));
        toN.push_back(atol(r[c_to].c_str()));
        double t = (c_fftt >= 0 && !r[c_fftt].empty()) ? atof(r[c_fftt].c_str()) : 0.0;
        if (t <= 0) {
            double ln = c_len >= 0 ? atof(r[c_len].c_str()) : 0.0;
            double sp = c_spd >= 0 ? atof(r[c_spd].c_str()) : 30.0;
            if (sp <= 0) sp = 30.0;
            t = 60.0 * ln / sp;
        }
        fftt.push_back(t > 1e-4 ? t : 1e-4);
    }
    fclose(f);
    long m = (long)lid.size();

    // node numbering
    std::unordered_map<long, int> n2i;
    for (long e = 0; e < m; ++e) {
        if (!n2i.count(frN[e])) n2i[frN[e]] = (int)n2i.size();
        if (!n2i.count(toN[e])) n2i[toN[e]] = (int)n2i.size();
    }
    int nn = (int)n2i.size();
    // CSR adjacency (edge list per from-node), first-parallel-edge kept like edge_of
    vector<int> head(nn + 1, 0);
    vector<int> efr(m), eto(m);
    for (long e = 0; e < m; ++e) { efr[e] = n2i[frN[e]]; eto[e] = n2i[toN[e]]; }
    for (long e = 0; e < m; ++e) head[efr[e] + 1]++;
    for (int i = 0; i < nn; ++i) head[i + 1] += head[i];
    vector<int> adj(m); vector<int> pos(head.begin(), head.end() - 1);
    for (long e = 0; e < m; ++e) adj[pos[efr[e]]++] = (int)e;

    // ---- demand.csv -> OD list grouped by origin
    f = fopen((dir + "/demand.csv").c_str(), "r");
    if (!f) { fprintf(stderr, "no demand.csv\n"); return 1; }
    fgets(buf, sizeof buf, f);
    CsvIdx dx; { auto h = dx.split(buf); for (size_t i = 0; i < h.size(); ++i) dx.col[h[i]] = (int)i; }
    int d_o = dx.col["o_zone_id"], d_d = dx.col["d_zone_id"], d_v = dx.col["volume"];
    std::unordered_map<long, vector<long>> by_origin;
    long n_od = 0;
    while (fgets(buf, sizeof buf, f)) {
        auto r = dx.split(buf);
        if ((int)r.size() <= d_v) continue;
        long o = atol(r[d_o].c_str()), d = atol(r[d_d].c_str());
        double v = atof(r[d_v].c_str());
        if (v <= 0 || o == d || !n2i.count(o) || !n2i.count(d)) continue;
        by_origin[o].push_back(d);
        ++n_od;
    }
    fclose(f);

    // ---- Dijkstra (binary heap) with predecessor EDGES
    vector<double> dist(nn);
    vector<int> pedge(nn);
    auto dijkstra = [&](int src, const vector<double>& cost) {
        std::fill(dist.begin(), dist.end(), 1e300);
        std::fill(pedge.begin(), pedge.end(), -1);
        using QN = std::pair<double, int>;
        std::priority_queue<QN, vector<QN>, std::greater<QN>> pq;
        dist[src] = 0; pq.push({0.0, src});
        while (!pq.empty()) {
            auto [dc, u] = pq.top(); pq.pop();
            if (dc > dist[u] + 1e-15) continue;
            for (int k = head[u]; k < head[u + 1]; ++k) {
                int e = adj[k], v = eto[e];
                double nd = dc + cost[e];
                if (nd < dist[v] - 1e-15) {
                    dist[v] = nd; pedge[v] = e; pq.push({nd, v});
                }
            }
        }
    };
    auto extract = [&](int src, int dst, vector<int>& out) {
        out.clear(); int cur = dst, guard = 0;
        while (cur != src && guard++ < 5000) {
            int e = pedge[cur];
            if (e < 0) { out.clear(); return false; }
            out.push_back(e); cur = efr[e];
        }
        std::reverse(out.begin(), out.end());
        return cur == src;
    };

    FILE* fo = fopen(out.c_str(), "w");
    fprintf(fo, "o_zone_id,d_zone_id,link_ids,prob_ref,volume_ref,cost_base,source,path_id\n");
    long written = 0, o_done = 0;
    vector<double> cost(m);
    vector<int> pth;
    for (auto& [o, dests] : by_origin) {
        int o_i = n2i[o];
        dijkstra(o_i, fftt);
        vector<int> pred0(pedge);           // base tree
        for (long dz : dests) {
            int d_i = n2i[dz];
            std::unordered_set<string> found;
            pedge = pred0;
            if (!extract(o_i, d_i, pth)) continue;
            auto emit = [&](const vector<int>& p) {
                string key; key.reserve(p.size() * 6);
                double cb = 0;
                for (int e : p) { key += std::to_string(lid[e]); key += ';'; cb += fftt[e]; }
                if (!found.insert(key).second) return;
                key.pop_back();
                fprintf(fo, "%ld,%ld,%s,,,%.6f,ksp,%ld\n", o, dz, key.c_str(), cb, written);
                ++written;
            };
            emit(pth);
            std::copy(fftt.begin(), fftt.end(), cost.begin());
            vector<int> last = pth;
            for (int k = 0; k < K_extra; ++k) {
                for (int e : last) cost[e] *= penalty;
                dijkstra(o_i, cost);
                if (!extract(o_i, d_i, pth)) break;
                last = pth;
                emit(pth);
            }
        }
        if (++o_done % 1000 == 0) {
            fprintf(stderr, "origins %ld  paths %ld  %.0fs\n", o_done, written,
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
                        .count());
        }
    }
    fclose(fo);
    printf("KSPGEN done: %ld ODs, %ld paths -> %s (%.0fs)\n", n_od, written, out.c_str(),
           std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    return 0;
}
