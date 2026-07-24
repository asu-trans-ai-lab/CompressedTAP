// Standalone C++17 verification of full-path vs grouped-atom FW kernels.
// No third-party libraries. Reads binary cases exported by
// run_fw_origin_experiments.py and benchmarks:
//   1) row costs M * link_cost,
//   2) link loading M^T * flow,
//   3) OD-block LMO scans.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

struct CSR {
    int64_t rows = 0, cols = 0;
    std::vector<int64_t> row_ptr;
    std::vector<int32_t> col_idx;
    std::vector<double> val;
};

template <typename T>
void read_vec(std::ifstream& f, std::vector<T>& x, size_t n) {
    x.resize(n);
    f.read(reinterpret_cast<char*>(x.data()), static_cast<std::streamsize>(n * sizeof(T)));
    if (!f) throw std::runtime_error("Unexpected end of binary case");
}

void row_cost(const CSR& M, const std::vector<double>& link_cost, std::vector<double>& out) {
    std::fill(out.begin(), out.end(), 0.0);
    for (int64_t r = 0; r < M.rows; ++r) {
        double s = 0.0;
        for (int64_t q = M.row_ptr[r]; q < M.row_ptr[r + 1]; ++q)
            s += M.val[q] * link_cost[M.col_idx[q]];
        out[r] = s;
    }
}

void link_load(const CSR& M, const std::vector<double>& flow, std::vector<double>& out) {
    std::fill(out.begin(), out.end(), 0.0);
    for (int64_t r = 0; r < M.rows; ++r)
        for (int64_t q = M.row_ptr[r]; q < M.row_ptr[r + 1]; ++q)
            out[M.col_idx[q]] += M.val[q] * flow[r];
}

double lmo_scan(const std::vector<double>& cost, const std::vector<int64_t>& od_ptr) {
    double checksum = 0.0;
    for (size_t i = 0; i + 1 < od_ptr.size(); ++i) {
        auto first = cost.begin() + od_ptr[i];
        auto last = cost.begin() + od_ptr[i + 1];
        checksum += *std::min_element(first, last);
    }
    return checksum;
}

double benchmark(const CSR& M, const std::vector<int64_t>& od_ptr, int repeats) {
    std::vector<double> link_cost(M.cols), flow(M.rows), costs(M.rows), load(M.cols);
    for (int64_t a = 0; a < M.cols; ++a) link_cost[a] = 1.0 + 0.001 * (a % 101);
    for (int64_t r = 0; r < M.rows; ++r) flow[r] = 1.0 / std::max<int64_t>(M.rows, 1);
    volatile double sink = 0.0;
    auto start = std::chrono::steady_clock::now();
    for (int k = 0; k < repeats; ++k) {
        row_cost(M, link_cost, costs);
        link_load(M, flow, load);
        sink += lmo_scan(costs, od_ptr) + load[k % std::max<int64_t>(M.cols, 1)];
    }
    auto end = std::chrono::steady_clock::now();
    const double sec = std::chrono::duration<double>(end - start).count();
    if (sink == -1.23456789) std::cerr << sink;
    return sec / repeats;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: fw_atom_kernel <cpp_case.bin> [repeats]\n";
        return 2;
    }
    int repeats = argc >= 3 ? std::stoi(argv[2]) : 20000;
    std::ifstream f(argv[1], std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open case file");
    char magic[8]; f.read(magic, 8);
    if (std::string(magic, 8) != "FWATOM01") throw std::runtime_error("Bad case magic");
    int64_t h[6]; f.read(reinterpret_cast<char*>(h), sizeof(h));
    const int64_t nlinks=h[0], frows=h[1], fnnz=h[2], grows=h[3], gnnz=h[4], nod=h[5];
    CSR full, group;
    full.rows=frows; full.cols=nlinks; group.rows=grows; group.cols=nlinks;
    read_vec(f, full.row_ptr, frows + 1); read_vec(f, full.col_idx, fnnz); read_vec(f, full.val, fnnz);
    read_vec(f, group.row_ptr, grows + 1); read_vec(f, group.col_idx, gnnz); read_vec(f, group.val, gnnz);
    std::vector<int64_t> fptr, gptr;
    read_vec(f, fptr, nod + 1); read_vec(f, gptr, nod + 1);

    // Warm-up and median of five batches.
    benchmark(full, fptr, 100); benchmark(group, gptr, 100);
    std::vector<double> tf, tg;
    for (int i=0;i<5;++i) {
        tf.push_back(benchmark(full, fptr, repeats));
        tg.push_back(benchmark(group, gptr, repeats));
    }
    std::sort(tf.begin(), tf.end()); std::sort(tg.begin(), tg.end());
    const double fsec=tf[2], gsec=tg[2];
    std::cout << "full_rows,group_rows,n_links,full_us,group_us,speedup\n";
    std::cout << frows << ',' << grows << ',' << nlinks << ','
              << std::fixed << std::setprecision(6)
              << 1e6*fsec << ',' << 1e6*gsec << ',' << fsec/gsec << "\n";
    return 0;
}
