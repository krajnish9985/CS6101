#include <omp.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

namespace {

inline float l2sq(const float* __restrict a, const float* __restrict b, int D) {
    float s = 0.f;
#pragma omp simd reduction(+ : s)
    for (int i = 0; i < D; ++i) {
        const float t = a[i] - b[i];
        s += t * t;
    }
    return s;
}

inline uint64_t splitmix64(uint64_t& x) {
    uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

struct Candidate {
    float d;
    int32_t id;
    int32_t expanded;
};

struct Scratch {
    std::vector<uint32_t> tag;
    uint32_t cur = 0;
    std::vector<Candidate> cand;
    std::vector<int32_t> vis_id;
    std::vector<float> vis_d;
    std::vector<std::pair<float, int32_t>> pool;
    std::vector<char> alive;

    void new_search(int64_t N) {
        if ((int64_t)tag.size() != N) {
            tag.assign((size_t)N, 0u);
            cur = 0;
        }
        if (++cur == 0) {
            std::fill(tag.begin(), tag.end(), 0u);
            cur = 1;
        }
    }
};

thread_local Scratch TLS;

int greedy_search(const float* X, int64_t N, int D, const int32_t* nbrs, const int32_t* deg, int R,
                  int32_t start, const float* q, int L, Scratch& s, bool record_visited,
                  int64_t& n_exp, int64_t& n_dist) {
    s.new_search(N);
    if ((int)s.cand.size() < L) s.cand.resize((size_t)L);
    Candidate* C = s.cand.data();
    if (record_visited) {
        s.vis_id.clear();
        s.vis_d.clear();
    }

    s.tag[start] = s.cur;
    C[0] = {l2sq(q, X + (int64_t)start * D, D), start, 0};
    int size = 1;
    ++n_dist;

    int k = 0;
    while (k < size) {
        const int32_t p = C[k].id;
        C[k].expanded = 1;
        ++n_exp;
        if (record_visited) {
            s.vis_id.push_back(p);
            s.vis_d.push_back(C[k].d);
        }

        const int32_t* nb = nbrs + (int64_t)p * R;
        const int dp = deg[p];
        for (int t = 0; t < dp; ++t) __builtin_prefetch(X + (int64_t)nb[t] * D);

        int nk = INT_MAX;
        for (int t = 0; t < dp; ++t) {
            const int32_t j = nb[t];
            if (s.tag[j] == s.cur) continue;
            s.tag[j] = s.cur;
            const float d = l2sq(q, X + (int64_t)j * D, D);
            ++n_dist;
            if (size == L && d >= C[L - 1].d) continue;
            int i = (size == L) ? L - 1 : size;
            while (i > 0 && C[i - 1].d > d) {
                C[i] = C[i - 1];
                --i;
            }
            C[i] = {d, j, 0};
            if (size < L) ++size;
            if (i < nk) nk = i;
        }
        if (nk <= k)
            k = nk;
        else
            ++k;
        while (k < size && C[k].expanded) ++k;
    }
    return size;
}

int robust_prune(const float* X, int D, int32_t p, std::vector<std::pair<float, int32_t>>& pool,
                 float alpha, int R, int32_t* out, std::vector<char>& alive) {
    std::sort(pool.begin(), pool.end(), [](const std::pair<float, int32_t>& a, const std::pair<float, int32_t>& b) {
        return a.second < b.second || (a.second == b.second && a.first < b.first);
    });
    size_t m = 0;
    for (size_t i = 0; i < pool.size(); ++i) {
        if (pool[i].second == p) continue;
        if (m > 0 && pool[m - 1].second == pool[i].second) continue;
        pool[m++] = pool[i];
    }
    pool.resize(m);
    std::sort(pool.begin(), pool.end());
    alive.assign(m, 1);

    const float a2 = alpha * alpha;
    int cnt = 0;
    for (size_t i = 0; i < m; ++i) {
        if (!alive[i]) continue;
        const int32_t ps = pool[i].second;
        out[cnt++] = ps;
        if (cnt == R) break;
        const float* xs = X + (int64_t)ps * D;
        for (size_t j = i + 1; j < m; ++j) {
            if (!alive[j]) continue;
            if (a2 * l2sq(xs, X + (int64_t)pool[j].second * D, D) <= pool[j].first) alive[j] = 0;
        }
    }
    return cnt;
}

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

}

extern "C" {

int32_t vamana_medoid(const float* X, int64_t N, int D, int nthreads) {
    omp_set_num_threads(nthreads);
    std::vector<double> mean((size_t)D, 0.0);
#pragma omp parallel
    {
        std::vector<double> loc((size_t)D, 0.0);
#pragma omp for schedule(static)
        for (int64_t i = 0; i < N; ++i)
            for (int d = 0; d < D; ++d) loc[d] += X[i * D + d];
#pragma omp critical
        for (int d = 0; d < D; ++d) mean[d] += loc[d];
    }
    std::vector<float> c((size_t)D);
    for (int d = 0; d < D; ++d) c[d] = (float)(mean[d] / (double)N);

    float best = 3.4e38f;
    int32_t best_i = 0;
#pragma omp parallel
    {
        float lb = 3.4e38f;
        int32_t li = 0;
#pragma omp for schedule(static)
        for (int64_t i = 0; i < N; ++i) {
            const float d = l2sq(c.data(), X + i * D, D);
            if (d < lb) {
                lb = d;
                li = (int32_t)i;
            }
        }
#pragma omp critical
        if (lb < best || (lb == best && li < best_i)) {
            best = lb;
            best_i = li;
        }
    }
    return best_i;
}

int vamana_build(const float* X, int64_t N, int D, int R, int L, float alpha, uint64_t seed,
                 int nthreads, double max_batch_frac, int32_t* nbrs, int32_t* deg, double* stats_out,
                 int verbose) {
    if (N < 2 || R < 1 || L < 1) return -1;
    omp_set_num_threads(nthreads);
    const double t0 = now_s();

    const int Rinit = (int)std::min<int64_t>(R, N - 1);
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < N; ++i) {
        uint64_t st = seed * 0x100000001B3ULL + (uint64_t)i * 0x9E3779B97F4A7C15ULL + 12345;
        int32_t* row = nbrs + i * R;
        int c = 0;
        while (c < Rinit) {
            const int32_t j = (int32_t)(splitmix64(st) % (uint64_t)N);
            if (j == (int32_t)i) continue;
            bool dup = false;
            for (int t = 0; t < c; ++t)
                if (row[t] == j) {
                    dup = true;
                    break;
                }
            if (!dup) row[c++] = j;
        }
        for (int t = c; t < R; ++t) row[t] = -1;
        deg[i] = c;
    }
    const int32_t s = vamana_medoid(X, N, D, nthreads);
    const double t_init = now_s() - t0;
    if (verbose) {
        std::printf("[vamana] N=%lld D=%d R=%d L=%d alpha=%.3f threads=%d  init+medoid %.2fs  medoid=%d\n",
                    (long long)N, D, R, L, alpha, nthreads, t_init, s);
        std::fflush(stdout);
    }

    const int64_t maxB = std::max<int64_t>(1, std::max<int64_t>(nthreads, (int64_t)(max_batch_frac * (double)N)));
    std::vector<int32_t> newN((size_t)(maxB * R));
    std::vector<int32_t> newDeg((size_t)maxB);
    std::vector<std::pair<int32_t, int32_t>> pairs;
    std::vector<int64_t> gstart;

    double pass_t[2] = {0, 0};
    double nbat[2] = {0, 0};
    double exp_tot = 0, dist_tot = 0;
    const float alphas[2] = {1.0f, alpha};

    for (int pass = 0; pass < 2; ++pass) {
        const double tp = now_s();
        const float a = alphas[pass];
        std::vector<int32_t> perm((size_t)N);
        std::iota(perm.begin(), perm.end(), 0);
        std::mt19937_64 rng(seed * 7919ULL + 17ULL + (uint64_t)pass);
        std::shuffle(perm.begin(), perm.end(), rng);

        int64_t pos = 0;
        int64_t B = (pass == 0) ? std::min<int64_t>(maxB, std::max(1, nthreads)) : maxB;
        int64_t next_report = N / 10;
        double pass_exp = 0, pass_dist = 0;

        while (pos < N) {
            const int64_t nb = std::min<int64_t>(B, N - pos);

#pragma omp parallel for schedule(dynamic, 4) reduction(+ : pass_exp, pass_dist)
            for (int64_t b = 0; b < nb; ++b) {
                const int32_t p = perm[pos + b];
                Scratch& sc = TLS;
                int64_t ne = 0, nd = 0;
                const float* xp = X + (int64_t)p * D;
                greedy_search(X, N, D, nbrs, deg, R, s, xp, L, sc, true, ne, nd);
                pass_exp += (double)ne;
                pass_dist += (double)nd;
                auto& pool = sc.pool;
                pool.clear();
                for (size_t t = 0; t < sc.vis_id.size(); ++t) pool.emplace_back(sc.vis_d[t], sc.vis_id[t]);
                const int32_t* row = nbrs + (int64_t)p * R;
                for (int t = 0; t < deg[p]; ++t) pool.emplace_back(l2sq(xp, X + (int64_t)row[t] * D, D), row[t]);
                newDeg[b] = robust_prune(X, D, p, pool, a, R, newN.data() + b * R, sc.alive);
            }
#pragma omp parallel for schedule(static)
            for (int64_t b = 0; b < nb; ++b) {
                const int32_t p = perm[pos + b];
                int32_t* row = nbrs + (int64_t)p * R;
                std::memcpy(row, newN.data() + b * R, sizeof(int32_t) * (size_t)newDeg[b]);
                for (int t = newDeg[b]; t < R; ++t) row[t] = -1;
                deg[p] = newDeg[b];
            }

            pairs.clear();
            for (int64_t b = 0; b < nb; ++b) {
                const int32_t p = perm[pos + b];
                for (int t = 0; t < newDeg[b]; ++t) pairs.emplace_back(newN[b * R + t], p);
            }
            std::sort(pairs.begin(), pairs.end());
            gstart.clear();
            for (size_t i = 0; i < pairs.size(); ++i)
                if (i == 0 || pairs[i].first != pairs[i - 1].first) gstart.push_back((int64_t)i);
            gstart.push_back((int64_t)pairs.size());
            const int64_t ng = (int64_t)gstart.size() - 1;

#pragma omp parallel for schedule(dynamic, 32)
            for (int64_t g = 0; g < ng; ++g) {
                const int32_t j = pairs[gstart[g]].first;
                int32_t* row = nbrs + (int64_t)j * R;
                const int dj = deg[j];
                Scratch& sc = TLS;
                auto& ids = sc.vis_id;
                ids.assign(row, row + dj);
                for (int64_t t = gstart[g]; t < gstart[g + 1]; ++t) {
                    const int32_t p = pairs[t].second;
                    if (p == j) continue;
                    bool present = false;
                    for (int u = 0; u < dj; ++u)
                        if (row[u] == p) {
                            present = true;
                            break;
                        }
                    if (!present) ids.push_back(p);
                }
                if ((int)ids.size() <= R) {
                    for (size_t u = 0; u < ids.size(); ++u) row[u] = ids[u];
                    deg[j] = (int32_t)ids.size();
                } else {
                    auto& pool = sc.pool;
                    pool.clear();
                    const float* xj = X + (int64_t)j * D;
                    for (int32_t c : ids) pool.emplace_back(l2sq(xj, X + (int64_t)c * D, D), c);
                    const int cnt = robust_prune(X, D, j, pool, a, R, row, sc.alive);
                    for (int t = cnt; t < R; ++t) row[t] = -1;
                    deg[j] = cnt;
                }
            }

            pos += nb;
            nbat[pass] += 1;
            B = std::min<int64_t>(B * 2, maxB);
            if (verbose && pos >= next_report) {
                std::printf("[vamana] pass %d (alpha=%.2f): %lld/%lld  %.1fs\n", pass + 1, a, (long long)pos,
                            (long long)N, now_s() - tp);
                std::fflush(stdout);
                next_report += N / 10;
            }
        }
        pass_t[pass] = now_s() - tp;
        if (pass == 1) {
            exp_tot = pass_exp / (double)N;
            dist_tot = pass_dist / (double)N;
        }
        if (verbose) {
            std::printf("[vamana] pass %d done in %.2fs (%d batches)\n", pass + 1, pass_t[pass], (int)nbat[pass]);
            std::fflush(stdout);
        }
    }
    if (stats_out) {
        stats_out[0] = t_init;
        stats_out[1] = pass_t[0];
        stats_out[2] = pass_t[1];
        stats_out[3] = nbat[0];
        stats_out[4] = nbat[1];
        stats_out[5] = exp_tot;
        stats_out[6] = dist_tot;
        stats_out[7] = (double)s;
    }
    return 0;
}

void vamana_search(const float* X, int64_t N, int D, const int32_t* nbrs, const int32_t* deg, int R,
                   int32_t start, const float* Q, int64_t nq, int k, int L, int nthreads, int32_t* out_ids,
                   float* out_d, int32_t* n_exp, int32_t* n_dist) {
    if (L < k) L = k;
    auto one = [&](int64_t i) {
        Scratch& sc = TLS;
        int64_t ne = 0, nd = 0;
        const int size = greedy_search(X, N, D, nbrs, deg, R, start, Q + i * D, L, sc, false, ne, nd);
        for (int t = 0; t < k; ++t) {
            out_ids[i * k + t] = (t < size) ? sc.cand[t].id : -1;
            if (out_d) out_d[i * k + t] = (t < size) ? sc.cand[t].d : 3.4e38f;
        }
        if (n_exp) n_exp[i] = (int32_t)ne;
        if (n_dist) n_dist[i] = (int32_t)nd;
    };
    if (nthreads <= 1) {
        for (int64_t i = 0; i < nq; ++i) one(i);
    } else {
        omp_set_num_threads(nthreads);
#pragma omp parallel for schedule(dynamic, 16)
        for (int64_t i = 0; i < nq; ++i) one(i);
    }
}

}
