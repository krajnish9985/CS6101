#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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

constexpr int64_t SECTOR = 4096;

struct Cand {
    float d;
    int32_t id;
    int32_t expanded;
};

struct Handle {
    int fd = -1;
    int64_t N = 0;
    int D = 0, R = 0, nps = 0, node_bytes = 0;
    int32_t medoid = 0;
    const uint8_t* codes = nullptr;
    int M = 0, dsub = 0;
    const float* centroids = nullptr;
    std::vector<uint32_t> tag;
    uint32_t cur = 0;
    std::vector<float> table;
    std::vector<Cand> cand;
    std::vector<std::pair<float, int32_t>> exact;
    uint8_t* buf = nullptr;
    int buf_w = 0;
};

}

extern "C" {

void* dk_open(const char* path, int64_t N, int D, int R, int nps, int node_bytes, int32_t medoid,
              const uint8_t* codes, int M, const float* centroids) {
    const int fd = open(path, O_RDONLY);
    if (fd < 0) return nullptr;
    posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM);
    Handle* h = new Handle();
    h->fd = fd;
    h->N = N;
    h->D = D;
    h->R = R;
    h->nps = nps;
    h->node_bytes = node_bytes;
    h->medoid = medoid;
    h->codes = codes;
    h->M = M;
    h->dsub = D / M;
    h->centroids = centroids;
    h->tag.assign((size_t)N, 0u);
    h->table.resize((size_t)M * 256);
    return h;
}

void dk_close(void* hp) {
    Handle* h = (Handle*)hp;
    if (!h) return;
    if (h->fd >= 0) close(h->fd);
    free(h->buf);
    delete h;
}

int dk_search(void* hp, const float* q, int k, int L, int W, int32_t* out_ids, float* out_d, int64_t* st) {
    Handle* h = (Handle*)hp;
    const int D = h->D, M = h->M, dsub = h->dsub;
    if (L < k) L = k;
    if (W < 1) W = 1;
    if (h->buf_w < W) {
        free(h->buf);
        if (posix_memalign((void**)&h->buf, SECTOR, (size_t)W * SECTOR) != 0) return -1;
        h->buf_w = W;
    }
    if (++h->cur == 0) {
        std::fill(h->tag.begin(), h->tag.end(), 0u);
        h->cur = 1;
    }
    for (int i = 0; i < 7; ++i) st[i] = 0;

    for (int m = 0; m < M; ++m) {
        const float* qs = q + m * dsub;
        const float* cm = h->centroids + (size_t)m * 256 * dsub;
        for (int c = 0; c < 256; ++c) h->table[(size_t)m * 256 + c] = l2sq(qs, cm + (size_t)c * dsub, dsub);
    }
    auto pqdist = [&](int32_t id) {
        const uint8_t* cd = h->codes + (size_t)id * M;
        const float* t = h->table.data();
        float s = 0.f;
        for (int m = 0; m < M; ++m) s += t[(size_t)m * 256 + cd[m]];
        return s;
    };

    if ((int)h->cand.size() < L) h->cand.resize((size_t)L);
    Cand* C = h->cand.data();
    int size = 0;
    auto insert = [&](float d, int32_t id) {
        if (size == L && d >= C[L - 1].d) return;
        int i = (size == L) ? L - 1 : size;
        while (i > 0 && C[i - 1].d > d) {
            C[i] = C[i - 1];
            --i;
        }
        C[i] = {d, id, 0};
        if (size < L) ++size;
    };
    h->exact.clear();
    h->tag[h->medoid] = h->cur;
    insert(pqdist(h->medoid), h->medoid);
    st[4]++;

    std::vector<int32_t> beam;
    std::vector<int64_t> sect;
    beam.reserve(W);
    sect.reserve(W);
    while (true) {
        beam.clear();
        for (int i = 0; i < size && (int)beam.size() < W; ++i)
            if (!C[i].expanded) {
                C[i].expanded = 1;
                beam.push_back(C[i].id);
            }
        if (beam.empty()) break;
        st[2]++;
        sect.clear();
        for (int32_t id : beam) {
            const int64_t s = 1 + (int64_t)id / h->nps;
            sect.push_back(s);
            posix_fadvise(h->fd, s * SECTOR, SECTOR, POSIX_FADV_WILLNEED);
        }
        for (size_t b = 0; b < beam.size(); ++b) {
            size_t src = b;
            for (size_t e = 0; e < b; ++e)
                if (sect[e] == sect[b]) {
                    src = e;
                    break;
                }
            if (src == b) {
                ssize_t got = pread(h->fd, h->buf + b * SECTOR, SECTOR, sect[b] * SECTOR);
                if (got != SECTOR) return -2;
                st[0]++;
                st[1] += SECTOR;
            } else {
                std::memcpy(h->buf + b * SECTOR, h->buf + src * SECTOR, SECTOR);
                st[6]++;
            }
        }
        for (size_t b = 0; b < beam.size(); ++b) {
            const int32_t id = beam[b];
            const uint8_t* rec = h->buf + b * SECTOR + (size_t)(id % h->nps) * h->node_bytes;
            const float* vec = (const float*)rec;
            h->exact.emplace_back(l2sq(q, vec, D), id);
            st[5]++;
            st[3]++;
            uint32_t deg;
            std::memcpy(&deg, rec + 4 * D, 4);
            const int32_t* nb = (const int32_t*)(rec + 4 * D + 4);
            for (uint32_t t = 0; t < deg && (int)t < h->R; ++t) {
                const int32_t j = nb[t];
                if (j < 0 || j >= h->N || h->tag[j] == h->cur) continue;
                h->tag[j] = h->cur;
                insert(pqdist(j), j);
                st[4]++;
            }
        }
    }
    auto& ex = h->exact;
    const int kk = std::min<int>(k, (int)ex.size());
    std::partial_sort(ex.begin(), ex.begin() + kk, ex.end());
    for (int t = 0; t < k; ++t) {
        out_ids[t] = t < kk ? ex[t].second : -1;
        if (out_d) out_d[t] = t < kk ? ex[t].first : 3.4e38f;
    }
    return 0;
}

}
