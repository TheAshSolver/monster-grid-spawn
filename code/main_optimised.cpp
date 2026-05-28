// code/main_optimised.cpp — Monster Spawning Grid simulator (all optimisations)
//
// Key improvements over main_with_barrier_with_simd.cpp:
//
//   1. Sliding-window hsum  (biggest win)
//      step_pair previously recomputed hsum for all 5 neighbour rows
//      from scratch for every (w, y) cell.  Adjacent y values share 4 of those
//      5 rows.  We now precompute hrow[0..TILE_ROWS+3] once per (tile, w) and
//      index directly — reducing hsum calls from 5*TILE_ROWS to TILE_ROWS+4
//      per tile column, a 3.3x reduction in adult-plane loads and CSA work.
//
//   2. No skip-empty in the hot path
//      adult_neighbourhood_empty() did 15 loads per word on every iteration;
//      on a dense grid it always returned false, so those loads were pure waste
//      (~doubling adult-plane bandwidth).  Skip-empty is removed entirely.
//
//   3. Cache tiling (TILE_ROWS = 8, w-outer y-inner)
//      Keeps the hrow precomputed slice resident in L1 while the y inner loop
//      runs, so no re-fetching of hrow data within a tile.
//
//   4. Software prefetch (PREFETCH_DIST = 4)
//      Prefetches the adult row needed PREFETCH_DIST iterations ahead during
//      the hrow precomputation loop, hiding DRAM latency.
//
//   5. NEON (2 lanes, 128-bit, 128 cells per call)
//      AlignedBits3 stores hrow data as plain uint64_t pairs so the sliding
//      window works with normal C arrays.
//
//   6. Persistent thread pool + std::barrier, no redundant clear.
//
// Build (on AArch64 with NEON):
//   g++-14 -std=c++23 -O3 -mcpu=neoverse-v2
//       code/main_optimised.cpp -o spawn_sim -lpthread

#include <algorithm>
#include <arm_neon.h>
#include <barrier>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Tuning knobs
// ─────────────────────────────────────────────────────────────────────────────

static constexpr int NEIGHBOUR_RANGE = 2;
static constexpr int NUM_THREADS     = 8;
static constexpr int TILE_ROWS       = 8;
static constexpr int PREFETCH_DIST   = 4;
static int           GENERATIONS     = 10000;

// ─────────────────────────────────────────────────────────────────────────────
// BitGrid
// ─────────────────────────────────────────────────────────────────────────────

struct BitGrid {
    int size;
    int words_per_row;
    std::vector<uint64_t> egg;
    std::vector<uint64_t> juv;
    std::vector<uint64_t> adult;

    BitGrid() : size(0), words_per_row(0) {}

    explicit BitGrid(int n) : size(n), words_per_row(n / 64)
    {
        const size_t total = static_cast<size_t>(n) * words_per_row;
        egg.assign(total, 0);
        juv.assign(total, 0);
        adult.assign(total, 0);
    }

    void pack_row(int y, const uint8_t* row_bytes) noexcept
    {
        const size_t base = static_cast<size_t>(y) * words_per_row;
        const uint8x8_t scales = {1, 2, 4, 8, 16, 32, 64, 128};
        for (int w = 0; w < words_per_row; ++w) {
            const uint8_t* s = row_bytes + w * 64;
            uint64_t e = 0, j = 0, a = 0;
            for (int c = 0; c < 8; ++c) {
                const uint8x8_t v = vld1_u8(s + c * 8);
                // bit0 → egg, bit1 → juv, both → adult
                const uint64x1_t b0 = vpaddl_u32(vpaddl_u16(vpaddl_u8(
                    vmul_u8(vand_u8(v, vdup_n_u8(1)), scales))));
                const uint64x1_t b1 = vpaddl_u32(vpaddl_u16(vpaddl_u8(
                    vmul_u8(vand_u8(vshr_n_u8(v, 1), vdup_n_u8(1)), scales))));
                const uint64_t lo = vget_lane_u64(b0, 0) << (c * 8);
                const uint64_t hi = vget_lane_u64(b1, 0) << (c * 8);
                e |= lo & ~hi;
                j |= hi & ~lo;
                a |= lo &  hi;
            }
            egg[base + w] = e; juv[base + w] = j; adult[base + w] = a;
        }
    }

    void unpack_row(int y, uint8_t* row_bytes) const noexcept
    {
        const size_t base = static_cast<size_t>(y) * words_per_row;
        const int8x8_t neg_shifts = {0, -1, -2, -3, -4, -5, -6, -7};
        for (int w = 0; w < words_per_row; ++w) {
            const uint64_t e = egg[base+w], j = juv[base+w], a = adult[base+w];
            uint8_t* d = row_bytes + w * 64;
            for (int c = 0; c < 8; ++c) {
                // reconstruct 2-bit encoding: egg=1, juv=2, adult=3
                const uint8_t e_byte = (e >> (c * 8)) & 0xFF;
                const uint8_t j_byte = (j >> (c * 8)) & 0xFF;
                const uint8_t a_byte = (a >> (c * 8)) & 0xFF;
                const uint8x8_t e_bits = vand_u8(vshl_u8(vdup_n_u8(e_byte), neg_shifts), vdup_n_u8(1));
                const uint8x8_t j_bits = vand_u8(vshl_u8(vdup_n_u8(j_byte), neg_shifts), vdup_n_u8(1));
                const uint8x8_t a_bits = vand_u8(vshl_u8(vdup_n_u8(a_byte), neg_shifts), vdup_n_u8(1));
                // state: egg→1, juv→2, adult→3
                vst1_u8(d + c * 8, vorr_u8(
                    vorr_u8(e_bits, vshl_n_u8(j_bits, 1)),
                    vadd_u8(a_bits, vshl_n_u8(a_bits, 1))));  // adult: 1 | 2 = 3
            }
        }
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Thread pool + global pointers
// ─────────────────────────────────────────────────────────────────────────────

static BitGrid* src = nullptr;
static BitGrid* dst = nullptr;

struct ThreadPool {
    std::vector<std::thread> workers;
    struct Completion { void operator()() noexcept { std::swap(src, dst); } };
    std::barrier<Completion> b{NUM_THREADS, Completion{}};
};

static void step_thread(int y_start, ThreadPool& pool) noexcept;

void initialize_threads(ThreadPool& pool)
{
    const int rows_per_thread = src->size / NUM_THREADS;
    for (int t = 0; t < NUM_THREADS; ++t)
        pool.workers.emplace_back(step_thread, t * rows_per_thread, std::ref(pool));
    for (auto& w : pool.workers)
        if (w.joinable()) w.join();
}

// ─────────────────────────────────────────────────────────────────────────────
// AlignedBits3 — 3-bit-plane hsum result for 2 words (128 cells)
// ─────────────────────────────────────────────────────────────────────────────

struct alignas(16) AlignedBits3 {
    uint64_t b0[2];
    uint64_t b1[2];
    uint64_t b2[2];
};

// ─────────────────────────────────────────────────────────────────────────────
// NEON helpers
// ─────────────────────────────────────────────────────────────────────────────

static inline uint64x2_t vnot(uint64x2_t a) noexcept
{ return veorq_u64(a, vdupq_n_u64(~uint64_t{0})); }

static void n_sum5(uint64x2_t v0, uint64x2_t v1, uint64x2_t v2,
                   uint64x2_t v3, uint64x2_t v4,
                   uint64x2_t& b0, uint64x2_t& b1, uint64x2_t& b2) noexcept
{
    const uint64x2_t s1 = veorq_u64(veorq_u64(v0,v1),v2);
    const uint64x2_t c1 = vorrq_u64(vorrq_u64(vandq_u64(v0,v1),vandq_u64(v1,v2)),vandq_u64(v0,v2));
    const uint64x2_t s2 = veorq_u64(veorq_u64(v3,v4),s1);
    const uint64x2_t c2 = vorrq_u64(vorrq_u64(vandq_u64(v3,v4),vandq_u64(v4,s1)),vandq_u64(v3,s1));
    b0 = s2; b1 = veorq_u64(c1,c2); b2 = vandq_u64(c1,c2);
}

// hsum_pair: horizontal adult-neighbour sum for words (w, w+1) of a row.
static void hsum_pair(const uint64_t* adult_row, int wpr, int w,
                       AlignedBits3& out) noexcept
{
    const uint64x2_t L_vec = vld1q_u64(&adult_row[(w - 2 + wpr) & (wpr - 1)]);
    const uint64x2_t C     = vld1q_u64(&adult_row[w]);
    const uint64x2_t R_vec = vld1q_u64(&adult_row[(w + 2) & (wpr - 1)]);
    const uint64x2_t L_ctx = vextq_u64(L_vec, C,     1);
    const uint64x2_t R_ctx = vextq_u64(C,     R_vec, 1);
    uint64x2_t b0, b1, b2;
    n_sum5(vorrq_u64(vshlq_n_u64(C,2), vshrq_n_u64(L_ctx,62)),
           vorrq_u64(vshlq_n_u64(C,1), vshrq_n_u64(L_ctx,63)),
           C,
           vorrq_u64(vshrq_n_u64(C,1), vshlq_n_u64(R_ctx,63)),
           vorrq_u64(vshrq_n_u64(C,2), vshlq_n_u64(R_ctx,62)),
           b0, b1, b2);
    vst1q_u64(out.b0, b0); vst1q_u64(out.b1, b1); vst1q_u64(out.b2, b2);
}

// vstep_pair: vertical sum + rules for words (w, w+1) given 5 precomputed hsums.
static void vstep_pair(const BitGrid& s, BitGrid& d, int w, int y,
                        const AlignedBits3& h0, const AlignedBits3& h1,
                        const AlignedBits3& h2, const AlignedBits3& h3,
                        const AlignedBits3& h4) noexcept
{
    const int wpr = s.words_per_row;
    const size_t wi = static_cast<size_t>(y) * wpr + w;

    const uint64x2_t h0b0=vld1q_u64(h0.b0), h0b1=vld1q_u64(h0.b1), h0b2=vld1q_u64(h0.b2);
    const uint64x2_t h1b0=vld1q_u64(h1.b0), h1b1=vld1q_u64(h1.b1), h1b2=vld1q_u64(h1.b2);
    const uint64x2_t h2b0=vld1q_u64(h2.b0), h2b1=vld1q_u64(h2.b1), h2b2=vld1q_u64(h2.b2);
    const uint64x2_t h3b0=vld1q_u64(h3.b0), h3b1=vld1q_u64(h3.b1), h3b2=vld1q_u64(h3.b2);
    const uint64x2_t h4b0=vld1q_u64(h4.b0), h4b1=vld1q_u64(h4.b1), h4b2=vld1q_u64(h4.b2);

    uint64x2_t col0b0, col0b1, col0b2;
    uint64x2_t col1b0, col1b1, col1b2;
    uint64x2_t col2b0, col2b1, col2b2;
    n_sum5(h0b0,h1b0,h2b0,h3b0,h4b0, col0b0,col0b1,col0b2);
    n_sum5(h0b1,h1b1,h2b1,h3b1,h4b1, col1b0,col1b1,col1b2);
    n_sum5(h0b2,h1b2,h2b2,h3b2,h4b2, col2b0,col2b1,col2b2);

    const uint64x2_t n0  = col0b0;
    const uint64x2_t n1  = veorq_u64(col0b1,col1b0), cy1  = vandq_u64(col0b1,col1b0);
    const uint64x2_t t2  = veorq_u64(veorq_u64(col0b2,col1b1),col2b0);
    const uint64x2_t cy2a= vorrq_u64(vorrq_u64(vandq_u64(col0b2,col1b1),vandq_u64(col1b1,col2b0)),vandq_u64(col0b2,col2b0));
    const uint64x2_t n2  = veorq_u64(t2,cy1),        cy2b = vandq_u64(t2,cy1);
    const uint64x2_t t3  = veorq_u64(veorq_u64(col1b2,col2b1),cy2a);
    const uint64x2_t cy3a= vorrq_u64(vorrq_u64(vandq_u64(col1b2,col2b1),vandq_u64(col2b1,cy2a)),vandq_u64(col1b2,cy2a));
    const uint64x2_t n3  = veorq_u64(t3,cy2b),       cy3b = vandq_u64(t3,cy2b);
    const uint64x2_t n4  = veorq_u64(veorq_u64(col2b2,cy3a),cy3b);

    const uint64x2_t cur_adlt = vld1q_u64(&s.adult[wi]);
    uint64x2_t a0=n0, a1=n1, a2=n2, a3=n3, a4=n4;
    { const uint64x2_t br0=vandq_u64(cur_adlt,vnot(a0)); a0=veorq_u64(a0,cur_adlt);
      const uint64x2_t br1=vandq_u64(br0,vnot(a1));      a1=veorq_u64(a1,br0);
      const uint64x2_t br2=vandq_u64(br1,vnot(a2));      a2=veorq_u64(a2,br1);
      const uint64x2_t br3=vandq_u64(br2,vnot(a3));      a3=veorq_u64(a3,br2); a4=veorq_u64(a4,br3); }

    const uint64x2_t cond35 =
        vandq_u64(vandq_u64(vandq_u64(vnot(a4),vnot(a3)),vnot(vandq_u64(a2,a1))),
                  vorrq_u64(a2,vandq_u64(a1,a0)));
    const uint64x2_t cond49 =
        vandq_u64(vandq_u64(vorrq_u64(a3,a2),vnot(a4)),
                  vnot(vandq_u64(a3,vorrq_u64(a2,a1))));

    const uint64x2_t ce = vld1q_u64(&s.egg[wi]);
    const uint64x2_t cj = vld1q_u64(&s.juv[wi]);
    const uint64x2_t ca = cur_adlt;
    vst1q_u64(&d.egg  [wi], vandq_u64(vnot(vorrq_u64(vorrq_u64(ce,cj),ca)), cond35));
    vst1q_u64(&d.juv  [wi], ce);
    vst1q_u64(&d.adult[wi], vorrq_u64(cj, vandq_u64(ca,cond49)));
}

// ─────────────────────────────────────────────────────────────────────────────
// step_thread — sliding-window cache-tiled worker
// ─────────────────────────────────────────────────────────────────────────────

static void step_thread(int y_start, ThreadPool& pool) noexcept
{
    const int N               = src->size;
    const int wpr             = src->words_per_row;
    const int rows_per_thread = N / NUM_THREADS;
    const int y_end           = y_start + rows_per_thread;

    AlignedBits3 hrow[TILE_ROWS + 4];

    for (int i = 0; i < GENERATIONS; ++i) {
        const BitGrid& S = *src;
        BitGrid&       D = *dst;

        const uint64_t* adult = S.adult.data();

        for (int y_tile = y_start; y_tile < y_end; y_tile += TILE_ROWS) {
            const int tile_end  = std::min(y_tile + TILE_ROWS, y_end);
            const int tile_rows = tile_end - y_tile;
            const int n_hrows   = tile_rows + 4;

            for (int w = 0; w < wpr; w += 2) {
                for (int r = 0; r < n_hrows; ++r) {
                    const int ry  = (y_tile - NEIGHBOUR_RANGE + r + N) & (N - 1);
                    const int pry = (ry + PREFETCH_DIST + N) & (N - 1);
                    __builtin_prefetch(adult + static_cast<size_t>(pry) * wpr + w, 0, 1);
                    hsum_pair(adult + static_cast<size_t>(ry) * wpr, wpr, w, hrow[r]);
                }

                for (int y = y_tile; y < tile_end; ++y) {
                    const int b = y - y_tile;
                    vstep_pair(S, D, w, y,
                               hrow[b], hrow[b+1], hrow[b+2], hrow[b+3], hrow[b+4]);
                }
            }
        }

        pool.b.arrive_and_wait();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
    if (argc < 3 || argc > 4) {
        std::fprintf(stderr, "Usage: %s <input.bin> <output.bin> [generations]\n", argv[0]);
        return 1;
    }

    if (argc == 4) {
        char* end;
        const long g = std::strtol(argv[3], &end, 10);
        if (*end != '\0' || g <= 0) {
            std::fprintf(stderr, "Error: generations must be a positive integer\n");
            return 1;
        }
        GENERATIONS = static_cast<int>(g);
    }

    FILE* fin = std::fopen(argv[1], "rb");
    if (!fin) { std::fprintf(stderr, "Error: cannot open '%s'\n", argv[1]); return 2; }

    uint64_t width = 0, height = 0;
    if (std::fread(&width, sizeof(uint64_t), 1, fin) != 1 ||
        std::fread(&height,sizeof(uint64_t), 1, fin) != 1) {
        std::fprintf(stderr, "Error: cannot read header\n"); std::fclose(fin); return 3;
    }
    if (width == 0 || width != height || width % 128 != 0) {
        std::fprintf(stderr,
            "Error: grid must be square, non-empty, multiple of 128; got %"
            PRIu64 "x%" PRIu64 "\n", width, height);
        std::fclose(fin); return 3;
    }

    const int    grid_size  = static_cast<int>(width);
    const size_t cell_count = static_cast<size_t>(grid_size) * grid_size;

    std::vector<uint8_t> raw(cell_count);
    if (std::fread(raw.data(), 1, cell_count, fin) != cell_count) {
        std::fprintf(stderr, "Error: cell data truncated\n"); std::fclose(fin); return 4;
    }
    std::fclose(fin);

    BitGrid grid_a(grid_size);
    for (int y = 0; y < grid_size; ++y)
        grid_a.pack_row(y, raw.data() + static_cast<size_t>(y) * grid_size);
    raw.clear(); raw.shrink_to_fit();

    BitGrid grid_b(grid_size);
    src = &grid_a; dst = &grid_b;

    const auto t0 = std::chrono::steady_clock::now();
    ThreadPool p;
    initialize_threads(p);
    const double ms = std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    std::printf("%.3f ms\n", ms);

    FILE* fout = std::fopen(argv[2], "wb");
    if (!fout) { std::fprintf(stderr, "Error: cannot open '%s'\n", argv[2]); return 5; }

    std::vector<uint8_t> out_buf(cell_count);
    for (int y = 0; y < grid_size; ++y)
        src->unpack_row(y, out_buf.data() + static_cast<size_t>(y) * grid_size);

    if (std::fwrite(&width,         sizeof(uint64_t), 1,          fout) != 1  ||
        std::fwrite(&height,        sizeof(uint64_t), 1,          fout) != 1  ||
        std::fwrite(out_buf.data(), 1,                cell_count, fout) != cell_count) {
        std::fprintf(stderr, "Error: write error\n"); std::fclose(fout); return 6;
    }
    std::fclose(fout);
    return 0;
}
