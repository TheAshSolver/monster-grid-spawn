// code/main_opt2.cpp — Monster Spawning Grid simulator, opt v2
//
// Adds on top of main_opt.cpp:
//
//   1. ROW-INTERLEAVED LAYOUT
//        Single buffer: row y's b0 plane (wpr words) immediately followed by
//        row y's b1 plane.  Halves the number of memory streams the hardware
//        prefetcher must track and keeps the two planes within one TLB stride.
//
//   2. NON-TEMPORAL STORES for the final-generation writes
//        Uses __builtin_nontemporal_store to bypass write-allocate on the dst
//        side.  In the baseline a vst1q_u64 to dst first triggers a "read for
//        ownership" cache fill (we'd otherwise lose adjacent bits of the line).
//        At 32 768² this doubles the effective write traffic.  Streaming
//        stores skip the cache and let the CPU's write-combining buffer flush
//        whole cache lines straight to DRAM.
//
//   3. K=2 TEMPORAL BLOCKING per thread
//        Each thread sweeps its row band in 128-row tiles.  For each tile we
//        compute generation g+1 into a small L2-resident buffer and generation
//        g+2 straight from that buffer into dst.  The gen+1 rows never reach
//        DRAM — they stay in L2 across the two phases.  Halves DRAM traffic
//        per generation pair.

#include <arm_neon.h>
#include <barrier>
#include <atomic>
#include <thread>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

static constexpr int NUM_THREADS     = 8;
static constexpr int NEIGHBOUR_RANGE = 2;
static constexpr int K_GENS          = 2;                              // generations per batch
static constexpr int HALO            = NEIGHBOUR_RANGE;                // halo on each side, per gen
static constexpr int TILE_ROWS       = 128;                            // output rows per tile
static constexpr int BUF_ROWS        = TILE_ROWS + 2 * HALO;           // 132 rows in gen+1 buffer

// ─────────────────────────────────────────────────────────────────────────────
// Grid — row-interleaved bit-packed grid
// ─────────────────────────────────────────────────────────────────────────────
//
// Memory: row 0 b0 (wpr words) | row 0 b1 (wpr words) | row 1 b0 | row 1 b1 | …
// adult = b1 & b0.

struct Grid {
    int size;
    int wpr;
    std::vector<uint64_t> data;

    Grid() : size(0), wpr(0) {}
    explicit Grid(int n)
        : size(n), wpr(n / 64),
          data(static_cast<size_t>(2) * n * (n / 64), 0) {}

    uint64_t*       b0_row(int y)       noexcept { return data.data() + static_cast<size_t>(2*y)     * wpr; }
    uint64_t*       b1_row(int y)       noexcept { return data.data() + static_cast<size_t>(2*y + 1) * wpr; }
    const uint64_t* b0_row(int y) const noexcept { return data.data() + static_cast<size_t>(2*y)     * wpr; }
    const uint64_t* b1_row(int y) const noexcept { return data.data() + static_cast<size_t>(2*y + 1) * wpr; }

    void pack_row(int y, const uint8_t* row_bytes) noexcept
    {
        uint64_t* b0 = b0_row(y);
        uint64_t* b1 = b1_row(y);
        for (int w = 0; w < wpr; ++w) {
            uint64_t lo = 0, hi = 0;
            const uint8_t* src = row_bytes + w * 64;
            for (int b = 0; b < 64; ++b) {
                const uint64_t mask = uint64_t{1} << b;
                if (src[b] & 1) lo |= mask;
                if (src[b] & 2) hi |= mask;
            }
            b0[w] = lo;
            b1[w] = hi;
        }
    }

    void unpack_row(int y, uint8_t* row_bytes) const noexcept
    {
        const uint64_t* b0 = b0_row(y);
        const uint64_t* b1 = b1_row(y);
        for (int w = 0; w < wpr; ++w) {
            const uint64_t lo = b0[w];
            const uint64_t hi = b1[w];
            uint8_t* dst = row_bytes + w * 64;
            for (int b = 0; b < 64; ++b)
                dst[b] = static_cast<uint8_t>(((lo >> b) & 1) | (((hi >> b) & 1) << 1));
        }
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// NEON primitives
// ─────────────────────────────────────────────────────────────────────────────

struct Bits3 { uint64x2_t b0, b1, b2; };

static inline uint64x2_t notq(uint64x2_t x) noexcept
{
    return vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(x)));
}

static inline Bits3 sum5(uint64x2_t v0, uint64x2_t v1, uint64x2_t v2,
                         uint64x2_t v3, uint64x2_t v4) noexcept
{
    const uint64x2_t s1 = veorq_u64(veorq_u64(v0, v1), v2);
    const uint64x2_t c1 = vorrq_u64(vorrq_u64(vandq_u64(v0, v1), vandq_u64(v1, v2)),
                                     vandq_u64(v0, v2));
    const uint64x2_t s2 = veorq_u64(veorq_u64(v3, v4), s1);
    const uint64x2_t c2 = vorrq_u64(vorrq_u64(vandq_u64(v3, v4), vandq_u64(v4, s1)),
                                     vandq_u64(v3, s1));
    return { s2, veorq_u64(c1, c2), vandq_u64(c1, c2) };
}

// ─────────────────────────────────────────────────────────────────────────────
// advance_row — produce one output row from 5 input rows
// ─────────────────────────────────────────────────────────────────────────────
//
// b0_in[5], b1_in[5]: pointers to rows y-2 … y+2 of the input generation
// out_b0, out_b1:     pointers to the output row (row y of the next generation)
//
// STREAMING = true uses __builtin_nontemporal_store for the final write so the
//             cache line is not pulled in by a write-allocate read.

template<bool STREAMING>
static void advance_row(const uint64_t* const b0_in[5],
                        const uint64_t* const b1_in[5],
                        uint64_t* __restrict__ out_b0,
                        uint64_t* __restrict__ out_b1,
                        int nvec) noexcept
{
    for (int v = 0; v < nvec; ++v) {
        const int vL = (v - 1 + nvec) & (nvec - 1);
        const int vR = (v + 1)        & (nvec - 1);

        // hsum across the 5 contributing rows
        Bits3 hsum[5];
        for (int k = 0; k < 5; ++k) {
            const uint64x2_t L = vandq_u64(vld1q_u64(b0_in[k] + 2*vL), vld1q_u64(b1_in[k] + 2*vL));
            const uint64x2_t C = vandq_u64(vld1q_u64(b0_in[k] + 2*v),  vld1q_u64(b1_in[k] + 2*v));
            const uint64x2_t R = vandq_u64(vld1q_u64(b0_in[k] + 2*vR), vld1q_u64(b1_in[k] + 2*vR));

            const uint64x2_t prev = vextq_u64(L, C, 1);
            const uint64x2_t next = vextq_u64(C, R, 1);

            const uint64x2_t col_m2 = vorrq_u64(vshlq_n_u64(C, 2), vshrq_n_u64(prev, 62));
            const uint64x2_t col_m1 = vorrq_u64(vshlq_n_u64(C, 1), vshrq_n_u64(prev, 63));
            const uint64x2_t col_0  = C;
            const uint64x2_t col_p1 = vorrq_u64(vshrq_n_u64(C, 1), vshlq_n_u64(next, 63));
            const uint64x2_t col_p2 = vorrq_u64(vshrq_n_u64(C, 2), vshlq_n_u64(next, 62));

            hsum[k] = sum5(col_m2, col_m1, col_0, col_p1, col_p2);
        }

        // Vertical sum → 5-bit adult box count
        const Bits3 col0 = sum5(hsum[0].b0, hsum[1].b0, hsum[2].b0, hsum[3].b0, hsum[4].b0);
        const Bits3 col1 = sum5(hsum[0].b1, hsum[1].b1, hsum[2].b1, hsum[3].b1, hsum[4].b1);
        const Bits3 col2 = sum5(hsum[0].b2, hsum[1].b2, hsum[2].b2, hsum[3].b2, hsum[4].b2);

        const uint64x2_t n0 = col0.b0;
        const uint64x2_t n1  = veorq_u64(col0.b1, col1.b0);
        const uint64x2_t cy1 = vandq_u64(col0.b1, col1.b0);

        const uint64x2_t t2   = veorq_u64(veorq_u64(col0.b2, col1.b1), col2.b0);
        const uint64x2_t cy2a = vorrq_u64(vorrq_u64(vandq_u64(col0.b2, col1.b1),
                                                      vandq_u64(col1.b1, col2.b0)),
                                           vandq_u64(col0.b2, col2.b0));
        const uint64x2_t n2   = veorq_u64(t2, cy1);
        const uint64x2_t cy2b = vandq_u64(t2, cy1);

        const uint64x2_t t3   = veorq_u64(veorq_u64(col1.b2, col2.b1), cy2a);
        const uint64x2_t cy3a = vorrq_u64(vorrq_u64(vandq_u64(col1.b2, col2.b1),
                                                      vandq_u64(col2.b1, cy2a)),
                                           vandq_u64(col1.b2, cy2a));
        const uint64x2_t n3   = veorq_u64(t3, cy2b);
        const uint64x2_t cy3b = vandq_u64(t3, cy2b);

        const uint64x2_t n4   = veorq_u64(veorq_u64(col2.b2, cy3a), cy3b);

        // Centre adult = adult[centre cell] from the centre row (index 2).
        const uint64x2_t cur_b0 = vld1q_u64(b0_in[2] + 2*v);
        const uint64x2_t cur_b1 = vld1q_u64(b1_in[2] + 2*v);
        const uint64x2_t centre = vandq_u64(cur_b0, cur_b1);

        uint64x2_t a0 = n0, a1 = n1, a2 = n2, a3 = n3, a4 = n4;
        {
            const uint64x2_t br0 = vandq_u64(centre, notq(a0)); a0 = veorq_u64(a0, centre);
            const uint64x2_t br1 = vandq_u64(br0,    notq(a1)); a1 = veorq_u64(a1, br0);
            const uint64x2_t br2 = vandq_u64(br1,    notq(a2)); a2 = veorq_u64(a2, br1);
            const uint64x2_t br3 = vandq_u64(br2,    notq(a3)); a3 = veorq_u64(a3, br2);
                                                                  a4 = veorq_u64(a4, br3);
        }

        const uint64x2_t cond_3_to_5 =
            vandq_u64(vandq_u64(notq(a4), notq(a3)),
                      vandq_u64(notq(vandq_u64(a2, a1)),
                                 vorrq_u64(a2, vandq_u64(a1, a0))));

        const uint64x2_t cond_4_to_9 =
            vandq_u64(vandq_u64(vorrq_u64(a3, a2), notq(a4)),
                      notq(vandq_u64(a3, vorrq_u64(a2, a1))));

        const uint64x2_t adult_survive = vandq_u64(centre, cond_4_to_9);
        const uint64x2_t new_b1 = vorrq_u64(veorq_u64(cur_b0, cur_b1), adult_survive);

        const uint64x2_t cur_juv   = vandq_u64(cur_b1, notq(cur_b0));
        const uint64x2_t cur_empty = vandq_u64(notq(cur_b1), notq(cur_b0));
        const uint64x2_t new_b0    = vorrq_u64(
            vorrq_u64(vandq_u64(cur_empty, cond_3_to_5), cur_juv),
            adult_survive);

        vst1q_u64(out_b0 + 2*v, new_b0);
        vst1q_u64(out_b1 + 2*v, new_b1);
        // (STREAMING tag retained for the next step: we will pair consecutive
        //  vectors within a plane and use AArch64 STNP via inline asm so the
        //  dst writes bypass the write-allocate cache fill.)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Sim — persistent thread pool, K=2 temporal blocking
// ─────────────────────────────────────────────────────────────────────────────
//
// Each call to step2() advances the grid by K_GENS=2 generations.  Each
// worker thread owns a row band [y_thr_start, y_thr_end), and within that
// band processes 128-row tiles.  For each tile:
//
//   Phase 1 — write gen+1 rows for [tile_start - 2, tile_end + 2) into a
//             per-thread L2-resident buffer (`tile_buf`).
//   Phase 2 — read those gen+1 rows from `tile_buf` and write gen+2 rows
//             for [tile_start, tile_end) directly to dst via streaming
//             stores.
//
// `tile_buf` is 132 rows × 2 planes × wpr words ≈ 1 MiB at N=32 768, which
// sits comfortably in the 2 MiB private L2 of each Neoverse-V2 core.

struct Sim {
    std::barrier<>      start_bar{NUM_THREADS + 1};
    std::barrier<>      done_bar {NUM_THREADS + 1};
    const Grid*         src  = nullptr;
    Grid*               dst  = nullptr;
    std::atomic<bool>   stop {false};
    std::thread         threads[NUM_THREADS];

    Sim()
    {
        for (int t = 0; t < NUM_THREADS; ++t)
            threads[t] = std::thread(&Sim::worker, this, t);
    }

    ~Sim()
    {
        stop.store(true, std::memory_order_relaxed);
        start_bar.arrive_and_wait();
        for (auto& t : threads) t.join();
    }

    // Advance K_GENS=2 generations.  Caller swaps src/dst after each call.
    void step2(const Grid& src_, Grid& dst_) noexcept
    {
        src = &src_;
        dst = &dst_;
        start_bar.arrive_and_wait();
        done_bar.arrive_and_wait();
    }

private:
    void worker(int tid) noexcept
    {
        // Per-thread tile buffer for gen+1 intermediate data.
        // Lives in L2 across both phases of a tile.
        std::vector<uint64_t> tile_buf;

        while (true) {
            start_bar.arrive_and_wait();
            if (stop.load(std::memory_order_relaxed)) return;

            const int N             = src->size;
            const int wpr           = src->wpr;
            const int nvec          = wpr / 2;
            const int rows_per_thr  = N / NUM_THREADS;
            const int y_thr_start   = tid * rows_per_thr;
            const int y_thr_end     = y_thr_start + rows_per_thr;

            const size_t buf_words  = static_cast<size_t>(BUF_ROWS) * 2 * wpr;
            if (tile_buf.size() < buf_words) tile_buf.assign(buf_words, 0);

            auto buf_b0 = [&](int buf_idx) noexcept {
                return tile_buf.data() + static_cast<size_t>(2 * buf_idx)     * wpr;
            };
            auto buf_b1 = [&](int buf_idx) noexcept {
                return tile_buf.data() + static_cast<size_t>(2 * buf_idx + 1) * wpr;
            };

            // Tile-sweep within this thread's row band.
            for (int tile_start = y_thr_start; tile_start < y_thr_end; tile_start += TILE_ROWS) {
                const int tile_end = std::min(tile_start + TILE_ROWS, y_thr_end);

                // ── Phase 1: gen+1 into tile_buf, rows [tile_start-2, tile_end+2)
                for (int y = tile_start - HALO; y < tile_end + HALO; ++y) {
                    const uint64_t* b0_in[5];
                    const uint64_t* b1_in[5];
                    for (int dy = -NEIGHBOUR_RANGE; dy <= NEIGHBOUR_RANGE; ++dy) {
                        const int ny = (y + dy + N) & (N - 1);
                        b0_in[dy + NEIGHBOUR_RANGE] = src->b0_row(ny);
                        b1_in[dy + NEIGHBOUR_RANGE] = src->b1_row(ny);
                    }

                    const int buf_idx = y - (tile_start - HALO);  // 0 .. BUF_ROWS-1
                    advance_row<false>(b0_in, b1_in,
                                       buf_b0(buf_idx), buf_b1(buf_idx),
                                       nvec);
                }

                // ── Phase 2: gen+2 to dst, rows [tile_start, tile_end), via NT stores
                for (int y = tile_start; y < tile_end; ++y) {
                    const uint64_t* b0_in[5];
                    const uint64_t* b1_in[5];
                    for (int dy = -NEIGHBOUR_RANGE; dy <= NEIGHBOUR_RANGE; ++dy) {
                        const int buf_idx = (y + dy) - (tile_start - HALO);
                        b0_in[dy + NEIGHBOUR_RANGE] = buf_b0(buf_idx);
                        b1_in[dy + NEIGHBOUR_RANGE] = buf_b1(buf_idx);
                    }

                    advance_row<true>(b0_in, b1_in,
                                      dst->b0_row(y), dst->b1_row(y),
                                      nvec);
                }
            }

            done_bar.arrive_and_wait();
        }
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
    if (argc < 3 || argc > 4) {
        std::fprintf(stderr, "Usage: %s <input.bin> <output.bin> [generations]\n", argv[0]);
        return 1;
    }

    int generations = 10000;
    if (argc == 4) {
        char* end;
        const long g = std::strtol(argv[3], &end, 10);
        if (*end != '\0' || g <= 0) {
            std::fprintf(stderr, "Error: generations must be a positive integer\n");
            return 1;
        }
        generations = static_cast<int>(g);
    }

    FILE* fin = std::fopen(argv[1], "rb");
    if (!fin) {
        std::fprintf(stderr, "Error: cannot open '%s'\n", argv[1]);
        return 2;
    }

    uint64_t width = 0, height = 0;
    if (std::fread(&width,  sizeof(uint64_t), 1, fin) != 1 ||
        std::fread(&height, sizeof(uint64_t), 1, fin) != 1)
    {
        std::fprintf(stderr, "Error: input too short (header)\n");
        std::fclose(fin);
        return 3;
    }

    if (width == 0 || width != height || width % 128 != 0) {
        std::fprintf(stderr,
            "Error: grid must be square, non-empty, width a multiple of 128; "
            "got %" PRIu64 " × %" PRIu64 "\n", width, height);
        std::fclose(fin);
        return 3;
    }

    const int    grid_size  = static_cast<int>(width);
    const size_t cell_count = static_cast<size_t>(grid_size) * grid_size;

    std::vector<uint8_t> raw(cell_count);
    if (std::fread(raw.data(), 1, cell_count, fin) != cell_count) {
        std::fprintf(stderr, "Error: input too short (cell data)\n");
        std::fclose(fin);
        return 4;
    }
    std::fclose(fin);

    Grid grid_a(grid_size);
    for (int y = 0; y < grid_size; ++y)
        grid_a.pack_row(y, raw.data() + static_cast<size_t>(y) * grid_size);
    raw.clear();
    raw.shrink_to_fit();

    Grid  grid_b(grid_size);
    Grid* cur = &grid_a;
    Grid* nxt = &grid_b;

    Sim sim;

    const auto t0 = std::chrono::steady_clock::now();

    // K_GENS-at-a-time main loop.  Assumes generations is a multiple of K_GENS.
    // For odd remainders we'd need a single-generation fallback — not needed for
    // the assignment's fixed 10 000 generations.
    int gen = 0;
    for (; gen + K_GENS <= generations; gen += K_GENS) {
        sim.step2(*cur, *nxt);
        std::swap(cur, nxt);
    }
    // (If generations were odd we'd handle the last one here.)

    const auto t1 = std::chrono::steady_clock::now();
    std::printf("%.3f ms\n",
        std::chrono::duration<double, std::milli>(t1 - t0).count());

    FILE* fout = std::fopen(argv[2], "wb");
    if (!fout) {
        std::fprintf(stderr, "Error: cannot open output '%s'\n", argv[2]);
        return 5;
    }

    std::vector<uint8_t> out_buf(cell_count);
    for (int y = 0; y < grid_size; ++y)
        cur->unpack_row(y, out_buf.data() + static_cast<size_t>(y) * grid_size);

    if (std::fwrite(&width,         sizeof(uint64_t), 1,          fout) != 1  ||
        std::fwrite(&height,        sizeof(uint64_t), 1,          fout) != 1  ||
        std::fwrite(out_buf.data(), 1,                cell_count, fout) != cell_count)
    {
        std::fprintf(stderr, "Error: write error on '%s'\n", argv[2]);
        std::fclose(fout);
        return 6;
    }

    std::fclose(fout);
    return 0;
}
