// code/main_opt4.cpp — Monster Spawning Grid simulator, opt v4
//
// Builds on main_opt2.cpp (row-interleaved layout + K=2 temporal blocking).
//
// New in v4: HSUM ROW CACHING.
//
//   compute_vec previously called hsum_vec 5 times per output cell — once for
//   each of the 5 input rows in the neighbour window.  But moving from output
//   row y to y+1 only changes ONE of those input rows (drop y-2, add y+3);
//   the other 4 rows' horizontal sums are identical to the previous iteration
//   yet were being recomputed from scratch.
//
//   v4 maintains a 5-row sliding window of horizontal-sum results per thread.
//   For each output row we compute hsum for exactly one new row and reuse the
//   other four from cache.  Per-row hsum work drops 5× (from 5 calls to 1).
//
//   Combined with the existing temporal blocking, this is run at both phases
//   (gen+0 → gen+1 reads src; gen+1 → gen+2 reads tile_buf).  Each phase has
//   its own sliding window.

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
static constexpr int K_GENS          = 2;
static constexpr int HALO            = NEIGHBOUR_RANGE;
static constexpr int TILE_ROWS       = 128;
static constexpr int BUF_ROWS        = TILE_ROWS + 2 * HALO;
static constexpr int WINDOW          = 2 * NEIGHBOUR_RANGE + 1;   // 5

// ─────────────────────────────────────────────────────────────────────────────
// Grid (row-interleaved, two-plane)  — identical to main_opt2.cpp
// ─────────────────────────────────────────────────────────────────────────────

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

    void pack_row(int y, const uint8_t* row_bytes) noexcept {
        uint64_t* b0 = b0_row(y);
        uint64_t* b1 = b1_row(y);
        for (int w = 0; w < wpr; ++w) {
            uint64_t lo = 0, hi = 0;
            const uint8_t* src = row_bytes + w * 64;
            for (int b = 0; b < 64; ++b) {
                const uint64_t m = uint64_t{1} << b;
                if (src[b] & 1) lo |= m;
                if (src[b] & 2) hi |= m;
            }
            b0[w] = lo; b1[w] = hi;
        }
    }
    void unpack_row(int y, uint8_t* row_bytes) const noexcept {
        const uint64_t* b0 = b0_row(y);
        const uint64_t* b1 = b1_row(y);
        for (int w = 0; w < wpr; ++w) {
            const uint64_t lo = b0[w], hi = b1[w];
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
struct CellPair { uint64x2_t b0, b1; };

static inline uint64x2_t notq(uint64x2_t x) noexcept {
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
// hsum_row — compute horizontal adult sum for an ENTIRE row
// ─────────────────────────────────────────────────────────────────────────────
//
// Writes nvec Bits3 entries into out_hsum[0..nvec-1].  Same per-vector logic
// as the inner part of compute_vec in main_opt2, but factored so a single row
// can be hsummed once and reused 5× via the sliding window.

static void hsum_row(const uint64_t* __restrict__ b0_row,
                     const uint64_t* __restrict__ b1_row,
                     int nvec,
                     Bits3* __restrict__ out_hsum) noexcept
{
    for (int v = 0; v < nvec; ++v) {
        const int vL = (v - 1 + nvec) & (nvec - 1);
        const int vR = (v + 1)        & (nvec - 1);

        const uint64x2_t L = vandq_u64(vld1q_u64(b0_row + 2*vL), vld1q_u64(b1_row + 2*vL));
        const uint64x2_t C = vandq_u64(vld1q_u64(b0_row + 2*v),  vld1q_u64(b1_row + 2*v));
        const uint64x2_t R = vandq_u64(vld1q_u64(b0_row + 2*vR), vld1q_u64(b1_row + 2*vR));

        const uint64x2_t prev = vextq_u64(L, C, 1);
        const uint64x2_t next = vextq_u64(C, R, 1);

        const uint64x2_t col_m2 = vorrq_u64(vshlq_n_u64(C, 2), vshrq_n_u64(prev, 62));
        const uint64x2_t col_m1 = vorrq_u64(vshlq_n_u64(C, 1), vshrq_n_u64(prev, 63));
        const uint64x2_t col_0  = C;
        const uint64x2_t col_p1 = vorrq_u64(vshrq_n_u64(C, 1), vshlq_n_u64(next, 63));
        const uint64x2_t col_p2 = vorrq_u64(vshrq_n_u64(C, 2), vshlq_n_u64(next, 62));

        out_hsum[v] = sum5(col_m2, col_m1, col_0, col_p1, col_p2);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// vsum_and_transition — vertical CSA + state transition, one output vector
// ─────────────────────────────────────────────────────────────────────────────
//
// hs[5]:        pointers to 5 hsum rows (logically rows y-2 … y+2)
// cur_b0/b1:    centre row state plane bytes (for the b1 & b0 → adult subtract
//               and the actual transition formulas)
// v:            vector index
//
// Returns the (new_b0, new_b1) pair for the 128 cells at column v.

static inline CellPair vsum_and_transition(Bits3* const hs[WINDOW],
                                            const uint64_t* cur_b0_row,
                                            const uint64_t* cur_b1_row,
                                            int v) noexcept
{
    const Bits3 col0 = sum5(hs[0][v].b0, hs[1][v].b0, hs[2][v].b0, hs[3][v].b0, hs[4][v].b0);
    const Bits3 col1 = sum5(hs[0][v].b1, hs[1][v].b1, hs[2][v].b1, hs[3][v].b1, hs[4][v].b1);
    const Bits3 col2 = sum5(hs[0][v].b2, hs[1][v].b2, hs[2][v].b2, hs[3][v].b2, hs[4][v].b2);

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

    const uint64x2_t cur_b0 = vld1q_u64(cur_b0_row + 2*v);
    const uint64x2_t cur_b1 = vld1q_u64(cur_b1_row + 2*v);
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

    return { new_b0, new_b1 };
}

// ─────────────────────────────────────────────────────────────────────────────
// generate_row — compute one output row using a 5-row hsum window
// ─────────────────────────────────────────────────────────────────────────────
//
// hs[5]:   the hsum sliding window — hs[i] points to row (y-2+i)'s hsum
// cur_b0/b1: row y's b0/b1 planes (for centre-cell access in transitions)
// out_b0/b1: where to write the output row
// nvec:     vectors per row

static void generate_row(Bits3* const hs[WINDOW],
                         const uint64_t* cur_b0_row,
                         const uint64_t* cur_b1_row,
                         uint64_t* __restrict__ out_b0,
                         uint64_t* __restrict__ out_b1,
                         int nvec) noexcept
{
    for (int v = 0; v < nvec; ++v) {
        const CellPair p = vsum_and_transition(hs, cur_b0_row, cur_b1_row, v);
        vst1q_u64(out_b0 + 2*v, p.b0);
        vst1q_u64(out_b1 + 2*v, p.b1);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Sim — persistent thread pool, K=2 temporal blocking, hsum row caching
// ─────────────────────────────────────────────────────────────────────────────

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
    void step2(const Grid& src_, Grid& dst_) noexcept
    {
        src = &src_; dst = &dst_;
        start_bar.arrive_and_wait();
        done_bar.arrive_and_wait();
    }

private:
    // For each phase, drive the sliding hsum window over rows [y_first, y_last]
    // (inclusive on both ends) using b0_at / b1_at to fetch input rows, writing
    // to out_b0_at / out_b1_at.  The hsum cache is provided in `slots` — a 5-row
    // buffer pre-allocated by the caller.
    //
    // get_b{0,1}_at(y) returns a pointer to the input plane row at absolute
    // index y (with toroidal wrap pre-applied by the caller as appropriate).
    template <class GetB0, class GetB1, class OutB0, class OutB1>
    static void sweep_with_cache(int y_first, int y_last,
                                  GetB0 get_b0, GetB1 get_b1,
                                  OutB0 out_b0_at, OutB1 out_b1_at,
                                  Bits3* slot_storage,  // 5 * nvec entries
                                  int nvec) noexcept
    {
        // 5 logical-slot pointers into slot_storage.
        Bits3* slots[WINDOW];
        for (int i = 0; i < WINDOW; ++i) slots[i] = slot_storage + static_cast<size_t>(i) * nvec;

        // Warm-up: compute hsum for rows (y_first - 2) … (y_first + 2) into slots[0..4].
        for (int i = 0; i < WINDOW; ++i)
            hsum_row(get_b0(y_first - NEIGHBOUR_RANGE + i),
                     get_b1(y_first - NEIGHBOUR_RANGE + i),
                     nvec, slots[i]);

        // Main loop.
        for (int y = y_first; y <= y_last; ++y) {
            generate_row(slots, get_b0(y), get_b1(y),
                         out_b0_at(y), out_b1_at(y), nvec);

            if (y < y_last) {
                // Rotate: drop oldest (slots[0]), shift, recycle into slots[4].
                Bits3* recycled = slots[0];
                for (int i = 0; i < WINDOW - 1; ++i) slots[i] = slots[i + 1];
                slots[WINDOW - 1] = recycled;

                // Compute hsum for the new row (y+1)+2 = y+3 into recycled slot.
                hsum_row(get_b0(y + 1 + NEIGHBOUR_RANGE),
                         get_b1(y + 1 + NEIGHBOUR_RANGE),
                         nvec, slots[WINDOW - 1]);
            }
        }
    }

    void worker(int tid) noexcept
    {
        std::vector<uint64_t> tile_buf;
        std::vector<Bits3>    hsum_buf_phase1;   // 5 × nvec
        std::vector<Bits3>    hsum_buf_phase2;   // 5 × nvec

        while (true) {
            start_bar.arrive_and_wait();
            if (stop.load(std::memory_order_relaxed)) return;

            const int N             = src->size;
            const int wpr           = src->wpr;
            const int nvec          = wpr / 2;
            const int rows_per_thr  = N / NUM_THREADS;
            const int y_thr_start   = tid * rows_per_thr;
            const int y_thr_end     = y_thr_start + rows_per_thr;

            const size_t tile_words = static_cast<size_t>(BUF_ROWS) * 2 * wpr;
            if (tile_buf.size()         < tile_words)                  tile_buf.assign(tile_words, 0);
            const size_t hsum_words = static_cast<size_t>(WINDOW) * nvec;
            if (hsum_buf_phase1.size() < hsum_words) hsum_buf_phase1.assign(hsum_words, Bits3{});
            if (hsum_buf_phase2.size() < hsum_words) hsum_buf_phase2.assign(hsum_words, Bits3{});

            // Helpers into tile_buf
            auto buf_b0 = [&](int buf_idx) noexcept {
                return tile_buf.data() + static_cast<size_t>(2 * buf_idx)     * wpr;
            };
            auto buf_b1 = [&](int buf_idx) noexcept {
                return tile_buf.data() + static_cast<size_t>(2 * buf_idx + 1) * wpr;
            };

            // Tile-sweep within this thread's row band
            for (int tile_start = y_thr_start; tile_start < y_thr_end; tile_start += TILE_ROWS) {
                const int tile_end = std::min(tile_start + TILE_ROWS, y_thr_end);

                // ── Phase 1: gen+1 into tile_buf, rows [tile_start-2, tile_end+2) ──
                {
                    const Grid* s = src;
                    auto get_b0 = [&, s, N](int y) noexcept {
                        const int ny = (y + N) & (N - 1);
                        return s->b0_row(ny);
                    };
                    auto get_b1 = [&, s, N](int y) noexcept {
                        const int ny = (y + N) & (N - 1);
                        return s->b1_row(ny);
                    };
                    auto out_b0 = [&](int y) noexcept { return buf_b0(y - (tile_start - HALO)); };
                    auto out_b1 = [&](int y) noexcept { return buf_b1(y - (tile_start - HALO)); };

                    sweep_with_cache(tile_start - HALO, tile_end + HALO - 1,
                                     get_b0, get_b1, out_b0, out_b1,
                                     hsum_buf_phase1.data(), nvec);
                }

                // ── Phase 2: gen+2 to dst, rows [tile_start, tile_end) ─────────────
                {
                    auto get_b0 = [&](int y) noexcept { return buf_b0(y - (tile_start - HALO)); };
                    auto get_b1 = [&](int y) noexcept { return buf_b1(y - (tile_start - HALO)); };
                    Grid* d = dst;
                    auto out_b0 = [&, d](int y) noexcept { return d->b0_row(y); };
                    auto out_b1 = [&, d](int y) noexcept { return d->b1_row(y); };

                    sweep_with_cache(tile_start, tile_end - 1,
                                     get_b0, get_b1, out_b0, out_b1,
                                     hsum_buf_phase2.data(), nvec);
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
    if (!fin) { std::fprintf(stderr, "Error: cannot open '%s'\n", argv[1]); return 2; }

    uint64_t width = 0, height = 0;
    if (std::fread(&width, sizeof(uint64_t), 1, fin) != 1 ||
        std::fread(&height, sizeof(uint64_t), 1, fin) != 1) {
        std::fprintf(stderr, "Error: input too short (header)\n");
        std::fclose(fin); return 3;
    }
    if (width == 0 || width != height || width % 128 != 0) {
        std::fprintf(stderr, "Error: bad grid dims %" PRIu64 " × %" PRIu64 "\n", width, height);
        std::fclose(fin); return 3;
    }

    const int    grid_size  = static_cast<int>(width);
    const size_t cell_count = static_cast<size_t>(grid_size) * grid_size;

    std::vector<uint8_t> raw(cell_count);
    if (std::fread(raw.data(), 1, cell_count, fin) != cell_count) {
        std::fprintf(stderr, "Error: input too short\n");
        std::fclose(fin); return 4;
    }
    std::fclose(fin);

    Grid grid_a(grid_size);
    for (int y = 0; y < grid_size; ++y)
        grid_a.pack_row(y, raw.data() + static_cast<size_t>(y) * grid_size);
    raw.clear(); raw.shrink_to_fit();

    Grid  grid_b(grid_size);
    Grid* cur = &grid_a;
    Grid* nxt = &grid_b;

    Sim sim;

    const auto t0 = std::chrono::steady_clock::now();
    int gen = 0;
    for (; gen + K_GENS <= generations; gen += K_GENS) {
        sim.step2(*cur, *nxt);
        std::swap(cur, nxt);
    }
    const auto t1 = std::chrono::steady_clock::now();
    std::printf("%.3f ms\n",
        std::chrono::duration<double, std::milli>(t1 - t0).count());

    FILE* fout = std::fopen(argv[2], "wb");
    if (!fout) { std::fprintf(stderr, "Error: cannot open output '%s'\n", argv[2]); return 5; }

    std::vector<uint8_t> out_buf(cell_count);
    for (int y = 0; y < grid_size; ++y)
        cur->unpack_row(y, out_buf.data() + static_cast<size_t>(y) * grid_size);

    if (std::fwrite(&width, sizeof(uint64_t), 1, fout) != 1 ||
        std::fwrite(&height, sizeof(uint64_t), 1, fout) != 1 ||
        std::fwrite(out_buf.data(), 1, cell_count, fout) != cell_count) {
        std::fprintf(stderr, "Error: write error\n"); std::fclose(fout); return 6;
    }
    std::fclose(fout);
    return 0;
}
