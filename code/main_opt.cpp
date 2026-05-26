// code/main_opt.cpp — optimised Monster Spawning Grid simulator
//
// Changes vs main.cpp:
//
//   1. TWO-PLANE ENCODING
//        Three planes (egg, juv, adult) → two planes (b0, b1):
//
//          b1  b0  →  state
//           0   0  →  EMPTY
//           0   1  →  EGG
//           1   0  →  JUVENILE
//           1   1  →  ADULT
//
//        adult = b1 & b0.  Reduces per-generation memory traffic by ~33%
//        (512 MB vs 768 MB for a 32 768² grid) and also eliminates the
//        redundant dst.clear() — every word is overwritten directly.
//
//   2. NEON SIMD (ARM uint64x2_t — 128 cells per operation)
//        Replaces plain uint64_t throughout hsum / step.  Cross-lane bit
//        carries use vextq_u64.  Algorithm is identical to main.cpp; only
//        the word width doubles.
//
//   3. PERSISTENT THREAD POOL via std::barrier
//        main.cpp creates and joins 8 threads on every step() call
//        (80 000 thread-creation round-trips for 10 000 generations).
//        Here threads are created once at startup and rendezvous each
//        generation on two std::barrier<> checkpoints: start_bar (main
//        releases workers) and done_bar (workers signal completion).

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

static constexpr int NEIGHBOUR_RANGE = 2;
static constexpr int NUM_THREADS     = 8;

// ─────────────────────────────────────────────────────────────────────────────
// Grid2 — two-plane bit-packed grid
// ─────────────────────────────────────────────────────────────────────────────
//
// Cell (x,y) lives in:
//   word  =  y * wpr + (x >> 6)
//   bit   =  (x & 63)
//
// Each NEON vector covers two consecutive words = 128 cells.

struct Grid2 {
    int size;
    int wpr;    // uint64_t words per row = size / 64

    std::vector<uint64_t> b0;
    std::vector<uint64_t> b1;

    Grid2() : size(0), wpr(0) {}

    explicit Grid2(int n) : size(n), wpr(n / 64)
    {
        const size_t total = static_cast<size_t>(n) * wpr;
        b0.assign(total, 0);
        b1.assign(total, 0);
    }

    // byte-per-cell (0/1/2/3) → two-plane bit-packed
    void pack_row(int y, const uint8_t* row_bytes) noexcept
    {
        const size_t base = static_cast<size_t>(y) * wpr;
        for (int w = 0; w < wpr; ++w) {
            uint64_t lo = 0, hi = 0;
            const uint8_t* src = row_bytes + w * 64;
            for (int b = 0; b < 64; ++b) {
                const uint64_t mask = uint64_t{1} << b;
                if (src[b] & 1) lo |= mask;   // b0 = bit 0 of state
                if (src[b] & 2) hi |= mask;   // b1 = bit 1 of state
            }
            b0[base + w] = lo;
            b1[base + w] = hi;
        }
    }

    // two-plane bit-packed → byte-per-cell (0/1/2/3)
    void unpack_row(int y, uint8_t* row_bytes) const noexcept
    {
        const size_t base = static_cast<size_t>(y) * wpr;
        for (int w = 0; w < wpr; ++w) {
            const uint64_t lo = b0[base + w];
            const uint64_t hi = b1[base + w];
            uint8_t* dst = row_bytes + w * 64;
            for (int b = 0; b < 64; ++b)
                dst[b] = static_cast<uint8_t>(((lo >> b) & 1) | (((hi >> b) & 1) << 1));
        }
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// NEON bit-parallel neighbour counting
// ─────────────────────────────────────────────────────────────────────────────

// Bitwise NOT for uint64x2_t — notq does not exist in NEON.
static inline uint64x2_t notq(uint64x2_t x) noexcept
{
    return vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(x)));
}

// 3-bit integer per bit-position, backed by NEON words.
struct Bits3 {
    uint64x2_t b0, b1, b2;
};

// Carry-save adder tree: five 1-bit inputs → 3-bit sum (per bit position).
static inline Bits3 sum5(uint64x2_t v0, uint64x2_t v1, uint64x2_t v2,
                         uint64x2_t v3, uint64x2_t v4) noexcept
{
    // Stage 1: v0,v1,v2 → s1 (1s place), c1 (2s place)
    const uint64x2_t s1 = veorq_u64(veorq_u64(v0, v1), v2);
    const uint64x2_t c1 = vorrq_u64(vorrq_u64(vandq_u64(v0, v1), vandq_u64(v1, v2)),
                                     vandq_u64(v0, v2));
    // Stage 2: v3,v4,s1 → s2, c2
    const uint64x2_t s2 = veorq_u64(veorq_u64(v3, v4), s1);
    const uint64x2_t c2 = vorrq_u64(vorrq_u64(vandq_u64(v3, v4), vandq_u64(v4, s1)),
                                     vandq_u64(v3, s1));
    // Stage 3: half-adder on carries → 2s and 4s places
    return { s2, veorq_u64(c1, c2), vandq_u64(c1, c2) };
}

// 5-wide horizontal adult sum for one NEON vector (128 cells) in one row.
//
// adult = b1 & b0, computed on the fly — no separate adult plane needed.
//
// Cross-lane carries:
//   vextq_u64(a, b, 1) = { a[1], b[0] }   (lane 1 of a, lane 0 of b)
//
// Shift left by k (cell at x-k → position x):
//   vshlq_n_u64(C, k) | vshrq_n_u64({L[1],C[0]}, 64-k)
//
// Shift right by k (cell at x+k → position x):
//   vshrq_n_u64(C, k) | vshlq_n_u64({C[1],R[0]}, 64-k)

static inline Bits3 hsum_vec(const uint64_t* __restrict__ b0_row,
                              const uint64_t* __restrict__ b1_row,
                              int nvec, int v) noexcept
{
    const int vL = (v - 1 + nvec) & (nvec - 1);
    const int vR = (v + 1)        & (nvec - 1);

    const uint64x2_t L = vandq_u64(vld1q_u64(b0_row + 2*vL), vld1q_u64(b1_row + 2*vL));
    const uint64x2_t C = vandq_u64(vld1q_u64(b0_row + 2*v),  vld1q_u64(b1_row + 2*v));
    const uint64x2_t R = vandq_u64(vld1q_u64(b0_row + 2*vR), vld1q_u64(b1_row + 2*vR));

    const uint64x2_t prev = vextq_u64(L, C, 1);   // {L[1], C[0]}
    const uint64x2_t next = vextq_u64(C, R, 1);   // {C[1], R[0]}

    const uint64x2_t col_m2 = vorrq_u64(vshlq_n_u64(C, 2), vshrq_n_u64(prev, 62));
    const uint64x2_t col_m1 = vorrq_u64(vshlq_n_u64(C, 1), vshrq_n_u64(prev, 63));
    const uint64x2_t col_0  = C;
    const uint64x2_t col_p1 = vorrq_u64(vshrq_n_u64(C, 1), vshlq_n_u64(next, 63));
    const uint64x2_t col_p2 = vorrq_u64(vshrq_n_u64(C, 2), vshlq_n_u64(next, 62));

    return sum5(col_m2, col_m1, col_0, col_p1, col_p2);
}

// ─────────────────────────────────────────────────────────────────────────────
// step_vec — advance one NEON vector (128 cells) by one generation
// ─────────────────────────────────────────────────────────────────────────────

static void step_vec(const Grid2& src, Grid2& dst, int v, int y) noexcept
{
    const int    N    = src.size;
    const int    wpr  = src.wpr;
    const int    nvec = wpr / 2;
    const size_t vi   = static_cast<size_t>(y) * nvec + v;

    // Phase 1: horizontal adult sum for each of the 5 contributing rows.
    Bits3 hsum[2 * NEIGHBOUR_RANGE + 1];
    for (int dy = -NEIGHBOUR_RANGE; dy <= NEIGHBOUR_RANGE; ++dy) {
        const int ny = (y + dy + N) & (N - 1);
        hsum[dy + NEIGHBOUR_RANGE] = hsum_vec(
            src.b0.data() + static_cast<size_t>(ny) * wpr,
            src.b1.data() + static_cast<size_t>(ny) * wpr,
            nvec, v);
    }

    // Phase 2: vertical sum of the five 3-bit counts → 5-bit adult box total.
    const Bits3 col0 = sum5(hsum[0].b0, hsum[1].b0, hsum[2].b0, hsum[3].b0, hsum[4].b0);
    const Bits3 col1 = sum5(hsum[0].b1, hsum[1].b1, hsum[2].b1, hsum[3].b1, hsum[4].b1);
    const Bits3 col2 = sum5(hsum[0].b2, hsum[1].b2, hsum[2].b2, hsum[3].b2, hsum[4].b2);

    // Ripple-carry adder combines weighted column sums into one 5-bit count.
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

    // Subtract the centre adult cell from the box total to get neighbourhood count.
    const uint64x2_t cur_b0 = vld1q_u64(src.b0.data() + vi * 2);
    const uint64x2_t cur_b1 = vld1q_u64(src.b1.data() + vi * 2);
    const uint64x2_t centre = vandq_u64(cur_b0, cur_b1);   // adult = b1 & b0

    uint64x2_t a0 = n0, a1 = n1, a2 = n2, a3 = n3, a4 = n4;
    {
        const uint64x2_t br0 = vandq_u64(centre, notq(a0)); a0 = veorq_u64(a0, centre);
        const uint64x2_t br1 = vandq_u64(br0,    notq(a1)); a1 = veorq_u64(a1, br0);
        const uint64x2_t br2 = vandq_u64(br1,    notq(a2)); a2 = veorq_u64(a2, br1);
        const uint64x2_t br3 = vandq_u64(br2,    notq(a3)); a3 = veorq_u64(a3, br2);
                                                                    a4 = veorq_u64(a4, br3);
    }

    // Phase 3: range conditions (same Karnaugh-map derivations as main.cpp).
    const uint64x2_t cond_3_to_5 =
        vandq_u64(vandq_u64(notq(a4), notq(a3)),
                  vandq_u64(notq(vandq_u64(a2, a1)),
                             vorrq_u64(a2, vandq_u64(a1, a0))));

    const uint64x2_t cond_4_to_9 =
        vandq_u64(vandq_u64(vorrq_u64(a3, a2), notq(a4)),
                  notq(vandq_u64(a3, vorrq_u64(a2, a1))));

    // Two-plane state transitions (derived from truth table):
    //
    //   cur state → new state
    //   EMPTY (00)  → EGG   (01)  if cond_3_to_5, else EMPTY (00)
    //   EGG   (01)  → JUV   (10)  always
    //   JUV   (10)  → ADULT (11)  always
    //   ADULT (11)  → ADULT (11)  if cond_4_to_9, else EMPTY (00)
    //
    //   new_b1 = (cur_b0 ^ cur_b1)          ← EGG→JUV and JUV→ADULT
    //          | (adult & cond_4_to_9)       ← ADULT→ADULT
    //
    //   new_b0 = (empty & cond_3_to_5)       ← EMPTY→EGG
    //          | cur_juv                     ← JUV→ADULT
    //          | (adult & cond_4_to_9)       ← ADULT→ADULT

    const uint64x2_t adult_survive = vandq_u64(centre, cond_4_to_9);

    const uint64x2_t new_b1 = vorrq_u64(veorq_u64(cur_b0, cur_b1), adult_survive);

    const uint64x2_t cur_juv   = vandq_u64(cur_b1, notq(cur_b0));
    const uint64x2_t cur_empty = vandq_u64(notq(cur_b1), notq(cur_b0));
    const uint64x2_t new_b0    = vorrq_u64(
        vorrq_u64(vandq_u64(cur_empty, cond_3_to_5), cur_juv),
        adult_survive);

    vst1q_u64(dst.b0.data() + vi * 2, new_b0);
    vst1q_u64(dst.b1.data() + vi * 2, new_b1);
}

// ─────────────────────────────────────────────────────────────────────────────
// Sim — persistent thread pool coordinated with std::barrier
// ─────────────────────────────────────────────────────────────────────────────
//
// Two barriers, each with (NUM_THREADS + 1) parties:
//
//   start_bar: main sets src/dst, then arrives → workers wake up and process
//   done_bar:  workers arrive when done → main unblocks for next generation
//
// Memory ordering: std::barrier::arrive_and_wait() is a full synchronisation
// point, so the src/dst pointer writes (before start_bar) and the grid data
// writes (before done_bar) are guaranteed visible across threads.

struct Sim {
    std::barrier<>       start_bar{NUM_THREADS + 1};
    std::barrier<>       done_bar {NUM_THREADS + 1};
    const Grid2*         src  = nullptr;
    Grid2*               dst  = nullptr;
    std::atomic<bool>    stop {false};
    std::thread          threads[NUM_THREADS];

    Sim()
    {
        for (int t = 0; t < NUM_THREADS; ++t)
            threads[t] = std::thread(&Sim::worker, this, t);
    }

    ~Sim()
    {
        stop.store(true, std::memory_order_relaxed);
        start_bar.arrive_and_wait();           // wake workers one last time
        for (auto& t : threads) t.join();      // workers see stop=true and exit
    }

    void step(const Grid2& src_, Grid2& dst_) noexcept
    {
        src = &src_;
        dst = &dst_;
        start_bar.arrive_and_wait();           // release workers
        done_bar.arrive_and_wait();            // wait for all workers
    }

private:
    void worker(int tid) noexcept
    {
        while (true) {
            start_bar.arrive_and_wait();
            if (stop.load(std::memory_order_relaxed)) return;

            const int N            = src->size;
            const int nvec         = src->wpr / 2;
            const int rows_per_thr = N / NUM_THREADS;
            const int y0           = tid * rows_per_thr;
            const int y1           = y0 + rows_per_thr;

            for (int y = y0; y < y1; ++y)
                for (int v = 0; v < nvec; ++v)
                    step_vec(*src, *dst, v, y);

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

    Grid2 grid_a(grid_size);
    for (int y = 0; y < grid_size; ++y)
        grid_a.pack_row(y, raw.data() + static_cast<size_t>(y) * grid_size);
    raw.clear();
    raw.shrink_to_fit();

    Grid2  grid_b(grid_size);
    Grid2* cur = &grid_a;
    Grid2* nxt = &grid_b;

    Sim sim;

    const auto t0 = std::chrono::steady_clock::now();
    for (int gen = 0; gen < generations; ++gen) {
        sim.step(*cur, *nxt);
        std::swap(cur, nxt);
    }
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
