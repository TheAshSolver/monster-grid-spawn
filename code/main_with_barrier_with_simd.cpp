// code/main_with_barrier_with_simd.cpp — Monster Spawning Grid simulator
//
// Combines two optimisations from earlier files:
//
//   1. Persistent thread pool + std::barrier  (main_with_barrier.cpp)
//      Threads are created once and live for all generations.
//      A barrier synchronises them between generations; the completion
//      callback only swaps the src/dst pointers.
//
//   2. ARM NEON SIMD adult counting  (main_simd.cpp)
//      step_pair replaces step_word, processing 128 cells per operation
//      instead of 64.  The w-loop steps by 2.
//
//   3. No clear
//      step_pair (and step_word) unconditionally assigns every word in dst
//      with = so whatever was there before is always overwritten.
//      dst->clear() between generations is therefore redundant and removed.
//
// Falls back to the scalar path when __ARM_NEON is not defined.
// NEON path requires grid size >= 128 (words_per_row >= 2).

#include <algorithm>
#include <barrier>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#ifdef __ARM_NEON
#  include <arm_neon.h>
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

static constexpr int NEIGHBOUR_RANGE = 2;
static constexpr int NUM_THREADS     = 8;
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

    explicit BitGrid(int n)
        : size(n), words_per_row(n / 64)
    {
        const size_t total = static_cast<size_t>(n) * words_per_row;
        egg.assign(total, 0);
        juv.assign(total, 0);
        adult.assign(total, 0);
    }

    [[nodiscard]] inline bool get_bit(const std::vector<uint64_t>& plane,
                                      int x, int y) const noexcept
    {
        return (plane[static_cast<size_t>(y) * words_per_row + (x >> 6)]
                >> (x & 63)) & 1u;
    }

    inline void set_bit(std::vector<uint64_t>& plane, int x, int y) const noexcept
    {
        plane[static_cast<size_t>(y) * words_per_row + (x >> 6)]
            |= uint64_t{1} << (x & 63);
    }

    void pack_row(int y, const uint8_t* row_bytes) noexcept
    {
        const size_t base = static_cast<size_t>(y) * words_per_row;
        for (int w = 0; w < words_per_row; ++w) {
            uint64_t e = 0, j = 0, a = 0;
            const uint8_t* src = row_bytes + w * 64;
            for (int b = 0; b < 64; ++b) {
                const uint64_t mask = uint64_t{1} << b;
                switch (src[b]) {
                    case 1: e |= mask; break;
                    case 2: j |= mask; break;
                    case 3: a |= mask; break;
                    default: break;
                }
            }
            egg  [base + w] = e;
            juv  [base + w] = j;
            adult[base + w] = a;
        }
    }

    void unpack_row(int y, uint8_t* row_bytes) const noexcept
    {
        const size_t base = static_cast<size_t>(y) * words_per_row;
        for (int w = 0; w < words_per_row; ++w) {
            const uint64_t e = egg  [base + w];
            const uint64_t j = juv  [base + w];
            const uint64_t a = adult[base + w];
            uint8_t* dst = row_bytes + w * 64;
            for (int b = 0; b < 64; ++b) {
                const uint64_t mask = uint64_t{1} << b;
                if      (e & mask) dst[b] = 1;
                else if (j & mask) dst[b] = 2;
                else if (a & mask) dst[b] = 3;
                else               dst[b] = 0;
            }
        }
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Global grid pointers + ThreadPool
// ─────────────────────────────────────────────────────────────────────────────

static BitGrid* src = nullptr;
static BitGrid* dst = nullptr;

struct ThreadPool {
    std::vector<std::thread> workers;

    struct BarrierPhaseCompletion {
        void operator()() noexcept {
            std::swap(src, dst);
            // No clear: step_pair/step_word assigns every dst word unconditionally.
        }
    };

    std::barrier<BarrierPhaseCompletion> b{NUM_THREADS, BarrierPhaseCompletion{}};
};

static void step_thread(int y_start, ThreadPool& pool) noexcept;

void initialize_threads(ThreadPool& pool)
{
    const int N               = src->size;
    const int rows_per_thread = N / NUM_THREADS;

    for (int t = 0; t < NUM_THREADS; ++t)
        pool.workers.emplace_back(step_thread, t * rows_per_thread, std::ref(pool));

    for (auto& worker : pool.workers)
        if (worker.joinable())
            worker.join();
}

// ─────────────────────────────────────────────────────────────────────────────
// Scalar path
// ─────────────────────────────────────────────────────────────────────────────

struct Bits3 {
    uint64_t b0 = 0;
    uint64_t b1 = 0;
    uint64_t b2 = 0;
};

static Bits3 sum5(uint64_t v0, uint64_t v1, uint64_t v2,
                  uint64_t v3, uint64_t v4) noexcept
{
    const uint64_t s1 = v0 ^ v1 ^ v2;
    const uint64_t c1 = (v0 & v1) | (v1 & v2) | (v0 & v2);

    const uint64_t s2 = v3 ^ v4 ^ s1;
    const uint64_t c2 = (v3 & v4) | (v4 & s1) | (v3 & s1);

    return { s2, c1 ^ c2, c1 & c2 };
}

static Bits3 hsum_word(const uint64_t* adult_row, int wpr, int w) noexcept
{
    const uint64_t L = adult_row[(w - 1 + wpr) & (wpr - 1)];
    const uint64_t C = adult_row[w];
    const uint64_t R = adult_row[(w + 1)        & (wpr - 1)];

    const uint64_t col_m2 = (C << 2) | (L >> 62);
    const uint64_t col_m1 = (C << 1) | (L >> 63);
    const uint64_t col_0  =  C;
    const uint64_t col_p1 = (C >> 1) | (R << 63);
    const uint64_t col_p2 = (C >> 2) | (R << 62);

    return sum5(col_m2, col_m1, col_0, col_p1, col_p2);
}

static void step_word(const BitGrid& src, BitGrid& dst, int w, int y) noexcept
{
    const int    N   = src.size;
    const int    wpr = src.words_per_row;
    const size_t wi  = static_cast<size_t>(y) * wpr + w;

    Bits3 hsum[2 * NEIGHBOUR_RANGE + 1];
    for (int dy = -NEIGHBOUR_RANGE; dy <= NEIGHBOUR_RANGE; ++dy) {
        const int ny = (y + dy + N) & (N - 1);
        hsum[dy + NEIGHBOUR_RANGE] =
            hsum_word(src.adult.data() + static_cast<size_t>(ny) * wpr, wpr, w);
    }

    const Bits3 col0 = sum5(hsum[0].b0, hsum[1].b0, hsum[2].b0, hsum[3].b0, hsum[4].b0);
    const Bits3 col1 = sum5(hsum[0].b1, hsum[1].b1, hsum[2].b1, hsum[3].b1, hsum[4].b1);
    const Bits3 col2 = sum5(hsum[0].b2, hsum[1].b2, hsum[2].b2, hsum[3].b2, hsum[4].b2);

    const uint64_t n0 = col0.b0;

    const uint64_t n1   = col0.b1 ^ col1.b0;
    const uint64_t cy1  = col0.b1 & col1.b0;

    const uint64_t t2   = col0.b2 ^ col1.b1 ^ col2.b0;
    const uint64_t cy2a = (col0.b2 & col1.b1) | (col1.b1 & col2.b0) | (col0.b2 & col2.b0);
    const uint64_t n2   = t2 ^ cy1;
    const uint64_t cy2b = t2 & cy1;

    const uint64_t t3   = col1.b2 ^ col2.b1 ^ cy2a;
    const uint64_t cy3a = (col1.b2 & col2.b1) | (col2.b1 & cy2a) | (col1.b2 & cy2a);
    const uint64_t n3   = t3 ^ cy2b;
    const uint64_t cy3b = t3 & cy2b;

    const uint64_t n4   = col2.b2 ^ cy3a ^ cy3b;

    const uint64_t centre = src.adult[wi];

    uint64_t a0 = n0, a1 = n1, a2 = n2, a3 = n3, a4 = n4;
    {
        const uint64_t br0 = centre & ~a0;   a0 ^= centre;
        const uint64_t br1 = br0    & ~a1;   a1 ^= br0;
        const uint64_t br2 = br1    & ~a2;   a2 ^= br1;
        const uint64_t br3 = br2    & ~a3;   a3 ^= br2;
                                              a4 ^= br3;
    }

    const uint64_t cond_3_to_5 = ~a4 & ~a3 & ~(a2 & a1) & (a2 | (a1 & a0));
    const uint64_t cond_4_to_9 = (a3 | a2) & ~a4 & ~(a3 & (a2 | a1));

    const uint64_t cur_egg  = src.egg  [wi];
    const uint64_t cur_juv  = src.juv  [wi];
    const uint64_t cur_adlt = src.adult[wi];
    const uint64_t cur_empt = ~(cur_egg | cur_juv | cur_adlt);

    dst.egg  [wi] = cur_empt & cond_3_to_5;
    dst.juv  [wi] = cur_egg;
    dst.adult[wi] = cur_juv | (cur_adlt & cond_4_to_9);
}

// ─────────────────────────────────────────────────────────────────────────────
// NEON path
// ─────────────────────────────────────────────────────────────────────────────

#ifdef __ARM_NEON

static inline uint64x2_t vnot(uint64x2_t a) noexcept
{
    return veorq_u64(a, vdupq_n_u64(~uint64_t{0}));
}

struct NBits3 {
    uint64x2_t b0;
    uint64x2_t b1;
    uint64x2_t b2;
};

static NBits3 nsum5(uint64x2_t v0, uint64x2_t v1, uint64x2_t v2,
                    uint64x2_t v3, uint64x2_t v4) noexcept
{
    const uint64x2_t s1 = veorq_u64(veorq_u64(v0, v1), v2);
    const uint64x2_t c1 = vorrq_u64(vorrq_u64(vandq_u64(v0, v1),
                                               vandq_u64(v1, v2)),
                                               vandq_u64(v0, v2));
    const uint64x2_t s2 = veorq_u64(veorq_u64(v3, v4), s1);
    const uint64x2_t c2 = vorrq_u64(vorrq_u64(vandq_u64(v3, v4),
                                               vandq_u64(v4, s1)),
                                               vandq_u64(v3, s1));
    return { s2, veorq_u64(c1, c2), vandq_u64(c1, c2) };
}

// hsum_pair: horizontal adult sum for two consecutive words (w, w+1).
//
//   C  = { word[w], word[w+1] }
//   vextq_u64(A, B, 1) = { A[1], B[0] }
//   L_ctx[lane i] = word[w+i-1],  R_ctx[lane i] = word[w+i+1]
//
// Precondition: wpr >= 2, w even.
static NBits3 hsum_pair(const uint64_t* adult_row, int wpr, int w) noexcept
{
    const uint64x2_t L_vec = vld1q_u64(&adult_row[(w - 2 + wpr) & (wpr - 1)]);
    const uint64x2_t C     = vld1q_u64(&adult_row[w]);
    const uint64x2_t R_vec = vld1q_u64(&adult_row[(w + 2)        & (wpr - 1)]);

    const uint64x2_t L_ctx = vextq_u64(L_vec, C,     1);
    const uint64x2_t R_ctx = vextq_u64(C,     R_vec, 1);

    const uint64x2_t col_m2 = vorrq_u64(vshlq_n_u64(C, 2), vshrq_n_u64(L_ctx, 62));
    const uint64x2_t col_m1 = vorrq_u64(vshlq_n_u64(C, 1), vshrq_n_u64(L_ctx, 63));
    const uint64x2_t col_0  = C;
    const uint64x2_t col_p1 = vorrq_u64(vshrq_n_u64(C, 1), vshlq_n_u64(R_ctx, 63));
    const uint64x2_t col_p2 = vorrq_u64(vshrq_n_u64(C, 2), vshlq_n_u64(R_ctx, 62));

    return nsum5(col_m2, col_m1, col_0, col_p1, col_p2);
}

// step_pair: advance words (w, w+1) by one generation — 128 cells at once.
static void step_pair(const BitGrid& src, BitGrid& dst, int w, int y) noexcept
{
    const int    N   = src.size;
    const int    wpr = src.words_per_row;
    const size_t wi  = static_cast<size_t>(y) * wpr + w;

    // Phase 1: horizontal sum for each of the 5 contributing rows
    NBits3 hsum[2 * NEIGHBOUR_RANGE + 1];
    for (int dy = -NEIGHBOUR_RANGE; dy <= NEIGHBOUR_RANGE; ++dy) {
        const int ny = (y + dy + N) & (N - 1);
        hsum[dy + NEIGHBOUR_RANGE] =
            hsum_pair(src.adult.data() + static_cast<size_t>(ny) * wpr, wpr, w);
    }

    // Phase 2: vertical sum
    const NBits3 col0 = nsum5(hsum[0].b0, hsum[1].b0, hsum[2].b0, hsum[3].b0, hsum[4].b0);
    const NBits3 col1 = nsum5(hsum[0].b1, hsum[1].b1, hsum[2].b1, hsum[3].b1, hsum[4].b1);
    const NBits3 col2 = nsum5(hsum[0].b2, hsum[1].b2, hsum[2].b2, hsum[3].b2, hsum[4].b2);

    // Ripple-carry adder
    const uint64x2_t n0 = col0.b0;

    const uint64x2_t n1   = veorq_u64(col0.b1, col1.b0);
    const uint64x2_t cy1  = vandq_u64(col0.b1, col1.b0);

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

    // Subtract centre cell; cur_adlt doubles as centre — same data, one load.
    const uint64x2_t cur_adlt = vld1q_u64(&src.adult[wi]);

    uint64x2_t a0 = n0, a1 = n1, a2 = n2, a3 = n3, a4 = n4;
    {
        const uint64x2_t br0 = vandq_u64(cur_adlt, vnot(a0)); a0 = veorq_u64(a0, cur_adlt);
        const uint64x2_t br1 = vandq_u64(br0,      vnot(a1)); a1 = veorq_u64(a1, br0);
        const uint64x2_t br2 = vandq_u64(br1,      vnot(a2)); a2 = veorq_u64(a2, br1);
        const uint64x2_t br3 = vandq_u64(br2,      vnot(a3)); a3 = veorq_u64(a3, br2);
                                                                a4 = veorq_u64(a4, br3);
    }

    // Phase 3: transition rules
    const uint64x2_t cond_3_to_5 =
        vandq_u64(vandq_u64(vandq_u64(vnot(a4), vnot(a3)),
                            vnot(vandq_u64(a2, a1))),
                            vorrq_u64(a2, vandq_u64(a1, a0)));

    const uint64x2_t cond_4_to_9 =
        vandq_u64(vandq_u64(vorrq_u64(a3, a2), vnot(a4)),
                            vnot(vandq_u64(a3, vorrq_u64(a2, a1))));

    const uint64x2_t cur_egg  = vld1q_u64(&src.egg[wi]);
    const uint64x2_t cur_juv  = vld1q_u64(&src.juv[wi]);
    const uint64x2_t cur_empt = vnot(vorrq_u64(vorrq_u64(cur_egg, cur_juv), cur_adlt));

    vst1q_u64(&dst.egg  [wi], vandq_u64(cur_empt, cond_3_to_5));
    vst1q_u64(&dst.juv  [wi], cur_egg);
    vst1q_u64(&dst.adult[wi], vorrq_u64(cur_juv, vandq_u64(cur_adlt, cond_4_to_9)));
}

#endif // __ARM_NEON

// ─────────────────────────────────────────────────────────────────────────────
// step_thread — persistent worker; lives for all GENERATIONS
// ─────────────────────────────────────────────────────────────────────────────

static void step_thread(int y_start, ThreadPool& pool) noexcept
{
    const int N               = src->size;
    const int wpr             = src->words_per_row;
    const int rows_per_thread = N / NUM_THREADS;
    const int y_end           = y_start + rows_per_thread;

    for (int i = 0; i < GENERATIONS; ++i) {
        const BitGrid& current_src = *src;
        BitGrid&       current_dst = *dst;

        for (int y = y_start; y < y_end; ++y) {
#ifdef __ARM_NEON
            for (int w = 0; w < wpr; w += 2)
                step_pair(current_src, current_dst, w, y);
#else
            for (int w = 0; w < wpr; ++w)
                step_word(current_src, current_dst, w, y);
#endif
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
    if (!fin) {
        std::fprintf(stderr, "Error: cannot open input file '%s'\n", argv[1]);
        return 2;
    }

    uint64_t width = 0, height = 0;
    if (std::fread(&width,  sizeof(uint64_t), 1, fin) != 1 ||
        std::fread(&height, sizeof(uint64_t), 1, fin) != 1)
    {
        std::fprintf(stderr, "Error: input file too short (cannot read header)\n");
        std::fclose(fin);
        return 3;
    }

    if (width == 0 || width != height) {
        std::fprintf(stderr,
            "Error: grid must be square and non-empty, got %" PRIu64 " x %" PRIu64 "\n",
            width, height);
        std::fclose(fin);
        return 3;
    }

    if (width % 64 != 0) {
        std::fprintf(stderr,
            "Error: grid width must be a multiple of 64, got %" PRIu64 "\n", width);
        std::fclose(fin);
        return 3;
    }

#ifdef __ARM_NEON
    if (width < 128) {
        std::fprintf(stderr,
            "Error: NEON path requires grid size >= 128, got %" PRIu64 "\n", width);
        std::fclose(fin);
        return 3;
    }
#endif

    const int    grid_size  = static_cast<int>(width);
    const size_t cell_count = static_cast<size_t>(grid_size) * grid_size;

    std::vector<uint8_t> raw(cell_count);
    if (std::fread(raw.data(), 1, cell_count, fin) != cell_count) {
        std::fprintf(stderr, "Error: input file too short (cell data truncated)\n");
        std::fclose(fin);
        return 4;
    }
    std::fclose(fin);

    BitGrid grid_a(grid_size);
    for (int y = 0; y < grid_size; ++y)
        grid_a.pack_row(y, raw.data() + static_cast<size_t>(y) * grid_size);

    raw.clear();
    raw.shrink_to_fit();

    BitGrid grid_b(grid_size);
    src = &grid_a;
    dst = &grid_b;

    const auto t0 = std::chrono::steady_clock::now();

    ThreadPool p;
    initialize_threads(p);

    const auto   t1         = std::chrono::steady_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("%.3f ms\n", elapsed_ms);

    FILE* fout = std::fopen(argv[2], "wb");
    if (!fout) {
        std::fprintf(stderr, "Error: cannot open output file '%s'\n", argv[2]);
        return 5;
    }

    std::vector<uint8_t> out_buf(cell_count);
    for (int y = 0; y < grid_size; ++y)
        src->unpack_row(y, out_buf.data() + static_cast<size_t>(y) * grid_size);

    if (std::fwrite(&width,         sizeof(uint64_t), 1,          fout) != 1  ||
        std::fwrite(&height,        sizeof(uint64_t), 1,          fout) != 1  ||
        std::fwrite(out_buf.data(), 1,                cell_count, fout) != cell_count)
    {
        std::fprintf(stderr, "Error: write error on output file '%s'\n", argv[2]);
        std::fclose(fout);
        return 6;
    }

    std::fclose(fout);
    return 0;
}
