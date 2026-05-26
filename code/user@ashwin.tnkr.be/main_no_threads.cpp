// code/main_no_threads.cpp — Monster Spawning Grid simulator
//
// Variant: SIMD bit-parallel word processing, SINGLE-THREADED.
//
// Identical to main.cpp except step() processes all rows in a plain nested
// loop instead of distributing work across NUM_THREADS threads.
//
// ─────────────────────────────────────────────────────────────────────────────
// INTERNAL REPRESENTATION
// ─────────────────────────────────────────────────────────────────────────────
//
// Three parallel bit-planes (egg, juv, adult) of uint64_t words.
// Cell (x, y) → word  y * words_per_row + (x >> 6),  bit  (x & 63).
//
// ─────────────────────────────────────────────────────────────────────────────

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

static constexpr int NEIGHBOUR_RANGE = 2;   // range-2 Moore neighbourhood

// ─────────────────────────────────────────────────────────────────────────────
// BitGrid
// ─────────────────────────────────────────────────────────────────────────────

struct BitGrid {
    int size;           // width == height; always a power of two, ≥ 64
    int words_per_row;  // size / 64

    std::vector<uint64_t> egg;
    std::vector<uint64_t> juv;
    std::vector<uint64_t> adult;

    BitGrid() : size(0), words_per_row(0) {}

    explicit BitGrid(int n)
        : size(n)
        , words_per_row(n / 64)
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

    inline void set_bit(std::vector<uint64_t>& plane,
                        int x, int y) const noexcept
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

    void clear() noexcept
    {
        std::fill(egg.begin(),   egg.end(),   uint64_t{0});
        std::fill(juv.begin(),   juv.end(),   uint64_t{0});
        std::fill(adult.begin(), adult.end(), uint64_t{0});
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Bit-parallel neighbour counting — SIMD word-level operations
// ─────────────────────────────────────────────────────────────────────────────

struct Bits3 {
    uint64_t b0 = 0;   // 1s place
    uint64_t b1 = 0;   // 2s place
    uint64_t b2 = 0;   // 4s place
};

// Add five 1-bit values per position via a CSA tree → 3-bit result.
static Bits3 sum5(uint64_t v0, uint64_t v1, uint64_t v2,
                  uint64_t v3, uint64_t v4) noexcept
{
    const uint64_t s1 = v0 ^ v1 ^ v2;
    const uint64_t c1 = (v0 & v1) | (v1 & v2) | (v0 & v2);

    const uint64_t s2 = v3 ^ v4 ^ s1;
    const uint64_t c2 = (v3 & v4) | (v4 & s1) | (v3 & s1);

    return { s2,  c1 ^ c2,  c1 & c2 };
}

// 5-wide horizontal adult sum for one word of one row.
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

// Advance one word (64 cells) by one generation.
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

    const Bits3 col0 = sum5(hsum[0].b0, hsum[1].b0, hsum[2].b0,
                            hsum[3].b0, hsum[4].b0);
    const Bits3 col1 = sum5(hsum[0].b1, hsum[1].b1, hsum[2].b1,
                            hsum[3].b1, hsum[4].b1);
    const Bits3 col2 = sum5(hsum[0].b2, hsum[1].b2, hsum[2].b2,
                            hsum[3].b2, hsum[4].b2);

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
// step — advance the entire grid by one generation (single-threaded)
// ─────────────────────────────────────────────────────────────────────────────

static void step(const BitGrid& src, BitGrid& dst) noexcept
{
    dst.clear();

    const int N   = src.size;
    const int wpr = src.words_per_row;

    for (int y = 0; y < N; ++y)
        for (int w = 0; w < wpr; ++w)
            step_word(src, dst, w, y);
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
    if (argc < 3 || argc > 4) {
        std::fprintf(stderr, "Usage: %s <input.bin> <output.bin> [generations]\n",
                     argv[0]);
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
            "Error: grid must be square and non-empty, got %" PRIu64 " × %" PRIu64 "\n",
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

    BitGrid  grid_b(grid_size);
    BitGrid* cur = &grid_a;
    BitGrid* nxt = &grid_b;

    const auto t0 = std::chrono::steady_clock::now();

    for (int gen = 0; gen < generations; ++gen) {
        step(*cur, *nxt);
        std::swap(cur, nxt);
    }

    const auto   t1         = std::chrono::steady_clock::now();
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("%.3f ms\n", elapsed_ms);

    FILE* fout = std::fopen(argv[2], "wb");
    if (!fout) {
        std::fprintf(stderr, "Error: cannot open output file '%s'\n", argv[2]);
        return 5;
    }

    std::vector<uint8_t> out_buf(cell_count);
    for (int y = 0; y < grid_size; ++y)
        cur->unpack_row(y, out_buf.data() + static_cast<size_t>(y) * grid_size);

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
