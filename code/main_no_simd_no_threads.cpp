// code/main_no_simd_no_threads.cpp — Monster Spawning Grid simulator
//
// Variant: SCALAR cell-by-cell neighbour counting, SINGLE-THREADED.
//
// The simplest optimised variant: bit-plane storage for I/O compatibility,
// but the simulation kernel processes one cell at a time with a plain loop
// over the 24 neighbours.  No threads, no word-level bit operations.
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

static constexpr int NEIGHBOUR_RANGE = 2;

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
// Scalar neighbour counting and per-cell transition
// ─────────────────────────────────────────────────────────────────────────────

// Count adult cells in the range-2 Moore neighbourhood (24 cells) of (x, y).
static int count_adult_neighbours(const BitGrid& g, int x, int y) noexcept
{
    const int N = g.size;
    int count = 0;
    for (int dy = -NEIGHBOUR_RANGE; dy <= NEIGHBOUR_RANGE; ++dy)
        for (int dx = -NEIGHBOUR_RANGE; dx <= NEIGHBOUR_RANGE; ++dx)
            if (dx != 0 || dy != 0)
                count += g.get_bit(g.adult,
                                   (x + dx + N) & (N - 1),
                                   (y + dy + N) & (N - 1));
    return count;
}

// Apply transition rules for one cell and write to dst.
static void step_cell(const BitGrid& src, BitGrid& dst, int x, int y) noexcept
{
    const int A = count_adult_neighbours(src, x, y);

    const bool is_egg  = src.get_bit(src.egg,   x, y);
    const bool is_juv  = src.get_bit(src.juv,   x, y);
    const bool is_adlt = src.get_bit(src.adult,  x, y);
    const bool is_empt = !is_egg && !is_juv && !is_adlt;

    //   EMPTY   → EGG      if 3 ≤ A ≤ 5
    //   EGG     → JUVENILE always
    //   JUVENILE→ ADULT    always
    //   ADULT   → ADULT    if 4 ≤ A ≤ 9,  else → EMPTY

    if (is_empt && A >= 3 && A <= 5)  dst.set_bit(dst.egg,   x, y);
    if (is_egg)                        dst.set_bit(dst.juv,   x, y);
    if (is_juv)                        dst.set_bit(dst.adult, x, y);
    if (is_adlt && A >= 4 && A <= 9)   dst.set_bit(dst.adult, x, y);
}

// ─────────────────────────────────────────────────────────────────────────────
// step — advance the entire grid by one generation (single-threaded, scalar)
// ─────────────────────────────────────────────────────────────────────────────

static void step(const BitGrid& src, BitGrid& dst) noexcept
{
    dst.clear();

    const int N = src.size;

    for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x)
            step_cell(src, dst, x, y);
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
