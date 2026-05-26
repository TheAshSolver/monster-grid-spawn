// code/main.cpp — Monster Spawning Grid simulator
//
// ─────────────────────────────────────────────────────────────────────────────
// INTERNAL REPRESENTATION
// ─────────────────────────────────────────────────────────────────────────────
//
// The grid is stored as three parallel bit-planes:
//
//     egg    juv    adult    →  cell state
//     ───    ───    ─────
//      1      0      0      →  EGG
//      0      1      0      →  JUVENILE
//      0      0      1      →  ADULT
//      0      0      0      →  EMPTY
//
// Each bit-plane is a flat array of uint64_t words, row-major.
// Cell (x, y) lives in:
//
//     word  =  y * words_per_row + (x >> 6)
//     bit   =  (x & 63)          [bit 0 = smallest x in group]
//
// This layout enables a future SIMD path to process 64 cells per instruction
// using shift/OR/POPCNT on the adult bit-plane for neighbour counting, and
// bitwise logic for the state transitions.
//
// ─────────────────────────────────────────────────────────────────────────────
// DESIGN FOR FUTURE OPTIMISATIONS
// ─────────────────────────────────────────────────────────────────────────────
//
//  1. SIMD neighbour counting (NEON / SVE2):
//       count_adults() is the sole hotspot.  A SIMD replacement operates on
//       the adult plane one word (64 cells) at a time, performing horizontal
//       sliding-window sums via shifts and additions, then vertical sums
//       across the five contributing rows, yielding a 6-bit saturating count
//       per cell.
//
//  2. Multi-threading:
//       step_row() reads only from src (any row) and writes only to row y of
//       dst.  Rows have no write-side dependencies on each other, so a simple
//       parallel_for over rows (std::thread pool / std::execution) requires
//       no synchronisation beyond the step boundary.
//
//  3. Cache-friendly tiling:
//       The five src rows needed by step_row(y) are rows y-2 … y+2.
//       Processing rows in small tiles keeps those rows in L1/L2 cache.
//
// ─────────────────────────────────────────────────────────────────────────────

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

static constexpr int NEIGHBOUR_RANGE = 2;   // range-2 Moore neighbourhood
static constexpr int NUM_THREADS     = 8;   // one per physical core on c8g.2xlarge

// ─────────────────────────────────────────────────────────────────────────────
// BitGrid
// ─────────────────────────────────────────────────────────────────────────────

struct BitGrid {
    int size;           // width == height; always a power of two, ≥ 64
    int words_per_row;  // size / 64

    // Bit-planes: index  y * words_per_row + (x >> 6),  bit  (x & 63)
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

    // ── bit-level accessors ───────────────────────────────────────────────────

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

    // ── row-level pack/unpack (used for I/O) ─────────────────────────────────
    //
    // These process 64 cells per inner loop iteration, making the conversion
    // between the on-disk byte-per-cell format and the internal bit-plane
    // format as cheap as possible.

    // Convert one row of byte-per-cell data → bit-planes.
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
                    default: break;   // 0 = EMPTY, already zero
                }
            }
            egg  [base + w] = e;
            juv  [base + w] = j;
            adult[base + w] = a;
        }
    }

    // Convert one row of bit-planes → byte-per-cell data.
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

    // ── bulk helpers ─────────────────────────────────────────────────────────

    void clear() noexcept
    {
        std::fill(egg.begin(),   egg.end(),   uint64_t{0});
        std::fill(juv.begin(),   juv.end(),   uint64_t{0});
        std::fill(adult.begin(), adult.end(), uint64_t{0});
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Bit-parallel neighbour counting and state transition
// ─────────────────────────────────────────────────────────────────────────────
//
// Each uint64_t word in a bit-plane holds 64 cells.  We compute the adult
// neighbour count for all 64 cells in a word simultaneously using only
// bitwise operations — no per-cell loop.
//
// The computation has three phases, each operating on 64 cells at once:
//
//  Phase 1 — Horizontal sum (per row, per word)
//    For each of the 5 contributing rows, shift the adult word left and right
//    by 1 and 2 bit positions (carrying bits across word boundaries from the
//    adjacent words), then sum the 5 shifted copies with a CSA tree.
//    Result: a 3-bit count at every cell position — "how many of the 5
//    cells in this row's horizontal band are adult?"
//
//  Phase 2 — Vertical sum (across 5 rows)
//    Sum the five 3-bit horizontal counts.  The result is the adult count
//    over the full 5×5 box (0–25), encoded as a 5-bit value.  Subtract the
//    centre cell to get the true neighbourhood count (0–24).
//
//  Phase 3 — Transitions
//    Express "3 ≤ A ≤ 5" and "4 ≤ A ≤ 9" as bitwise formulas on the 5-bit
//    count planes.  Three word-level stores write all 64 output cells.
//
// The same logic runs on plain uint64_t today.  On ARM NEON, replace
// uint64_t with uint64x2_t and the operators with their vXXXq_u64
// intrinsics — the algorithm is identical, but 128 cells per instruction.

// ─────────────────────────────────────────────────────────────────────────────
// Bits3 — a 3-bit integer at each of 64 independent bit positions
// ─────────────────────────────────────────────────────────────────────────────
//
// bit position b holds the integer  (b2[b] << 2) | (b1[b] << 1) | b0[b].
// Range [0, 7]; in practice horizontal sums are always in [0, 5].

struct Bits3 {
    uint64_t b0 = 0;   // 1s place
    uint64_t b1 = 0;   // 2s place
    uint64_t b2 = 0;   // 4s place
};

// ─────────────────────────────────────────────────────────────────────────────
// sum5 — add five 1-bit values per position via a carry-save adder (CSA) tree
// ─────────────────────────────────────────────────────────────────────────────
//
// A CSA on three 1-bit inputs a, b, c:
//   sum   = a ^ b ^ c              (1s bit of a+b+c, no carry chain)
//   carry = (a & b) | (b & c) | (a & c)  (2s bit — majority of the three)
//
// Applied to 64-bit words one instruction handles 64 independent CSAs.
// Two CSA stages reduce five inputs to a sum word and a carry word;
// a half-adder on the two carries gives the final 3-bit result.

static Bits3 sum5(uint64_t v0, uint64_t v1, uint64_t v2,
                  uint64_t v3, uint64_t v4) noexcept
{
    // Stage 1: fold v0, v1, v2  →  s1 (1s bit),  c1 (2s bit)
    const uint64_t s1 = v0 ^ v1 ^ v2;
    const uint64_t c1 = (v0 & v1) | (v1 & v2) | (v0 & v2);

    // Stage 2: fold v3, v4, s1  →  s2 (1s bit),  c2 (2s bit)
    const uint64_t s2 = v3 ^ v4 ^ s1;
    const uint64_t c2 = (v3 & v4) | (v4 & s1) | (v3 & s1);

    // Stage 3: half-adder on the two carry words  →  2s bit,  4s bit
    return { s2,  c1 ^ c2,  c1 & c2 };
}

// ─────────────────────────────────────────────────────────────────────────────
// hsum_word — 5-wide horizontal adult sum for one word of one row
// ─────────────────────────────────────────────────────────────────────────────
//
// For every bit position b in word w:
//   result[b]  =  Σ adult_row[b + dx]   for dx ∈ {−2,−1, 0,+1,+2}  (toroidal)
//
// The five offsets span ±2 bits beyond the word boundary, so up to 2 bits
// are borrowed from the left neighbour word (L) and right neighbour word (R).

static Bits3 hsum_word(const uint64_t* adult_row, int wpr, int w) noexcept
{
    const uint64_t L = adult_row[(w - 1 + wpr) & (wpr - 1)];   // left word
    const uint64_t C = adult_row[w];                             // centre word
    const uint64_t R = adult_row[(w + 1)        & (wpr - 1)];   // right word

    // Five column slices — bits crossing word edges come from L or R.
    //
    // To place the cell at offset dx into bit position b we shift C in the
    // direction that moves bit (b+dx) to bit b:
    //   dx < 0  →  shift C LEFT   by |dx| (bit b−2 rises to b); borrow top
    //              bits from L by shifting L RIGHT (L's bit 62 or 63 → bit 0 or 1).
    //   dx > 0  →  shift C RIGHT  by  dx  (bit b+2 falls to b); fill bottom
    //              bits from R by shifting R LEFT  (R's bit 0 or 1 → bit 62 or 63).
    const uint64_t col_m2 = (C << 2) | (L >> 62);   // cells at x − 2
    const uint64_t col_m1 = (C << 1) | (L >> 63);   // cells at x − 1
    const uint64_t col_0  =  C;                      // cells at x
    const uint64_t col_p1 = (C >> 1) | (R << 63);   // cells at x + 1
    const uint64_t col_p2 = (C >> 2) | (R << 62);   // cells at x + 2

    return sum5(col_m2, col_m1, col_0, col_p1, col_p2);
}

// ─────────────────────────────────────────────────────────────────────────────
// step_word — advance one word (64 cells) by one generation
// ─────────────────────────────────────────────────────────────────────────────
//
// Reads from src (any rows/words).  Writes exactly the one word at (w, y) in
// dst.  dst must be fully cleared before calling step_word for a generation.
// Every word writes to a disjoint location, so words can be processed in any
// order or in parallel with no synchronisation.

static void step_word(const BitGrid& src, BitGrid& dst, int w, int y) noexcept
{
    const int    N   = src.size;
    const int    wpr = src.words_per_row;
    const size_t wi  = static_cast<size_t>(y) * wpr + w;   // word index

    // ── Phase 1: horizontal sum for each of the 5 contributing rows ──────────
    //
    // hsum[k] = 3-bit per-position count of adults across columns x−2…x+2
    //           in row  y + (k−2),  i.e. k ∈ {0,1,2,3,4} maps to dy ∈ {−2…+2}.

    Bits3 hsum[2 * NEIGHBOUR_RANGE + 1];
    for (int dy = -NEIGHBOUR_RANGE; dy <= NEIGHBOUR_RANGE; ++dy) {
        const int ny = (y + dy + N) & (N - 1);
        hsum[dy + NEIGHBOUR_RANGE] =
            hsum_word(src.adult.data() + static_cast<size_t>(ny) * wpr, wpr, w);
    }

    // ── Phase 2: vertical sum across the 5 rows ───────────────────────────────
    //
    // Sum the five 3-bit hsum values at each bit position to get the total
    // adult count over the full 5×5 box (including the centre cell), stored
    // as a 5-bit value.
    //
    // Each bit plane (b0, b1, b2) is summed independently over the 5 rows
    // using sum5, giving three column sums:
    //
    //   col0  =  Σ hsum[k].b0  — weighted 1  (1s place of hsum values)
    //   col1  =  Σ hsum[k].b1  — weighted 2  (2s place of hsum values)
    //   col2  =  Σ hsum[k].b2  — weighted 4  (4s place of hsum values)
    //
    // The final total is  col0 * 1  +  col1 * 2  +  col2 * 4.
    // We combine them with a ripple-carry adder using the column structure:
    //
    //   output bit 0 :  col0.b0
    //   output bit 1 :  col0.b1  col1.b0
    //   output bit 2 :  col0.b2  col1.b1  col2.b0
    //   output bit 3 :           col1.b2  col2.b1
    //   output bit 4 :                    col2.b2
    //
    // Each column has at most 3 addends (+ carry-in), so a full-adder suffices
    // at the widest point.  Max result = 25 (all 25 box cells adult) < 32.

    const Bits3 col0 = sum5(hsum[0].b0, hsum[1].b0, hsum[2].b0,
                            hsum[3].b0, hsum[4].b0);   // weight 1
    const Bits3 col1 = sum5(hsum[0].b1, hsum[1].b1, hsum[2].b1,
                            hsum[3].b1, hsum[4].b1);   // weight 2
    const Bits3 col2 = sum5(hsum[0].b2, hsum[1].b2, hsum[2].b2,
                            hsum[3].b2, hsum[4].b2);   // weight 4

    // Ripple-carry adder, column by column.
    // HA = half-adder: sum = a^b, carry = a&b
    // FA = full-adder: sum = a^b^c, carry = (a&b)|(b&c)|(a&c)

    const uint64_t n0 = col0.b0;                              // bit 0: single input

    const uint64_t n1   = col0.b1 ^ col1.b0;                 // bit 1: HA
    const uint64_t cy1  = col0.b1 & col1.b0;

    const uint64_t t2   = col0.b2 ^ col1.b1 ^ col2.b0;       // bit 2: FA …
    const uint64_t cy2a = (col0.b2 & col1.b1) | (col1.b1 & col2.b0) | (col0.b2 & col2.b0);
    const uint64_t n2   = t2 ^ cy1;                           //   … + HA with cy1
    const uint64_t cy2b = t2 & cy1;

    const uint64_t t3   = col1.b2 ^ col2.b1 ^ cy2a;          // bit 3: FA …
    const uint64_t cy3a = (col1.b2 & col2.b1) | (col2.b1 & cy2a) | (col1.b2 & cy2a);
    const uint64_t n3   = t3 ^ cy2b;                          //   … + HA with cy2b
    const uint64_t cy3b = t3 & cy2b;

    const uint64_t n4   = col2.b2 ^ cy3a ^ cy3b;             // bit 4: FA (no carry out)

    // Subtract the centre cell to exclude it from the neighbourhood count.
    //
    // We decrement the 5-bit value {n4..n0} by 1 wherever the centre cell is
    // adult.  Binary decrement propagates a borrow through contiguous low zeros:
    //   e.g.  ...0100 − 1  =  ...0011  (two zeros flip before hitting the 1).
    //
    // The borrow at each bit: borrow_k = 1 iff centre=1 and bits k−1..0 are all 0.

    const uint64_t centre = src.adult[wi];

    uint64_t a0 = n0, a1 = n1, a2 = n2, a3 = n3, a4 = n4;
    {
        const uint64_t br0 = centre & ~a0;   a0 ^= centre;
        const uint64_t br1 = br0    & ~a1;   a1 ^= br0;
        const uint64_t br2 = br1    & ~a2;   a2 ^= br1;
        const uint64_t br3 = br2    & ~a3;   a3 ^= br2;
                                              a4 ^= br3;
        // br4 (borrow out of bit 4) is always 0:
        // the centre cell was counted in the box sum, so box_sum ≥ 1 when centre=1.
    }

    // ── Phase 3: bit-parallel transition rules ────────────────────────────────
    //
    // {a4..a0} encodes the adult neighbour count A at each of the 64 positions.
    //
    // Condition derivations (Karnaugh-map reduction of the 5-bit ranges):
    //
    //   3 ≤ A ≤ 5  ≡  !a4 & !a3 & !(a2 & a1) & (a2 | (a1 & a0))
    //
    //     A   binary   formula
    //     0   00000    0  (a2|(a1&a0)) = 0
    //     1   00001    0
    //     2   00010    0
    //     3   00011    1  ✓
    //     4   00100    1  ✓
    //     5   00101    1  ✓
    //     6   00110    0  (a2&a1) = 1 → blocked
    //     7   00111    0
    //     8+  ≥01000   0  (a3 or a4 set)
    //
    //   4 ≤ A ≤ 9  ≡  (a3 | a2) & !a4 & !(a3 & (a2 | a1))
    //
    //     A   binary   formula
    //     0–3 0000x    0  (a3|a2) = 0
    //     4   00100    1  ✓
    //     …
    //     9   01001    1  ✓
    //     10  01010    0  (a3 & a1) blocks
    //     11  01011    0
    //     12  01100    0  (a3 & a2) blocks
    //     16+ 1xxxx    0  (a4 set)

    const uint64_t cond_3_to_5 = ~a4 & ~a3 & ~(a2 & a1) & (a2 | (a1 & a0));
    const uint64_t cond_4_to_9 = (a3 | a2) & ~a4 & ~(a3 & (a2 | a1));

    const uint64_t cur_egg  = src.egg  [wi];
    const uint64_t cur_juv  = src.juv  [wi];
    const uint64_t cur_adlt = src.adult[wi];
    const uint64_t cur_empt = ~(cur_egg | cur_juv | cur_adlt);

    // Three stores, all 64 output cells written simultaneously.
    dst.egg  [wi] = cur_empt & cond_3_to_5;              // EMPTY → EGG
    dst.juv  [wi] = cur_egg;                              // EGG   → JUVENILE (always)
    dst.adult[wi] = cur_juv | (cur_adlt & cond_4_to_9);  // JUV   → ADULT    (always)
                                                          // ADULT → ADULT if 4 ≤ A ≤ 9
}

// ─────────────────────────────────────────────────────────────────────────────
// step — advance the entire grid by one generation
// ─────────────────────────────────────────────────────────────────────────────
//
// The grid is split into NUM_THREADS contiguous row bands.  Each thread owns
// its band exclusively: step_word reads only from src (shared, read-only
// during a step) and writes only to its assigned rows in dst.  There is no
// shared mutable state, so no locks or atomics are needed.
//
// Row bands are contiguous so each thread's working set (5 src rows + 1 dst
// row per step_word call) stays in L1/L2 cache as the inner w-loop runs.
//
// N is always a power of two ≥ 512, and NUM_THREADS = 8 = 2³, so
// N / NUM_THREADS is always exact (≥ 64 rows per thread).

static void step(const BitGrid& src, BitGrid& dst) noexcept
{
    dst.clear();

    const int N              = src.size;
    const int wpr            = src.words_per_row;
    const int rows_per_thread = N / NUM_THREADS;

    // Worker: process rows [y_start, y_start + rows_per_thread).
    auto worker = [&](int y_start) noexcept {
        const int y_end = y_start + rows_per_thread;
        for (int y = y_start; y < y_end; ++y)
            for (int w = 0; w < wpr; ++w)
                step_word(src, dst, w, y);
    };

    std::thread threads[NUM_THREADS];
    for (int t = 0; t < NUM_THREADS; ++t)
        threads[t] = std::thread(worker, t * rows_per_thread);

    for (std::thread& t : threads)
        t.join();
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
    // ── argument parsing ──────────────────────────────────────────────────────

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

    // ── read input ────────────────────────────────────────────────────────────

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

    // The bit-plane representation requires the row length to be a multiple of
    // 64.  All valid inputs (power-of-two sizes ≥ 64) satisfy this.
    if (width % 64 != 0) {
        std::fprintf(stderr,
            "Error: grid width must be a multiple of 64, got %" PRIu64 "\n", width);
        std::fclose(fin);
        return 3;
    }

    const int    grid_size  = static_cast<int>(width);
    const size_t cell_count = static_cast<size_t>(grid_size) * grid_size;

    // Read raw byte-per-cell data into a temporary buffer.
    std::vector<uint8_t> raw(cell_count);
    if (std::fread(raw.data(), 1, cell_count, fin) != cell_count) {
        std::fprintf(stderr, "Error: input file too short (cell data truncated)\n");
        std::fclose(fin);
        return 4;
    }
    std::fclose(fin);

    // Pack byte-per-cell data into the three-plane bit representation.
    BitGrid grid_a(grid_size);
    for (int y = 0; y < grid_size; ++y)
        grid_a.pack_row(y, raw.data() + static_cast<size_t>(y) * grid_size);

    raw.clear();
    raw.shrink_to_fit();    // release I/O buffer before simulation starts

    // ── simulate ──────────────────────────────────────────────────────────────

    BitGrid  grid_b(grid_size);
    BitGrid* cur = &grid_a;
    BitGrid* nxt = &grid_b;

    const auto t0 = std::chrono::steady_clock::now();

    for (int gen = 0; gen < generations; ++gen) {
        step(*cur, *nxt);
        std::swap(cur, nxt);
    }

    const auto   t1          = std::chrono::steady_clock::now();
    const double elapsed_ms  =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("%.3f ms\n", elapsed_ms);

    // ── write output ──────────────────────────────────────────────────────────

    FILE* fout = std::fopen(argv[2], "wb");
    if (!fout) {
        std::fprintf(stderr, "Error: cannot open output file '%s'\n", argv[2]);
        return 5;
    }

    // Unpack bit-planes back to byte-per-cell format for the on-disk output.
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
