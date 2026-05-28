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

// ─────────────────────────────────────────────────────────────────────────────
// NEON helpers — explicit ternary logic so we get BSL for the CSA majorities.
// ─────────────────────────────────────────────────────────────────────────────
//
// GCC's auto-vectorizer expands the carry majority maj(a,b,c) to the 4-op form
//   (a&b) | (c & (a^b)).
// ARM NEON has a single-instruction bit-select that gives the majority in one
// op once the sum's a^b is in hand:
//   maj(a,b,c) = bsl(a^b, c, a) = ((a^b)&c) | (~(a^b)&a)        (== majority)
// We hand-vectorise the per-word kernels (2 words / 128 cells per NEON vector)
// to exploit this; they are the dominant cost.
typedef uint64x2_t u64v;
static inline u64v v_xor (u64v a, u64v b) noexcept { return veorq_u64(a, b); }
static inline u64v v_and (u64v a, u64v b) noexcept { return vandq_u64(a, b); }
static inline u64v v_orr (u64v a, u64v b) noexcept { return vorrq_u64(a, b); }
static inline u64v v_bic (u64v a, u64v b) noexcept { return vbicq_u64(a, b); }  // a & ~b
static inline u64v v_xor3(u64v a, u64v b, u64v c) noexcept {
#ifdef __ARM_FEATURE_SHA3
    return veor3q_u64(a, b, c);
#else
    return veorq_u64(veorq_u64(a, b), c);
#endif
}
// majority(a,b,c) in a single BSL once a^b is computed by the sum.
static inline u64v v_maj(u64v a, u64v b, u64v c) noexcept {
    return vbslq_u64(veorq_u64(a, b), c, a);
}

// NEON sum5: add five 1-bit lanes per position → 3-bit result {b0,b1,b2}.
// One CSA = one EOR3 (sum) + one BSL (majority carry).
static inline void v_sum5(u64v v0, u64v v1, u64v v2, u64v v3, u64v v4,
                          u64v& b0, u64v& b1, u64v& b2) noexcept {
    const u64v s1 = v_xor3(v0, v1, v2), c1 = v_maj(v0, v1, v2);
    const u64v s2 = v_xor3(v3, v4, s1), c2 = v_maj(v3, v4, s1);
    b0 = s2; b1 = v_xor(c1, c2); b2 = v_and(c1, c2);
}

// ─────────────────────────────────────────────────────────────────────────────
// compute_hrow — horizontal 5-wide adult sum for an ENTIRE row
// ─────────────────────────────────────────────────────────────────────────────
//
// Phase 1, but computed once per row instead of once per (row, output-row) pair.
// hrow[w] holds the 3-bit per-cell count of adults over columns x−2…x+2 for
// every cell in word w of the given adult row.  The separable 5×5 box sum reuses
// each of these across the 5 output rows that need it (see the worker loop).
// SoA output: three separate bit-planes (b0/b1/b2) for unit-stride, auto-
// vectorizable access in emit_row.
//
// The toroidal wrap only affects the first and last words of the row (they read
// the opposite-edge word).  We peel those two off as scalar special cases; the
// interior loop reads neighbours at fixed offsets w-1 / w+1 (plain misaligned
// loads, no data-dependent index), so the compiler auto-vectorizes it to SVE2 —
// the per-lane bit shifts handle the cross-word carry directly.
static void compute_hrow(const uint64_t* __restrict A, int wpr,
                         uint64_t* __restrict ob0,
                         uint64_t* __restrict ob1,
                         uint64_t* __restrict ob2) noexcept
{
    if (wpr <= 2) {                       // tiny grids: edge wrap is degenerate
        for (int w = 0; w < wpr; ++w) {
            const Bits3 h = hsum_word(A, wpr, w);
            ob0[w] = h.b0; ob1[w] = h.b1; ob2[w] = h.b2;
        }
        return;
    }

    // Boundary words 0 and wpr-1 (toroidal wrap) handled individually.
    for (int w : { 0, wpr - 1 }) {
        const Bits3 h = hsum_word(A, wpr, w);
        ob0[w] = h.b0; ob1[w] = h.b1; ob2[w] = h.b2;
    }

    // Interior words [1, wpr-1): no wrap.  The left/right neighbour words are
    // just unaligned loads at offset w-1 / w+1, so a NEON pair load gives the
    // cross-word carry bits with no lane shuffling.  bsl-based v_sum5 keeps the
    // CSA cheap.  (wpr is a power of two ≥ 4 here, so the interior word count
    // wpr-2 is even and the pair loop covers it exactly; the tail handles the
    // unreachable odd case defensively.)
    int w = 1;
    for (; w + 2 <= wpr - 1; w += 2) {
        const u64v C = vld1q_u64(A + w);        // [A[w],   A[w+1]]
        const u64v L = vld1q_u64(A + w - 1);    // [A[w-1], A[w]  ]
        const u64v R = vld1q_u64(A + w + 1);    // [A[w+1], A[w+2]]
        const u64v col_m2 = v_orr(vshlq_n_u64(C, 2), vshrq_n_u64(L, 62));
        const u64v col_m1 = v_orr(vshlq_n_u64(C, 1), vshrq_n_u64(L, 63));
        const u64v col_0  = C;
        const u64v col_p1 = v_orr(vshrq_n_u64(C, 1), vshlq_n_u64(R, 63));
        const u64v col_p2 = v_orr(vshrq_n_u64(C, 2), vshlq_n_u64(R, 62));
        u64v b0, b1, b2;
        v_sum5(col_m2, col_m1, col_0, col_p1, col_p2, b0, b1, b2);
        vst1q_u64(ob0 + w, b0); vst1q_u64(ob1 + w, b1); vst1q_u64(ob2 + w, b2);
    }
    for (; w < wpr - 1; ++w) {                  // defensive scalar tail
        const Bits3 h = hsum_word(A, wpr, w);
        ob0[w] = h.b0; ob1[w] = h.b1; ob2[w] = h.b2;
    }
}

// One NEON lane-group (2 words) of the vertical-sum-5-of-3-bit + transition.
// Inputs are the 5 horizontal-sum vectors per bit-plane and the centre state.
static inline void emit_vec(u64v h0b0,u64v h1b0,u64v h2b0,u64v h3b0,u64v h4b0,
                            u64v h0b1,u64v h1b1,u64v h2b1,u64v h3b1,u64v h4b1,
                            u64v h0b2,u64v h1b2,u64v h2b2,u64v h3b2,u64v h4b2,
                            u64v ve, u64v vj, u64v va,
                            u64v& oe, u64v& oj, u64v& oa) noexcept
{
    // sum5 per bit-plane → col{0,1,2} (each a 3-bit value: s/c/cc)
    u64v c0s,c0c,c0cc, c1s,c1c,c1cc, c2s,c2c,c2cc;
    v_sum5(h0b0,h1b0,h2b0,h3b0,h4b0, c0s,c0c,c0cc);   // weight 1
    v_sum5(h0b1,h1b1,h2b1,h3b1,h4b1, c1s,c1c,c1cc);   // weight 2
    v_sum5(h0b2,h1b2,h2b2,h3b2,h4b2, c2s,c2c,c2cc);   // weight 4

    // Ripple combine → 5-bit box sum {m4..m0}.
    const u64v m0 = c0s;
    const u64v m1 = v_xor(c0c, c1s),  cy1 = v_and(c0c, c1s);
    const u64v t2 = v_xor3(c0cc, c1c, c2s), cy2a = v_maj(c0cc, c1c, c2s);
    const u64v m2 = v_xor(t2, cy1),   cy2b = v_and(t2, cy1);
    const u64v t3 = v_xor3(c1cc, c2c, cy2a), cy3a = v_maj(c1cc, c2c, cy2a);
    const u64v m3 = v_xor(t3, cy2b),  cy3b = v_and(t3, cy2b);
    const u64v m4 = v_xor3(c2cc, cy3a, cy3b);

    // Transition tests on the box sum (centre folded in, see scalar emit_bits):
    //   birth   = ~m4 & ~m3 & ((~m2 & m1 & m0) | (m2 & ~m1))
    //   survive = ~m4 & ((~m3 & m2 & (m1|m0)) | (m3 & ~m2 & ~(m1&m0)))
    const u64v birth =
        v_bic(v_bic(v_orr(v_and(v_bic(m1, m2), m0), v_bic(m2, m1)), m3), m4);
    const u64v survive =
        v_bic(v_orr(v_and(v_bic(m2, m3), v_orr(m1, m0)),
                    v_bic(v_bic(m3, m2), v_and(m1, m0))), m4);

    const u64v occupied = v_orr(v_orr(ve, vj), va);   // ~empty
    oe = v_bic(birth, occupied);          // EMPTY → EGG  (birth & ~occupied)
    oj = ve;                              // EGG   → JUVENILE
    oa = v_orr(vj, v_and(va, survive));   // JUV→ADULT; ADULT survives if boxsum∈[5,10]
}

// emit_row — vertical sum + transition for an ENTIRE row.  NEON fast path
// (2 words / iteration) with a scalar tail for an odd last word (tiny grids).
static void emit_row(int wpr,
        const uint64_t* __restrict p0b0, const uint64_t* __restrict p0b1, const uint64_t* __restrict p0b2,
        const uint64_t* __restrict p1b0, const uint64_t* __restrict p1b1, const uint64_t* __restrict p1b2,
        const uint64_t* __restrict p2b0, const uint64_t* __restrict p2b1, const uint64_t* __restrict p2b2,
        const uint64_t* __restrict p3b0, const uint64_t* __restrict p3b1, const uint64_t* __restrict p3b2,
        const uint64_t* __restrict p4b0, const uint64_t* __restrict p4b1, const uint64_t* __restrict p4b2,
        const uint64_t* __restrict se, const uint64_t* __restrict sj, const uint64_t* __restrict sa,
        uint64_t* __restrict de, uint64_t* __restrict dj, uint64_t* __restrict da) noexcept
{
    int w = 0;
    for (; w + 2 <= wpr; w += 2) {
        u64v oe, oj, oa;
        emit_vec(vld1q_u64(p0b0+w), vld1q_u64(p1b0+w), vld1q_u64(p2b0+w), vld1q_u64(p3b0+w), vld1q_u64(p4b0+w),
                 vld1q_u64(p0b1+w), vld1q_u64(p1b1+w), vld1q_u64(p2b1+w), vld1q_u64(p3b1+w), vld1q_u64(p4b1+w),
                 vld1q_u64(p0b2+w), vld1q_u64(p1b2+w), vld1q_u64(p2b2+w), vld1q_u64(p3b2+w), vld1q_u64(p4b2+w),
                 vld1q_u64(se+w), vld1q_u64(sj+w), vld1q_u64(sa+w), oe, oj, oa);
        vst1q_u64(de+w, oe); vst1q_u64(dj+w, oj); vst1q_u64(da+w, oa);
    }
    // Scalar tail (only when wpr is odd, i.e. a 64-wide grid).
    for (; w < wpr; ++w) {
        const Bits3 col0 = sum5(p0b0[w], p1b0[w], p2b0[w], p3b0[w], p4b0[w]);
        const Bits3 col1 = sum5(p0b1[w], p1b1[w], p2b1[w], p3b1[w], p4b1[w]);
        const Bits3 col2 = sum5(p0b2[w], p1b2[w], p2b2[w], p3b2[w], p4b2[w]);
        const uint64_t m0 = col0.b0;
        const uint64_t m1 = col0.b1 ^ col1.b0, cy1 = col0.b1 & col1.b0;
        const uint64_t t2 = col0.b2 ^ col1.b1 ^ col2.b0;
        const uint64_t cy2a = (col0.b2 & col1.b1)|(col1.b1 & col2.b0)|(col0.b2 & col2.b0);
        const uint64_t m2 = t2 ^ cy1, cy2b = t2 & cy1;
        const uint64_t t3 = col1.b2 ^ col2.b1 ^ cy2a;
        const uint64_t cy3a = (col1.b2 & col2.b1)|(col2.b1 & cy2a)|(col1.b2 & cy2a);
        const uint64_t m3 = t3 ^ cy2b, cy3b = t3 & cy2b;
        const uint64_t m4 = col2.b2 ^ cy3a ^ cy3b;
        const uint64_t birth   = ~m4 & ~m3 & ((~m2 & m1 & m0) | (m2 & ~m1));
        const uint64_t survive = ~m4 & ((~m3 & m2 & (m1|m0)) | (m3 & ~m2 & ~(m1&m0)));
        const uint64_t occupied = se[w] | sj[w] | sa[w];
        de[w] = birth & ~occupied;
        dj[w] = se[w];
        da[w] = sj[w] | (sa[w] & survive);
    }
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

// run_simulation — advance `generations` steps using a PERSISTENT thread pool.
//
// The previous design spawned and joined NUM_THREADS std::threads inside every
// generation.  Here the threads are created once; each worker owns a fixed band
// of rows for the whole run and synchronises at a std::barrier between
// generations.  The barrier's completion function (run by exactly one thread
// while all others are parked) swaps the cur/nxt grid pointers — no clear needed
// because step_word fully overwrites every output word.
//
// Returns a pointer to the grid holding the final generation.
static BitGrid* run_simulation(BitGrid& grid_a, BitGrid& grid_b, int generations)
{
    BitGrid* cur = &grid_a;
    BitGrid* nxt = &grid_b;

    const int N               = grid_a.size;
    const int wpr             = grid_a.words_per_row;
    const int rows_per_thread = N / NUM_THREADS;

    // Completion phase: runs once per generation, after all workers arrive and
    // before any is released.  Swapping here is therefore race-free.
    auto on_done = [&]() noexcept { std::swap(cur, nxt); };
    std::barrier sync(NUM_THREADS, on_done);

    auto worker = [&](int y_start) noexcept {
        const int y_end = y_start + rows_per_thread;

        // Per-thread sliding window: 5 rows of horizontal sums, stored SoA in
        // three bit-planes.  The horizontal sum for logical row r lives in ring
        // slot (r mod 5).  Allocated once, reused across generations.
        std::vector<uint64_t> ringbuf(static_cast<size_t>(3) * 5 * wpr);
        uint64_t* hb[3][5];
        for (int p = 0; p < 3; ++p)
            for (int k = 0; k < 5; ++k)
                hb[p][k] = ringbuf.data() + (static_cast<size_t>(p) * 5 + k) * wpr;
        auto slot = [](int r) noexcept { return ((r % 5) + 5) % 5; };

        auto do_hrow = [&](const uint64_t* A, int r) noexcept {
            const int ar = (r + N) & (N - 1);
            const int k  = slot(r);
            compute_hrow(A + static_cast<size_t>(ar) * wpr, wpr, hb[0][k], hb[1][k], hb[2][k]);
        };

        for (int gen = 0; gen < generations; ++gen) {
            const BitGrid& s = *cur;
            BitGrid&       d = *nxt;
            const uint64_t* A = s.adult.data();

            // Prime the window with horizontal sums for rows y_start-2 … y_start+1.
            for (int r = y_start - 2; r <= y_start + 1; ++r)
                do_hrow(A, r);

            for (int y = y_start; y < y_end; ++y) {
                do_hrow(A, y + 2);   // new bottom row entering the window

                const int k0 = slot(y - 2), k1 = slot(y - 1), k2 = slot(y),
                          k3 = slot(y + 1), k4 = slot(y + 2);

                const size_t   base = static_cast<size_t>(y) * wpr;
                emit_row(wpr,
                    hb[0][k0], hb[1][k0], hb[2][k0],
                    hb[0][k1], hb[1][k1], hb[2][k1],
                    hb[0][k2], hb[1][k2], hb[2][k2],
                    hb[0][k3], hb[1][k3], hb[2][k3],
                    hb[0][k4], hb[1][k4], hb[2][k4],
                    s.egg.data() + base, s.juv.data() + base, s.adult.data() + base,
                    d.egg.data() + base, d.juv.data() + base, d.adult.data() + base);
            }
            sync.arrive_and_wait();   // completion swaps cur/nxt
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(NUM_THREADS);
    for (int t = 0; t < NUM_THREADS; ++t)
        threads.emplace_back(worker, t * rows_per_thread);
    for (std::thread& t : threads)
        t.join();

    return cur;   // after `generations` swaps, cur holds the final grid
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

    BitGrid grid_b(grid_size);

    const auto t0 = std::chrono::steady_clock::now();

    BitGrid* cur = run_simulation(grid_a, grid_b, generations);

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
