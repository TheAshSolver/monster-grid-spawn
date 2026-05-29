
## 1. Cell Representation

The grid is stored as **three parallel bit-planes** — `egg`, `juv`, `adult` — each a flat `uint64_t[]` array of size `N × (N/64)`, row-major. Cell `(x, y)` occupies bit `x & 63` of word `y * words_per_row + (x >> 6)`.

| State    | egg | juv | adult |
|----------|-----|-----|-------|
| EMPTY    |  0  |  0  |   0   |
| EGG      |  1  |  0  |   0   |
| JUVENILE |  0  |  1  |   0   |
| ADULT    |  0  |  0  |   1   |

**Why three planes over one byte per cell:** A single byte-per-cell array uses 16–64× more memory (depending on grid size), destroying L1/L2 cache residency. The bit-plane layout lets one `uint64_t` operation touch 64 cells simultaneously, turning per-cell logic into bitwise arithmetic on 64- or 128-bit words.

**Why three planes over two planes:** The two-plane (b0/b1) encoding (`main_opt4`) packs the same data into `N × (N/32)` words and avoids a separate adult plane, but it requires a `b0 & b1` step before every SIMD hsum call and makes the transition formulas harder to express cleanly. The three-plane layout keeps the hot `adult` plane independent, so `compute_hrow` reads only one plane per row, and the transition stores three disjoint planes with no masking overhead.

---

## 2. Parallelisation Strategy

**Persistent thread pool with `std::barrier`.** Eight threads are created once at simulation start and never destroyed until the run completes. Each thread owns a fixed contiguous band of `N/8` rows for the entire run.

Within each generation, threads work on their row bands independently — `compute_hrow` reads from the shared read-only source grid (any row), and `emit_row` writes only to the thread's own rows in the destination grid. No locks or atomics are needed during the compute phase.

After each generation the threads rendezvous at `std::barrier::arrive_and_wait()`. The barrier's completion function (executed by exactly one thread while the rest are parked) swaps the `cur`/`nxt` grid pointers — a single pointer swap with no memclear, since `emit_row` fully overwrites every output word.

**Toroidal wrap** at tile boundaries is handled implicitly: `hsum_word` uses `(w ± 1 + wpr) & (wpr - 1)` for the two boundary words (first and last per row). The interior NEON loop uses plain misaligned loads at offsets `w-1` and `w+1` with no modular arithmetic, since no wrap is possible there.

---

## 3. SIMD Strategy

**NEON (ARM, 128-bit, `uint64x2_t`), hand-vectorised.** Each NEON register holds two `uint64_t` words = 128 cells. Every bitwise op processes 128 cells per instruction.

**Why hand-vectorised rather than auto-vectorised:** The CSA carry (majority function) is the critical bottleneck. GCC's auto-vectoriser generates the 4-operation form `(a&b)|(c&(a^b))`. By hand-writing the sum as EOR3 + BSL we reduce each CSA stage to 2 instructions:

```
sum   = veor3q_u64(a, b, c)          // EOR3:  a ^ b ^ c  (1 op, SHA3 extension)
carry = vbslq_u64(veorq_u64(a,b), c, a)  // BSL: majority in 1 op, reusing sum's a^b
```

`veor3q_u64` is available on Neoverse V1/V2 (Graviton3) via `__ARM_FEATURE_SHA3`; on older NEON it falls back to two `veorq_u64` calls.

**SRI / SLI for horizontal carry (new in v11):** The five column-offset shifts in `compute_hrow` previously used `vshlq_n_u64(C,k) | vshrq_n_u64(L, 64-k)` — two instructions per column. v11 replaces this with ARM's **Shift Right and Insert** / **Shift Left and Insert**:

```cpp
// v9:  col_m2 = vshlq_n_u64(C, 2) | vshrq_n_u64(L, 62)   (2 ops)
// v11: col_m2 = vsriq_n_u64(vshlq_n_u64(C, 2), L, 62)     (1 op)
```

`vsriq_n_u64(a, b, n)` shifts `b` right by `n` and inserts those bits into the low lanes of `a` **in-place**, with no separate OR. This saves 4 instructions per 128-cell NEON vector in the hrow loop.

**Algebraically reduced transition formulas (new in v11):** The birth/survive conditions were re-derived from the 5-bit box-sum K-maps:

```
birth   ∈ {3,4,5}: (m1^m2) & (m2|m0) & ~m3 & ~m4          (5 ops)
survive ∈ {5..10}: (m3^m2) & ((m1^m3)|(m0^m3)) & ~m4       (6 ops)
```

The survive XOR trick: when `m3=0`, `(m1^m3)|(m0^m3) = m1|m0`; when `m3=1`, `= ~m1|~m0 = ~(m1&m0)` — a BSL collapsed into two XORs and an OR, eliminating an explicit NOT. Together, birth + survive now take 11 ops vs the 14 ops in v9.

---

## 4. Memory Layout and Tiling

**Bit-planes, row-major, three separate `vector<uint64_t>` arrays.** Three independent allocations keep the hot `adult` plane on its own cache lines — `compute_hrow` accesses only `adult`, never touching `egg` or `juv` until the emit phase.

**Sliding 5-slot ring buffer (per thread).** Each thread allocates `3 × 5 × wpr` uint64_t words for the horizontal-sum cache (`hb[plane][slot]`). The horizontal sum for logical row `r` lives in slot `r % 5`. For a 4096-wide grid (`wpr = 64`), the ring buffer is `3 × 5 × 64 × 8 = 7,680 bytes` — well within the 64 KB L1 on Graviton3.

Each generation, the window is primed with rows `y_start-2` to `y_start+1` (4 hsum calls), then the inner loop adds one new row per output row and discards the oldest. Total hsum calls per generation per thread = `rows_per_thread + 4` instead of `5 × rows_per_thread` — a 5× reduction.

**No explicit cache tiling in v11.** The ring buffer naturally keeps the 5 live hsum rows resident in L1 because each new hrow overwrites the slot just vacated by the oldest — temporal reuse is implicit. There is no TILE_ROWS outer loop.

**No temporal blocking.** Each pass advances by one generation. The `main_opt4` path used K=2 temporal blocking (two generations per tile), but this required a separate tile buffer and more complex halo logic; on the 3-plane layout the added complexity did not pay off.

---

## 5. What Didn't Work



**SVE2 / tiered dispatch (`main_optimised` approach).** An SVE2 path (`svuint64_t`, 4 lanes on Graviton3) was implemented alongside NEON (2 lanes) and scalar (1 lane) fallbacks. Runtime dispatch via `#ifdef __ARM_FEATURE_SVE2` adds branch overhead on every hrow/emit call and prevented the compiler from fully inlining the critical inner loops. On the benchmark grid sizes the measured throughput was no better than the NEON path, and the code was substantially harder to maintain.

---

## 6. What You Would Do With Another Week

**Cache tiling + software prefetch in the ring-buffer worker.** The current ring buffer computes `hrow(y+2)` immediately before emitting row `y` — zero prefetch headroom. A tile-precompute structure (precompute all `TILE_ROWS+4` hrows up front, then emit) would let `__builtin_prefetch` run 4 rows ahead of each `compute_hrow` call, hiding the ~300-cycle DRAM latency on Graviton3. For `TILE_ROWS=8` the hrow buffer grows to `12 × 3 × 64 × 8 = 18 KB` — still within L1 — and the emit pass accesses only already-hot data. Early profiling with `perf stat` shows LLC misses concentrated in `compute_hrow`'s adult-plane reads, confirming this is the remaining bottleneck.

---

## 7. Benchmark Methodology

Timing is measured with `std::chrono::steady_clock` bracketing only `run_simulation()` — file I/O and grid allocation are excluded. The reported number is the **median of 10 consecutive runs** on the same process invocation to amortise OS scheduling noise.



Individual optimisations were isolated by changing one thing at a time (e.g., SRI vs OR-of-shifts) and re-running the full 10-run suite. Coefficient of variation was typically < 1% within a session; cross-session variance (after reboots) was < 3%.
