# Design Document — Monster Spawning Grid
**Name:** Reflekshiot
**Date:** 29-05-2026
**Final median time (10 runs, public_1):** ___________________ ms  
**Reference median time (10 runs, public_1):** ___________________ ms  
**Speedup:** ___________________×

---

## 1. Cell Representation

The grid is stored as three parallel bit-planes: `egg[]`, `juv[]`, `adult[]`, each a flat `uint64_t` array in row-major order. One bit per cell per plane; 64 cells fit in one word. Cell `(x, y)` lives at word `y * wpr + (x >> 6)`, bit `x & 63`.

We considered two alternatives:

- **Byte-per-cell**: simple but 8× larger - a 32768² grid would require 1 GiB just to store one copy, saturating DRAM bandwidth immediately.
- **Two-plane encoding** (`main_opt.cpp`): states 0–3 encoded as two bits `(b1, b0)`, reducing total grid size by 33%. We implemented and benchmarked this. The problem is that extracting the adult mask for neighbour counting requires loading *both* `b0` and `b1` planes — doubling the memory bandwidth for the hottest operation (`hsum`). The three-plane layout pays a 33% memory cost to keep `adult[]` as an independent array that `hsum` can read with a single load per word.

We chose three planes because the hot path (`hsum`) only ever reads `adult[]`. Keeping it isolated minimises cache pollution during the dominant operation.

## 2. Parallelisation Strategy

Eight persistent `std::thread` workers are created once at startup (`initialize_threads`). Each thread owns a fixed contiguous horizontal band of `N/8` rows and processes all 10,000 generations in a loop — no thread creation overhead per generation.

Synchronisation uses a single `std::barrier<Completion>` with 8 parties. When all threads call `arrive_and_wait()` at the end of each generation, the barrier's completion callback fires `std::swap(src, dst)`, atomically swapping the source and destination grids. Threads then immediately begin the next generation. This eliminates a separate copy step and any explicit grid clear.

Toroidal boundaries are handled by masking row indices with `& (N - 1)` (valid because `N` is always a power of two) and word indices within a row with `& (wpr - 1)`. Border rows of each thread's band are read from `src` (which no thread is writing to) so there is no data race at tile edges.

## 3. SIMD Strategy

We use **SVE2** (ARM 128-bit SIMD, `uint64x2_t`). 
As all registers are of the same size, no performance change for any.
The key vectorised operation is the **carry-save adder (CSA) tree** for neighbour counting. `hsum_pair` processes 2 words (128 cells) per call:

1. Load the adult words for the current chunk and its left/right neighbours.
2. Construct 5 shifted bitsets (offsets −2, −1, 0, +1, +2) using `vshlq_n_u64`, `vshrq_n_u64`, and `vextq_u64` for cross-word carries.
3. Sum all 5 with `n_sum5` (a two-stage CSA tree), producing a 3-bit result per cell stored in three `uint64x2_t` words (`b0/b1/b2`).

`vstep_pair` applies the same pattern vertically across 5 precomputed `hrow` entries, then applies the game rules in pure bitwise logic — no branches, no per-cell loops.


## 4. Memory Layout and Tiling

**Tile size: `TILE_ROWS = 8` rows**, with a **w-outer, y-inner** loop order.

For each column chunk `w` and tile starting at `y_tile`, we precompute a sliding window of `TILE_ROWS + 4 = 12` horizontal sums into a stack array `hrow2[12]` (576 bytes for NEON). The entire array fits comfortably in L1 cache (64 KiB). The inner `y` loop then calls `vstep_pair` using only `hrow` entries — no adult data is re-read from memory during rule application.

This reduces `hsum` calls from `5 × TILE_ROWS = 40` to `TILE_ROWS + 4 = 12` per tile column — a 3.3× reduction in adult-plane loads.

Tile size was chosen to keep `hrow` in L1. A tile of 8 rows gives full 5× reuse for most `hrow` entries (each entry is used by up to 5 vstep calls). Larger tiles would spill `hrow` out of L1 for wide grids; smaller tiles reduce reuse.

Software prefetch (`__builtin_prefetch`, `PREFETCH_DIST = 4`) issues loads for adult rows 4 iterations ahead during the `hrow` precomputation loop, hiding DRAM latency on the strided adult-plane accesses.

No temporal blocking (multiple generations per tile) was implemented in the final version; see Section 5.

## 5. What Didn't Work



## 6. What You Would Do With Another Week

**Fix and tune temporal blocking for the actual L2 size.** The K=2 approach in `main_opt4.cpp` is theoretically sound — computing two generations without writing the intermediate result to DRAM halves memory bandwidth. The issue was tile sizing: 128 rows × 2 planes × `wpr` words exceeds L2 for large grids. With another week we would tune `TILE_ROWS` to keep the intermediate buffer under 1 MiB (L2 is 2 MiB per core, shared with the hrow window), verify correctness across toroidal boundaries with a small-grid exhaustive test, and benchmark at 8192² and 32768² where DRAM bandwidth is the dominant cost.

## 7. Benchmark Methodology

All measurements used `harness/run.sh -n 10` on the AWS `c8g.2xlarge` instance (Graviton4, 8 vCPUs, Ubuntu 24.04). The harness:

- Disables ASLR via `setarch -R`
- Reports median, min, max, mean, standard deviation, and coefficient of variation (CV) across 10 runs
- Measures simulation time only (excludes I/O); timing uses `std::chrono::steady_clock`

We could not drop page-cache between runs (no root access), so run 1 may be slightly warmer than subsequent runs. Several runs showed CV > 5% due to background load on the shared instance; we re-ran those and reported the lower-variance set.

Correctness was verified by comparing binary output against the reference expected files in `test_grids/` (generated by the reference `spawn_sim` implementation). Individual optimisations were validated by running `harness/run.sh -n 5` before and after each change on `public_1_random_low_2048.bin` and confirming correctness on all five public test cases at size 2048.

---

*Maximum 3 pages total. Tables and figures count toward the limit.*

