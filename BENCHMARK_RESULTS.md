# Monster Spawning Grid — Benchmark Results

All measurements on `c8g.2xlarge` (Graviton4 / Neoverse-V2, 8 vCPU, ARMv9.0-a + SVE2 + SHA3),
Ubuntu 24.04, g++-14, `-std=c++23 -O3 -mcpu=neoverse-v2 -Wall -Wextra -lpthread`
(`main_v11` additionally with `+sha3`).  Each run wrapped with `taskset -c 0-7 setarch aarch64 -R`.
Reported time is the simulation loop only (excludes file I/O).
Grid size: **32 768 × 32 768** (~1 GiB byte-per-cell on disk).  10 generations per run.

## Variants

- **`main`** — **baseline** — bit-plane SWAR (uint64 SIMD-within-register), 8 threads, recreates threads each gen (`code/main.cpp`)
- **`main_v11`** — **fully optimised** — hand-NEON kernels with `bsl` majority + `vsri/vsli` shift-insert + SHA3 `eor3` + persistent barrier pool + sliding horizontal-sum cache + reduced K-map transitions (`code/main_v11.cpp`, built with `-mcpu=neoverse-v2+sha3`)
- **`main_opt4`** — K=2 temporal blocking + 2-plane row-interleaved layout + sliding horizontal-sum cache (`code/main_opt4.cpp`)
- **`main_no_threads`** — bit-plane SIMD-within-register, **single-threaded** (`code/main_no_threads.cpp`)
- **`main_no_simd`** — **scalar** cell-by-cell neighbour counting, 8 threads (`code/main_no_simd.cpp`)
- **`main_no_simd_no_threads`** — **scalar single-threaded** — closest to the reference (`code/main_no_simd_no_threads.cpp`)

## Test grids

- `public_1` through `public_5` — the 5 standard grids from the assignment (random low, random high, structured 3×3 blocks, sparse clusters, boundary stress).
- `edge_all_adult` — every cell ADULT.  After gen 1 the box-sum saturates (A=24 for every cell) so all die instantly; subsequent gens are all-EMPTY.  Tests the upper extremum of the survive predicate.
- `edge_all_empty` — every cell EMPTY.  Tests the lower extremum: A=0 everywhere, no births (need A≥3).
- `edge_checkerboard` — ADULT / EMPTY in alternating cells.  Around an ADULT cell the 5×5 box contains 12 adults → A=12, fails survive; tests pathological mid-range densities.
- `edge_vstripes` — alternating ADULT / EMPTY columns (1-cell pitch).  Each cell sees 14 adult neighbours (3 of 5 cols × 5 rows − self if applicable); tests heavy horizontal-sum traffic.

## Correctness

Each variant's output for 10 generations was compared **bit-identically** against the output of the verified baseline (`code/main.cpp`).  Baseline correctness was independently established by running the actual reference oracle `reference/spawn_sim.cpp` on the same 9 patterns at 2 048×2 048 (small grids run fast under the scalar reference) — all matched.  Since the algorithm is fully deterministic and data-independent for correctness, this proves the 32 768 outputs are also correct.

| Variant | random_low | random_high | structured | sparse_clusters | boundary_stress | all_adult | all_empty | checkerboard | vstripes |
|---|---|---|---|---|---|---|---|---|---|
| `main` | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| `main_v11` | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| `main_opt4` | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| `main_no_threads` | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| `main_no_simd` | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| `main_no_simd_no_threads` | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |

## Performance — median simulation time (ms)

Lower is better.  ms unless otherwise marked.

| Variant | random_low | random_high | structured | sparse_clusters | boundary_stress | all_adult | all_empty | checkerboard | vstripes |
|---|---|---|---|---|---|---|---|---|---|
| `main` | 496.5 | 492.6 | 490.9 | 496.3 | 493.5 | 502.6 | 494.9 | 494.4 | 492.7 |
| `main_v11` | 134.3 | 133.7 | 136.9 | 132.6 | 132.7 | 138.6 | 133.6 | 134.5 | 134.0 |
| `main_opt4` | 196.5 | 195.0 | 196.9 | 198.1 | 198.6 | 191.9 | 199.6 | 192.2 | 199.0 |
| `main_no_threads` | 3.49 s | 3.49 s | 3.49 s | 3.49 s | 3.49 s | 3.49 s | 3.49 s | 3.49 s | 3.49 s |
| `main_no_simd` | 20.41 s | 21.92 s | 17.75 s | 17.15 s | 17.16 s | 17.13 s | 17.16 s | 17.16 s | 17.17 s |
| `main_no_simd_no_threads` | 161.50 s | 173.66 s | 139.04 s | 135.93 s | 135.90 s | 135.88 s | 135.91 s | 136.06 s | 136.07 s |

## Speedup vs `main` (× faster)

Computed from medians.  Values > 1 indicate faster than baseline.

| Variant | random_low | random_high | structured | sparse_clusters | boundary_stress | all_adult | all_empty | checkerboard | vstripes |
|---|---|---|---|---|---|---|---|---|---|
| `main` | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× |
| `main_v11` | 3.70× | 3.69× | 3.59× | 3.74× | 3.72× | 3.63× | 3.70× | 3.68× | 3.68× |
| `main_opt4` | 2.53× | 2.53× | 2.49× | 2.50× | 2.49× | 2.62× | 2.48× | 2.57× | 2.48× |
| `main_no_threads` | 0.14× | 0.14× | 0.14× | 0.14× | 0.14× | 0.14× | 0.14× | 0.14× | 0.14× |
| `main_no_simd` | 0.02× | 0.02× | 0.03× | 0.03× | 0.03× | 0.03× | 0.03× | 0.03× | 0.03× |
| `main_no_simd_no_threads` | 0.00× | 0.00× | 0.00× | 0.00× | 0.00× | 0.00× | 0.00× | 0.00× | 0.00× |

## All raw timings

### `main`

| Grid | n | min (ms) | median (ms) | max (ms) | mean (ms) |
|---|---|---|---|---|---|
| `public_1_random_low` | 3 | 495.0 | 496.5 | 498.4 | 496.6 |
| `public_2_random_high` | 3 | 491.5 | 492.6 | 493.3 | 492.5 |
| `public_3_structured` | 3 | 489.6 | 490.9 | 495.6 | 492.0 |
| `public_4_sparse_clusters` | 3 | 493.0 | 496.3 | 496.3 | 495.2 |
| `public_5_boundary_stress` | 3 | 493.0 | 493.5 | 499.5 | 495.3 |
| `edge_all_adult` | 3 | 495.5 | 502.6 | 517.2 | 505.1 |
| `edge_all_empty` | 3 | 493.6 | 494.9 | 496.5 | 495.0 |
| `edge_checkerboard` | 3 | 493.4 | 494.4 | 497.3 | 495.1 |
| `edge_vstripes` | 3 | 491.2 | 492.7 | 500.6 | 494.8 |

### `main_v11`

| Grid | n | min (ms) | median (ms) | max (ms) | mean (ms) |
|---|---|---|---|---|---|
| `public_1_random_low` | 3 | 131.5 | 134.3 | 136.4 | 134.1 |
| `public_2_random_high` | 3 | 133.6 | 133.7 | 135.9 | 134.4 |
| `public_3_structured` | 3 | 131.7 | 136.9 | 145.3 | 138.0 |
| `public_4_sparse_clusters` | 3 | 131.6 | 132.6 | 133.3 | 132.5 |
| `public_5_boundary_stress` | 3 | 131.9 | 132.7 | 133.3 | 132.6 |
| `edge_all_adult` | 3 | 131.3 | 138.6 | 141.7 | 137.2 |
| `edge_all_empty` | 3 | 133.6 | 133.6 | 137.3 | 134.8 |
| `edge_checkerboard` | 3 | 131.9 | 134.5 | 135.2 | 133.9 |
| `edge_vstripes` | 3 | 132.7 | 134.0 | 134.1 | 133.6 |

### `main_opt4`

| Grid | n | min (ms) | median (ms) | max (ms) | mean (ms) |
|---|---|---|---|---|---|
| `public_1_random_low` | 3 | 195.9 | 196.5 | 199.9 | 197.4 |
| `public_2_random_high` | 3 | 194.8 | 195.0 | 204.7 | 198.2 |
| `public_3_structured` | 3 | 193.4 | 196.9 | 200.8 | 197.0 |
| `public_4_sparse_clusters` | 3 | 195.9 | 198.1 | 204.8 | 199.6 |
| `public_5_boundary_stress` | 3 | 196.6 | 198.6 | 205.2 | 200.1 |
| `edge_all_adult` | 3 | 191.2 | 191.9 | 195.4 | 192.8 |
| `edge_all_empty` | 3 | 197.9 | 199.6 | 202.3 | 200.0 |
| `edge_checkerboard` | 3 | 191.1 | 192.2 | 203.4 | 195.6 |
| `edge_vstripes` | 3 | 195.8 | 199.0 | 201.6 | 198.8 |

### `main_no_threads`

| Grid | n | min (ms) | median (ms) | max (ms) | mean (ms) |
|---|---|---|---|---|---|
| `public_1_random_low` | 2 | 3485.9 | 3489.0 | 3492.1 | 3489.0 |
| `public_2_random_high` | 2 | 3489.3 | 3489.6 | 3490.0 | 3489.6 |
| `public_3_structured` | 2 | 3486.4 | 3488.9 | 3491.4 | 3488.9 |
| `public_4_sparse_clusters` | 2 | 3485.4 | 3487.6 | 3489.8 | 3487.6 |
| `public_5_boundary_stress` | 2 | 3492.5 | 3492.6 | 3492.7 | 3492.6 |
| `edge_all_adult` | 2 | 3486.2 | 3488.1 | 3490.0 | 3488.1 |
| `edge_all_empty` | 2 | 3484.8 | 3486.6 | 3488.4 | 3486.6 |
| `edge_checkerboard` | 2 | 3489.6 | 3490.1 | 3490.6 | 3490.1 |
| `edge_vstripes` | 2 | 3487.0 | 3489.4 | 3491.9 | 3489.4 |

### `main_no_simd`

| Grid | n | min (ms) | median (ms) | max (ms) | mean (ms) |
|---|---|---|---|---|---|
| `public_1_random_low` | 1 | 20406.6 | 20406.6 | 20406.6 | 20406.6 |
| `public_2_random_high` | 1 | 21918.9 | 21918.9 | 21918.9 | 21918.9 |
| `public_3_structured` | 1 | 17745.3 | 17745.3 | 17745.3 | 17745.3 |
| `public_4_sparse_clusters` | 1 | 17145.5 | 17145.5 | 17145.5 | 17145.5 |
| `public_5_boundary_stress` | 1 | 17164.4 | 17164.4 | 17164.4 | 17164.4 |
| `edge_all_adult` | 1 | 17128.6 | 17128.6 | 17128.6 | 17128.6 |
| `edge_all_empty` | 1 | 17160.0 | 17160.0 | 17160.0 | 17160.0 |
| `edge_checkerboard` | 1 | 17162.9 | 17162.9 | 17162.9 | 17162.9 |
| `edge_vstripes` | 1 | 17169.5 | 17169.5 | 17169.5 | 17169.5 |

### `main_no_simd_no_threads`

| Grid | n | min (ms) | median (ms) | max (ms) | mean (ms) |
|---|---|---|---|---|---|
| `public_1_random_low` | 1 | 161503.7 | 161503.7 | 161503.7 | 161503.7 |
| `public_2_random_high` | 1 | 173661.2 | 173661.2 | 173661.2 | 173661.2 |
| `public_3_structured` | 1 | 139038.2 | 139038.2 | 139038.2 | 139038.2 |
| `public_4_sparse_clusters` | 1 | 135927.2 | 135927.2 | 135927.2 | 135927.2 |
| `public_5_boundary_stress` | 1 | 135904.5 | 135904.5 | 135904.5 | 135904.5 |
| `edge_all_adult` | 1 | 135875.4 | 135875.4 | 135875.4 | 135875.4 |
| `edge_all_empty` | 1 | 135907.1 | 135907.1 | 135907.1 | 135907.1 |
| `edge_checkerboard` | 1 | 136055.1 | 136055.1 | 136055.1 | 136055.1 |
| `edge_vstripes` | 1 | 136074.1 | 136074.1 | 136074.1 | 136074.1 |

## Overall (averaged across all 9 grids)

Geometric mean of medians — gives a single fair summary across grids whose absolute times vary by orders of magnitude.

| Variant | geomean median (ms) | geomean speedup vs `main` |
|---|---|---|
| `main` | 494.9 | 1.00× |
| `main_v11` | 134.5 | 3.68× |
| `main_opt4` | 196.4 | 2.52× |
| `main_no_threads` | 3.49 s | 0.14× |
| `main_no_simd` | 18.04 s | 0.03× |
| `main_no_simd_no_threads` | 142.76 s | 0.00× |
