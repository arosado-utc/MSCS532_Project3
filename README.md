# MSCS532 Project: EmuCache

A **block translation cache** for a dynamic game emulator, built
in Python. Emulators like QEMU, RPCS3, and Dolphin
run game code by translating it in small pieces on demand and
caching the results - this project implements and tests the data
structures that make that caching work.

## The data structures

| File         | Structure                                        | Complexity (get/put) |
| ------------ | ------------------------------------------------ | -------------------- |
| `cache.py`   | `HashTable` - open addressing, linear probing, tombstones, rehash at 0.7 load | O(1) avg |
| `cache.py`   | `LRUCache` - HashTable + doubly linked list      | O(1) |
| `smc.py`     | `PageWatcher` - page -> set of block IDs         | O(1) lookup |
| `guest.py`   | toy guest machine + basic block decoder          | - |
| `emulator.py`| the recompiler core tying the pieces together    | - |
| `workloads.py` | parameterized guest workloads (Phase 2)         | - |
| `benchmark.py` | cache-size and input-mix sweeps (Phase 2)      | - |
| `history.py`   | `HistoryCache` - 2Q-style quarantine + LRU main region (Phase 3) | O(n) |
| `interval.py`  | `IntervalWatcher` - exact [start, end) ranges, bisect (Phase 3) | O(log n + w) per write |
| `stress.py`    | Phase 3 stress workloads                        | - |
| `benchmark3.py` | Phase 3 sweeps against the Phase 1/2 baselines | - |

Why these choices:

- **Open addressing** instead of chaining: keys are small
  consecutive integers (program counters), the table stays in a
  couple of arrays, and lookups never chase pointers - the same
  shape of structure QEMU uses for its block table (Bellard,
  2005). Knuth's multiplicative hash spreads consecutive PCs
  across the whole table.
- **LRU = hash table + doubly linked list**: the table answers
  "is this block cached?" in O(1) and the list answers "which
  block is the least recent?" in O(1), so hits, inserts, and
  evictions are all constant time.
- **Page-granularity SMC watching**: a write to a watched code
  page evicts exactly the blocks that page covers - O(1) per
  write, with bounded over-invalidation instead of O(n)
  per-byte tracking.

## How it works

The guest machine (`guest.py`) is a small register machine with
a 3-byte instruction set. A **basic block** starts at a program
counter and ends at the first branch, jump, or halt. The
emulator (`emulator.py`) dispatches through the `LRUCache`: a
hit runs the cached block, a miss decodes and caches a new one.
When the guest `STORE`s into its own code region, the
`PageWatcher` returns the affected block IDs and they are
evicted, so the next dispatch retranslates the patched code.

## Demo

Two guest programs in `main.py`:

1. **A counter** - 100 iterations of a hot loop. Shows cache
   effectiveness (98% hits at capacity 8) and what capacity
   pressure does (capacity 1: every fetch retranslates).
2. **A self-patching game** - the game's own routine rewrites
   its loop's increment mid-run (self-modifying code, exactly
   what real console emulators hit with hot-patched games).
   The final score 400 = 50x1 + 50x7 only comes out right if
   the SMC invalidation and retranslation work.

Run it:

```bash
python3 main.py
```

## Phase 2: proof of concept

Phase 2 extends the demo to parameterized workloads and reports
the cache statistics as a function of cache size and input mix.
`workloads.py` builds four families on the same machine:

| Workload        | Access pattern            | Working set | Notes |
| --------------- | ------------------------- | ----------- | ----- |
| `counter(n)`    | one hot loop              | 4 blocks    | baseline; the loopback targets 0x06, inside the first block |
| `multi_phase_game(n)` | three sequential phases | 10 blocks | phases are visited once and retired |
| `hub_game(n)`   | round-robin over three routines | 7 blocks | every block stays live until the countdowns expire |
| `patch_game(cycles, n)` | one loop that rewrites its own increment register | 5 blocks + retranslations | the working set is created by SMC invalidation, not by code layout |

`benchmark.py` runs three sweeps and writes CSVs to `results/`
and plots to `plots/`:

- **Capacity sweep** - hit rate vs cache capacity (1..16) for
  each workload. The counter saturates at C = 2; the hub - a
  six-block round-robin - stays under 1% hit rate until C = 5
  and jumps to 96% at C = 6, because a cycle of six distinct
  blocks cannot be held in a smaller LRU; the multi-phase game
  saturates at C = 2 with no re-misses, because its retired
  blocks are never re-fetched.
- **Size sweep** - wall time vs workload size at C = 8. All
  three families scale linearly in the fetch count.
- **Patch sweep** - hit and invalidation rates vs patch period
  at a fixed 300 total iterations. Shorter patch periods cost
  ~3.2% of the hit rate and raise the invalidation rate from
  5 to 38 per 1000 fetches.

Run it:

```bash
python3 benchmark.py
```

## Phase 3: the trade-off study

Phase 3 pushes on the two spots where Phases 1 and 2 made
deliberate simplifications, the eviction policy and the SMC
granularity. Each new structure drops in behind the same
dispatch loop (the emulator constructor now takes an optional
cache and watcher) and is measured against the Phase 1/2
baseline on workloads built to expose it.

Two design notes:

- **HistoryCache** is the 2Q structure of Johnson and Shasha
  (1994) without the ghost queue. New blocks land in a FIFO
  quarantine (AIN, half the capacity) and a second reference
  promotes them to the LRU-ordered main region (AMR, the other
  half). Eviction takes the AIN head first, so a one-shot scan
  displaces other scan blocks before it can reach the resident
  core. The split is 50/50 rather than 2Q's 25/75 because the
  hot cores here are four to six blocks and a quarter-sized
  quarantine thrashes them.
- **IntervalWatcher** registers each block's exact half-open
  range and returns exactly the blocks a write touches. No
  collateral evictions, at the cost of an O(n) scan instead of
  the page watcher's O(1) lookup - a fine trade at a dozen
  cached blocks, and the invalidation count is the number of
  blocks the write actually hit.

The stress workloads in `stress.py` widen the code region to
[0x00, 0x80) (data moves to 0x90 and up) so a program can hold
a hot core plus a long tail of one-shot blocks:

| Workload                        | Access pattern                    | What it exposes |
| ------------------------------- | --------------------------------- | --------------- |
| `mixed_scan(rounds, stubs, levels)` | a four-block hot core interleaved with one-shot stub chains | scan resistance |
| `level_chain(levels)`           | the same builder, variable levels | linear scaling in the fetch count |
| `patch_storm(cycles, reps)`     | one loop, three SMC writes per phase | page vs interval invalidation cost |

`benchmark3.py` writes CSVs to `results3/` and plots to
`plots3/`. The sweep at C = 8 is the headline:

- **Capacity sweep** - on the mixed scan, LRU is flat at 37.86%
  from C = 4 to 16 and only reaches 77.67% at C = 24, when the
  cache finally holds all 23 distinct blocks. The history cache
  is at 0% below C = 8 (the four-block core will not fit in a
  three-slot AMR) and then holds 45.63% at every larger
  capacity. History wins the middle range where LRU keeps
  paying for scans and loses at C = 24, where plain LRU simply
  fits everything. That crossover is the honest read of the
  sweep.
- **Level chain** - fetches grow by 34 per level (one core
  round plus one scan), the same slope for both caches, and
  wall time scales with it.
- **Patch storm** - invalidations vs patch frequency. The
  interval watcher pays exactly one block per patch burst, the
  loop body, for a total of cycles - 1. The page watcher also
  kills the loopback block that shares page 0, and the hot
  loopback re-misses after every burst, for a total of
  3 + 2*(cycles - 2). At 20 cycles the interval watcher
  invalidates 19 blocks and misses 25 times where the page
  watcher invalidates 39 and misses 44.

Run it:

```bash
python3 benchmark3.py
```

## Tests

```bash
python3 tests.py
```

40 tests cover the hash table (probing, tombstones, rehashing,
key spread), the LRU cache (eviction order, refresh, removal),
the page watcher (registration, spanning pages, unwatched
pages), the decoder, end-to-end runs of the Phase 1 programs
checked against a naive single-step reference interpreter, the
Phase 2 workloads (reference-interpreter agreement, the
hand-counted block/eviction/invalidation totals, builder
validation, the monotonicity of the hit rate in the cache
capacity), and the Phase 3 structures: the interval watcher's
exact ranges and unregistration, the history cache's promotion
and AIN-first eviction, the stress workloads against the
reference interpreter, and the measured hit-rate and
invalidation totals from benchmark3.py pinned by hand.

## References

- Bellard, F. (2005). QEMU, a fast and portable dynamic
  translator. *Proceedings of the USENIX Annual Technical
  Conference (FREENIX Track)*, 41-46.
- Johnson, T., & Shasha, D. (1994). 2Q: A low overhead high
  performance buffer management replacement algorithm.
  *Proceedings of the 20th International Conference on Very
  Large Data Bases (VLDB '94)*, 439-450. Morgan Kaufmann.
- Pagh, R., & Rodler, F. F. (2002). Cuckoo hashing. In
  *Proceedings of the 9th European Symposium on Algorithms
  (ESA 2001)* (Lecture Notes in Computer Science, Vol. 2161,
  pp. 121-133). Springer.
- Liu, Z., Wang, H., Wong, W. K., Wu, D., & Wang, S. (2026).
  No more translation at runtime: LLM-empowered static binary
  translation. *Proceedings of the 21st European Conference on
  Computer Systems (EuroSys '26)*, 1023-1040.
  https://doi.org/10.1145/3767295.3803600
