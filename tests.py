"""
MSCS 532 - Project (Phases 1 + 2 + 3)
EmuCache test suite.

Run with:  python3 tests.py

Covers the hash table, the LRU cache, the SMC watcher, the block
decoder, and full end-to-end runs of the demo guest programs. The
end-to-end tests are checked against a naive single-step
interpreter, so any divergence between the cached path and the
reference path fails the suite. Phase 2 adds the parameterized
workloads from workloads.py and the benchmark harness from
benchmark.py: every workload is checked against the reference
interpreter, and the expected block counts, eviction counts, and
invalidation counts are asserted by hand. Phase 3 adds the
history-based cache, the interval watcher, and the stress
workloads: the measured hit-rate and invalidation numbers from
benchmark3.py are pinned here by hand.
"""

from benchmark import run_case
from cache import HashTable, LRUCache
from emulator import Emulator
from guest import (CODE_END, Guest, OP_ADD, OP_HALT, OP_JMP, OP_JZ,
                   OP_LOAD, OP_STORE, OP_SUB)
from history import HistoryCache
from interval import IntervalWatcher
from main import program_counter, program_self_patching
from smc import PageWatcher
from stress import level_chain, mixed_scan, patch_storm
from workloads import counter, hub_game, multi_phase_game, patch_game


# reference interpreter: single-step, no blocks, no cache
def naive_run(guest, max_steps=100_000):
    """Step through the guest one instruction at a time."""
    regs, mem = guest.regs, guest.mem
    pc = 0
    for _ in range(max_steps):
        op, a, b = mem[pc], mem[pc + 1], mem[pc + 2]
        if op == OP_ADD:
            regs[a] = (regs[a] + regs[b]) % 65536
        elif op == OP_LOAD:
            regs[a] = mem[b]
        elif op == OP_STORE:
            mem[a] = regs[b] % 256
        elif op == OP_SUB:
            regs[a] = (regs[a] - regs[b]) % 65536
        elif op == OP_JMP:
            pc = a
            continue  # JMP sets pc; do not fall through
        elif op == OP_JZ:
            if regs[a] == 0:
                pc = b
            else:
                pc += 3
            continue  # JZ sets pc; do not fall through
        elif op == OP_HALT:
            return regs
        else:
            raise ValueError(f"undecodable opcode 0x{op:02X} at 0x{pc:02X}")
        pc += 3
    raise RuntimeError("guest did not halt")


# HashTable
def test_ht_basic_put_get():
    ht = HashTable()
    ht.put(0x10, "block1")
    assert ht.get(0x10) == "block1"
    ht.put(0x10, "block1-v2")
    assert ht.get(0x10) == "block1-v2"
    assert ht.get(0x99) is None
    assert ht.get(0x99, "default") == "default"


def test_ht_contains_and_delete():
    ht = HashTable()
    ht.put(5, "x")
    assert 5 in ht
    ht.delete(5)
    assert 5 not in ht
    assert ht.delete(5) is False
    # re-inserting a deleted key must work (tombstone reuse)
    ht.put(5, "y")
    assert ht.get(5) == "y"


def test_ht_rehash_preserves_entries():
    ht = HashTable(4)
    for k in range(0, 500, 3):  # forces several rehashes
        ht.put(k, k * 10)
    for k in range(0, 500, 3):
        assert ht.get(k) == k * 10
    assert len(ht) == 167


def test_ht_sequential_keys_spread():
    """Consecutive program counters must not cluster in one probe
    chain - 300 sequential keys, all findable."""
    ht = HashTable(16)
    for k in range(300):
        ht.put(k, k)
    for k in range(300):
        assert ht.get(k) == k
    assert len(ht) == 300


# LRUCache
def test_lru_eviction_order():
    c = LRUCache(2)
    c.put(1, "a")
    c.put(2, "b")
    c.get(1)             # 1 becomes most recent
    evicted = c.put(3, "c")
    assert evicted == (2, "b")  # 2 was the LRU entry
    assert c.get(2) is None
    assert c.get(1) == "a"
    assert c.get(3) == "c"


def test_lru_refresh_does_not_evict():
    c = LRUCache(2)
    c.put(1, "a")
    c.put(2, "b")
    assert c.put(1, "a2") is None
    assert c.get(1) == "a2"
    assert len(c) == 2


def test_lru_remove():
    c = LRUCache(2)
    c.put(1, "a")
    c.put(2, "b")
    assert c.remove(1) == "a"
    assert c.get(1) is None
    assert c.remove(1) is None
    assert len(c) == 1
    # the freed slot must be reusable without growth
    c.put(1, "a3")
    assert len(c) == 2


# PageWatcher
def test_watcher_basic():
    w = PageWatcher(page_size=32)
    w.register(1, 0x00, 0x10)   # page 0
    w.register(2, 0x20, 0x30)   # page 1
    assert w.on_write(0x05) == {1}
    assert w.on_write(0x25) == {2}
    assert w.on_write(0x40) == set()  # data page: not watched
    w.unregister(1, 0x00, 0x10)
    assert w.on_write(0x05) == set()
    assert w.on_write(0x25) == {2}


def test_watcher_spanning_pages():
    w = PageWatcher(page_size=32)
    w.register(7, 0x10, 0x30)  # spans pages 0 and 1
    assert w.on_write(0x1F) == {7}
    assert w.on_write(0x25) == {7}
    w.unregister(7, 0x10, 0x30)
    assert w.on_write(0x1F) == set()
    assert w.on_write(0x25) == set()


def test_watcher_bad_page_size():
    try:
        PageWatcher(page_size=24)
        assert False, "expected ValueError"
    except ValueError:
        pass


# Guest block decoder
def test_decode_block_stops_at_branch():
    g = program_counter()
    start, insts, end = g.decode_block(0x00)
    assert start == 0x00
    assert insts[-1][0] == OP_JZ  # block ends at, including, the JZ
    assert end == 0x0F            # first address past the block


def test_decode_block_missing_code():
    g = program_counter()
    try:
        g.decode_block(0x50)  # no code there
        assert False, "expected ValueError"
    except ValueError:
        pass


# End-to-end: emulator vs naive interpreter
def test_counter_program_matches_reference():
    g = program_counter()
    expected = naive_run(g)
    g2 = program_counter()
    actual = Emulator(g2, cache_capacity=8).run()
    assert actual == expected
    assert actual[0] == 100  # the score


def test_counter_cache_stats():
    emu = Emulator(program_counter(), cache_capacity=8)
    emu.run()
    assert emu.stats["misses"] == 4       # 4 distinct block starts
    assert emu.stats["hits"] == 196
    assert emu.stats["evictions"] == 0
    assert emu.stats["invalidations"] == 0


def test_counter_eviction_pressure():
    emu = Emulator(program_counter(), cache_capacity=1)
    regs = emu.run()
    assert regs[0] == 100                # still correct
    assert emu.stats["hits"] == 0        # nothing survives a fetch
    assert emu.stats["evictions"] == 199


def test_self_patching_program_matches_reference():
    g = program_self_patching()
    expected = naive_run(g)
    g2 = program_self_patching()
    actual = Emulator(g2, cache_capacity=8).run()
    assert actual == expected
    assert actual[0] == 400  # 50 x 1 + 50 x 7: the patched loop ran


def test_self_patching_invalidation_count():
    emu = Emulator(program_self_patching(), cache_capacity=8)
    emu.run()
    # four cached blocks cover page 0 (0x00, 0x09, 0x12, 0x1A)
    assert emu.stats["invalidations"] == 4
    # the SMC write must not evict via capacity pressure
    assert emu.stats["evictions"] == 0


def test_data_write_does_not_invalidate():
    """Stores to the data region (>= 0x40) must not touch the cache."""
    emu = Emulator(program_counter(), cache_capacity=8)
    emu.run()
    before = dict(emu.stats)
    emu._store(0x50, 0xAB)  # data region write
    assert emu.stats["invalidations"] == before["invalidations"]


def test_non_halting_guest_hits_step_limit():
    # a program that jumps to itself: 0x00: JMP 0x00
    g = Guest([(0x00, OP_JMP), (0x01, 0x00), (0x02, 0)])
    emu = Emulator(g, cache_capacity=4)
    try:
        emu.run(max_steps=100)
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


# Phase 2: workloads
def test_workloads_match_reference():
    """Every workload must produce the same register file as the
    naive interpreter, and the expected final score."""
    cases = [
        (lambda: counter(100), 100),
        (lambda: multi_phase_game(50), 21 * 50),
        (lambda: hub_game(30), 7 * 30),
        (lambda: patch_game(3, 50), 17 * 50),
    ]
    for builder, score in cases:
        expected = naive_run(builder())
        actual = Emulator(builder(), cache_capacity=8).run()
        assert actual == expected
        assert actual[0] == score


def test_counter_working_set():
    """The counter has four block starts (the loopback targets
    0x06, inside the first block); C = 8 covers them all."""
    emu = Emulator(counter(255), cache_capacity=8)
    emu.run()
    assert emu.stats["misses"] == 4
    assert emu.stats["evictions"] == 0


def test_counter_capacity_one():
    emu = Emulator(counter(100), cache_capacity=1)
    emu.run()
    assert emu.stats["hits"] == 0
    assert emu.stats["misses"] == 200


def test_hub_working_set_and_thrash():
    """The hub's seven blocks stay live until the countdowns
    expire. A C = 8 cache holds them all (seven misses, no
    evictions); a C = 3 cache cannot hold the six-block round
    and re-misses on almost every fetch."""
    big = Emulator(hub_game(30), cache_capacity=8)
    big.run()
    assert big.stats["misses"] == 7
    assert big.stats["evictions"] == 0
    small = Emulator(hub_game(30), cache_capacity=3)
    small.run()
    assert small.stats["misses"] > 100
    assert small.stats["evictions"] > 0


def test_multi_phase_no_remisses():
    """The phases are sequential: retired blocks are never
    re-fetched, so misses stay at the working-set size (10)
    even when the cache cannot hold them - evictions happen,
    re-misses do not."""
    emu = Emulator(multi_phase_game(50), cache_capacity=2)
    emu.run()
    assert emu.stats["misses"] == 10
    assert emu.stats["evictions"] == 8


def test_patch_invalidations_follow_the_page():
    """The SMC write lands on page 0; the check and patch
    blocks sit on page 1 and survive their own writes. The
    first patch invalidates three cached blocks (prologue,
    body, loopback), later patches invalidate two (the
    prologue is gone), so four cycles give 3 + 2 + 2 = 7."""
    emu = Emulator(patch_game(4, 20), cache_capacity=8)
    emu.run()
    assert emu.stats["invalidations"] == 7
    assert emu.stats["misses"] == 12
    assert emu.stats["evictions"] == 0


def test_patch_single_cycle_no_smc():
    """One cycle means no patch routine run and no SMC write."""
    emu = Emulator(patch_game(1, 100), cache_capacity=8)
    emu.run()
    assert emu.stats["invalidations"] == 0
    assert emu.stats["misses"] == 5


def test_workload_builder_validation():
    for bad in (dict(reps=0), dict(reps=256)):
        try:
            counter(**bad)
            assert False, "expected ValueError"
        except ValueError:
            pass
    for bad in (0, 256):
        try:
            hub_game(bad)
            assert False, "expected ValueError"
        except ValueError:
            pass
    try:
        patch_game(cycles=0, reps=50)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_hit_rate_monotone_in_capacity():
    """A larger LRU cache never misses more than a smaller one
    on the same access sequence, so the hit rate must be
    non-decreasing in the capacity."""
    for builder in (lambda: counter(100), lambda: multi_phase_game(50),
                    lambda: hub_game(30), lambda: patch_game(3, 50)):
        rates = [run_case(builder, c)["hit_rate_pct"]
                 for c in (1, 2, 3, 4, 5, 6, 8, 10, 16)]
        for lo, hi in zip(rates, rates[1:]):
            assert hi >= lo, (builder, rates)


def test_run_case_row_is_consistent():
    row = run_case(lambda: patch_game(3, 50), 8)
    assert row["hits"] + row["misses"] == row["fetches"]
    assert row["hit_rate_pct"] == round(
        100.0 * row["hits"] / row["fetches"], 2)
    assert 0.0 <= row["hit_rate_pct"] <= 100.0


# Phase 3: interval watcher
def test_interval_exact_and_overlap():
    """A write returns exactly the ranges that contain it: the
    half-open edge excludes the range that ends there, and a
    write in the gap between ranges invalidates nothing."""
    w = IntervalWatcher()
    w.register(1, 0x00, 0x10)
    w.register(2, 0x10, 0x20)
    w.register(3, 0x08, 0x18)   # overlaps both of the above
    assert w.on_write(0x05) == {1}
    assert w.on_write(0x0A) == {1, 3}
    assert w.on_write(0x10) == {2, 3}   # 1 ends at 0x10, excluded
    assert w.on_write(0x18) == {2}      # 3 ends at 0x18, excluded
    assert w.on_write(0x20) == set()
    assert w.on_write(0x30) == set()    # gap: no range covers it


def test_interval_unregister():
    w = IntervalWatcher()
    w.register(1, 0x00, 0x10)
    w.register(3, 0x08, 0x18)
    w.unregister(3, 0x08, 0x18)
    assert w.on_write(0x09) == {1}      # only 1 remains over 0x09
    assert w.on_write(0x15) == set()    # nothing covers 0x15 now
    # dropping a range that was never registered is a silent no-op
    w.unregister(9, 0x40, 0x50)
    assert w.on_write(0x44) == set()


def test_interval_no_collateral():
    """The case the page watcher gets wrong: two blocks on the
    same page, and a write to one byte kills only the block that
    covers that byte."""
    w = IntervalWatcher()
    w.register(1, 0x00, 0x08)
    w.register(2, 0x08, 0x10)
    assert w.on_write(0x03) == {1}
    assert w.on_write(0x0B) == {2}


# Phase 3: history cache
def test_history_promotion_and_ain_first_eviction():
    """A hit in AIN promotes the entry to AMR, and eviction takes
    the AIN head before it touches the AMR - even when the AIN
    entry is the newest one in the cache."""
    c = HistoryCache(4)   # AIN holds at most 2
    c.put(1, "a")
    c.put(2, "b")
    assert c.get(1) == "a"      # 1 promoted: AIN [2], AMR [1]
    c.put(3, "c")               # AIN [2, 3]
    ev = c.put(4, "d")          # AIN would reach 3, over quota
    assert ev == (2, "b")       # the AIN head goes, not the AMR
    assert c.get(2) is None
    assert c.get(3) == "c"      # promote 3
    assert c.get(4) == "d"      # promote 4; AIN is now empty
    c.put(5, "e")               # AIN [5], cache is full
    ev = c.put(6, "f")          # AIN still non-empty
    assert ev == (5, "e")       # AIN head beats the AMR tail


def test_history_remove_and_interface():
    """remove() drops an entry without counting it as an
    eviction, and the get / put / remove / evictions surface
    matches LRUCache."""
    c = HistoryCache(4)
    c.put(1, "a")
    c.put(2, "b")
    assert len(c) == 2
    assert c.remove(1) == "a"
    assert c.get(1) is None
    assert c.remove(1) is None
    assert c.evictions == 0
    lru = LRUCache(4)
    lru.put(1, "a")
    assert lru.remove(1) == "a"
    assert lru.remove(1) is None


def test_history_capacity_validation():
    try:
        HistoryCache(1)   # needs at least one slot per segment
        assert False, "expected ValueError"
    except ValueError:
        pass


# Phase 3: stress workloads
def test_guest_code_end_default_and_wide():
    """The Phase 1/2 split is the default; the stress workloads
    widen it, and the emulator honors whatever the guest says."""
    assert program_counter().code_end == CODE_END
    assert mixed_scan().code_end == 0x80
    assert patch_storm(3, 20).code_end == 0x48


def test_stress_workloads_match_reference():
    """Every stress workload must produce the same register file
    as the naive interpreter, and the expected final score."""
    cases = [
        (lambda: mixed_scan(), 72),
        (lambda: mixed_scan(levels=2), 48),
        (lambda: mixed_scan(rounds=2, stubs=8, levels=4), 48),
        (lambda: level_chain(4), 96),
        (lambda: patch_storm(1, 20), 20),
        (lambda: patch_storm(2, 20), 120),
        (lambda: patch_storm(5, 20), 780),
    ]
    for builder, score in cases:
        expected = naive_run(builder())
        actual = Emulator(builder(), cache_capacity=8).run()
        assert actual == expected
        assert actual[0] == score


def test_mixed_scan_history_beats_lru():
    """The stub scans push the hot core out of the LRU cache;
    the history cache quarantines first references and keeps the
    core resident, so it misses 56 times where LRU misses 64.
    No store in the workload touches the code region, so both
    runs have zero invalidations."""
    lru = Emulator(mixed_scan(), cache=LRUCache(8))
    lru.run()
    hist = Emulator(mixed_scan(), cache=HistoryCache(8))
    hist.run()
    assert lru.stats["misses"] == 64
    assert hist.stats["misses"] == 56
    assert lru.stats["invalidations"] == 0
    assert hist.stats["invalidations"] == 0


def test_level_chain_fetches_scale_linearly():
    """Each level adds one core round and one scan, so fetches
    grow by 4*rounds + stubs + 3 = 34 per level at the default
    parameters, and the total matches the builder's formula."""
    prev = None
    for lv in (2, 3, 4, 5, 6):
        emu = Emulator(level_chain(lv), cache_capacity=8)
        emu.run()
        f = emu.stats["hits"] + emu.stats["misses"]
        if prev is not None:
            assert f - prev == 34, (lv, f, prev)
        prev = f
    assert prev == 1 + 6 * (4 * 4 + 15 + 3)


def test_patch_storm_interval_beats_page():
    """Each phase re-applies the patch three times, but only the
    first write of a burst finds the affected blocks still
    registered (the drop unregisters them). The interval watcher
    pays one block per burst, the loop body, while the page
    watcher also kills the loopback block that shares page 0 and
    re-misses it. With cycles >= 2: interval = cycles - 1,
    page = 3 + 2*(cycles - 2)."""
    for cycles in (3, 5, 10, 20):
        page = Emulator(patch_storm(cycles, 20), cache=LRUCache(8),
                        watcher=PageWatcher(page_size=32))
        page.run()
        interval = Emulator(patch_storm(cycles, 20), cache=LRUCache(8),
                            watcher=IntervalWatcher())
        interval.run()
        assert interval.stats["invalidations"] == cycles - 1
        assert page.stats["invalidations"] == 3 + 2 * (cycles - 2)
        assert interval.stats["misses"] < page.stats["misses"]


def main():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL: {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR: {t.__name__}: {type(e).__name__}: {e}")

    print()
    if failed:
        print(f"{failed} of {len(tests)} test(s) FAILED.")
        raise SystemExit(1)
    print(f"All {len(tests)} tests passed.")


if __name__ == "__main__":
    main()
