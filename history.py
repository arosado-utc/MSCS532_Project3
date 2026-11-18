"""
MSCS 532 - Project (Phase 3)
EmuCache: history-based block cache.

LRU treats every recent reference equally, so a one-shot scan
(level loads, table walks) pushes the hot core out of the cache
and the next core round pays the re-miss. History-based
policies fix this by remembering how many times a block was
referenced, not just when: a first reference goes to a small
FIFO quarantine (the AIN queue) and only a second reference
promotes the block to the adaptive main region (the AMR queue,
LRU ordered). This is the 2Q structure of Johnson and Shasha
(1994) without the ghost queue.

The split here is 50/50 rather than the 25/75 of the original
2Q, because the hot cores in the stress workloads are four to
six blocks and a quarter-quarantine thrashes them; the capacity
sweep in benchmark3.py shows where the trade-off lands.

Eviction order: a block that overflows the cache evicts from
the AIN head if AIN is not empty, otherwise from the AMR tail.
A scan therefore displaces other scan blocks until AIN drains,
and only then reaches the resident core.

The interface (get / put / remove / evictions) matches
LRUCache so the emulator can take either one. The operations
reorder Python lists, so they are O(n) in the cache size
instead of the LRU cache's O(1); at these sizes the scan
resistance buys back more than the reordering costs.
"""

from cache import HashTable

_AIN = 0   # segment tag: first-reference quarantine
_AMR = 1   # segment tag: adaptive main region (LRU)


class HistoryCache:
    """Bounded cache with 2Q-style history-based eviction.

    A HashTable maps guest PC -> block for O(1) lookup, and two
    ordered key lists hold the recency history: AIN is a FIFO of
    first-referenced keys (at most half the capacity) and AMR
    is an LRU list of promoted keys. A hit in AIN promotes the
    key to the AMR tail.

    Complexity:  get / put / remove  -  O(n) in the cache size
    """

    def __init__(self, capacity=8):
        if capacity < 2:
            raise ValueError("capacity must be >= 2")
        self._capacity = capacity
        self._ain_cap = capacity // 2
        self._table = HashTable(capacity * 4)   # key -> block
        self._seg = HashTable(capacity * 4)     # key -> segment tag
        self._ain = []   # FIFO; index 0 is the eviction candidate
        self._amr = []   # LRU order; index 0 is the least recent
        self._size = 0
        self.evictions = 0

    # public interface
    def get(self, key):
        """Return the cached value for key, or None on a miss.
        A hit in AIN promotes the entry to AMR; a hit in AMR
        refreshes its LRU position."""
        value = self._table.get(key)
        if value is None:
            return None
        if self._seg.get(key) == _AIN:
            self._ain.remove(key)
            self._amr.append(key)
            self._seg.put(key, _AMR)
        else:
            self._amr.remove(key)
            self._amr.append(key)
        return value

    def put(self, key, value):
        """Insert (or refresh) an entry. Returns the evicted
        (key, value) pair, or None if nothing was evicted.
        New entries always land in AIN."""
        if self._table.get(key) is not None:
            self._table.put(key, value)
            if self._seg.get(key) == _AIN:
                self._ain.remove(key)
                self._ain.append(key)
            else:
                self._amr.remove(key)
                self._amr.append(key)
            return None
        evicted = None
        if self._size >= self._capacity:
            evicted = self._evict_one()
        self._table.put(key, value)
        self._seg.put(key, _AIN)
        self._ain.append(key)
        self._size += 1
        if len(self._ain) > self._ain_cap:
            evicted = self._evict_one()
        return evicted

    def remove(self, key):
        """Remove and return the cached value for key, or None.
        Used for invalidation drops; not counted as an eviction."""
        value = self._table.get(key)
        if value is None:
            return None
        if self._seg.get(key) == _AIN:
            self._ain.remove(key)
        else:
            self._amr.remove(key)
        self._table.delete(key)
        self._seg.delete(key)
        self._size -= 1
        return value

    def __len__(self):
        return self._size

    # eviction
    def _evict_one(self):
        """Evict the AIN head if AIN is not empty, else the AMR
        tail, and return the evicted (key, value) pair."""
        if self._ain:
            key = self._ain.pop(0)
        else:
            key = self._amr.pop(0)
        value = self._table.get(key)
        self._table.delete(key)
        self._seg.delete(key)
        self._size -= 1
        self.evictions += 1
        return (key, value)


def main():
    c = HistoryCache(4)   # AIN holds at most 2, AMR the other 2
    for k in (0x10, 0x20, 0x30, 0x40):
        c.put(k, f"block@{k:#x}")
    print("after 4 inserts (2 AIN evictions):", len(c), "blocks")
    for k in (0x10, 0x20, 0x30, 0x40):
        print(f"  get {k:#x} -> {c.get(k)}")
    print("  evictions so far:", c.evictions)
    print("  amr now:", [hex(k) for k in c._amr])
    print("  ain now:", [hex(k) for k in c._ain])


if __name__ == "__main__":
    main()
