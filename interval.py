"""
MSCS 532 - Project (Phase 3)
EmuCache: interval-granularity self-modifying code watcher.

Phase 1's PageWatcher trades exactness for speed: a write to a
code page evicts every block that page covers, so a patch to one
byte of a loop body also kills the loopback block that sits next
to it on the same page. This module is the other end of that
trade-off: each block registers its exact half-open range
[start_pc, end_pc), and a write returns exactly the blocks whose
range contains the written address. No collateral evictions.

The ranges live in a list sorted by start address. register and
unregister find their spot with a binary search (bisect) and pay
an O(n) list shift, and on_write bisects the window of ranges
whose start is at or below the address, then filters that window
by end. At the scale of this emulator - a dozen cached blocks -
the scan is cheap, and the win is that the invalidation count
matches the number of blocks the write actually touches, which
the patch sweep in benchmark3.py measures directly.
"""

import bisect


class IntervalWatcher:
    """Watches exact half-open code ranges for self-modifying
    writes.

    A block registers the range it covers, and a write to addr
    returns the set of block IDs with start <= addr < end.

    Complexity:
        register / unregister  - O(log n) search + O(n) shift
        on_write               - O(log n + w), w the width of the
                                 candidate window (O(n) worst case)
    """

    def __init__(self):
        self._starts = []   # start addresses, parallel to _entries
        self._entries = []  # (start, end, block_id), sorted by start

    def register(self, block_id, start_pc, end_pc):
        """Remember that block_id covers [start_pc, end_pc)."""
        i = bisect.bisect_right(self._starts, start_pc)
        self._starts.insert(i, start_pc)
        self._entries.insert(i, (start_pc, end_pc, block_id))

    def unregister(self, block_id, start_pc, end_pc):
        """Forget block_id (it left the cache or was invalidated).
        Silently does nothing if the range was not registered."""
        i = bisect.bisect_left(self._starts, start_pc)
        for j in range(i, len(self._entries)):
            s, e, b = self._entries[j]
            if s > start_pc:
                return
            if (s, e, b) == (start_pc, end_pc, block_id):
                del self._entries[j]
                del self._starts[j]
                return

    def on_write(self, addr):
        """Return the set of block IDs invalidated by a write to
        addr. Only ranges that actually contain addr qualify, so
        a write between two blocks invalidates nothing."""
        i = bisect.bisect_right(self._starts, addr)
        victims = set()
        for j in range(i):
            s, e, b = self._entries[j]
            if e > addr:
                victims.add(b)
        return victims


def main():
    w = IntervalWatcher()
    w.register(1, 0x00, 0x10)
    w.register(2, 0x10, 0x20)
    w.register(3, 0x08, 0x18)   # overlaps both of the above
    print("write 0x05 ->", w.on_write(0x05))    # {1}
    print("write 0x10 ->", w.on_write(0x10))    # {2, 3}, half-open edge
    print("write 0x15 ->", w.on_write(0x15))    # {2, 3}
    print("write 0x20 ->", w.on_write(0x20))    # {}
    w.unregister(3, 0x08, 0x18)
    print("after dropping 3, write 0x15 ->", w.on_write(0x15))


if __name__ == "__main__":
    main()
