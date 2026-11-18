"""
MSCS 532 - Project (Phase 1)
EmuCache: the dynamic recompiler core.

Ties the pieces together: a dispatch loop that asks the LRU
cache for the translated block at the current program counter,
translates it on a miss, executes the block's instructions, and invalidates cached
blocks whenever the guest writes
to its own code (self-modifying code).

The statistics counters are what make the data structures
observable: hits/misses show cache effectiveness, evictions show
capacity pressure, and invalidations show the SMC cost.

Phase 3 makes the cache and the watcher pluggable: the
constructor takes optional cache and watcher objects, so the
history-based cache and the interval-granularity watcher can be
dropped in behind the same dispatch loop. The SMC check uses
the guest's own code_end, so a widened code region is honored.
"""

from cache import LRUCache
from guest import CODE_END, Guest, OP_ADD, OP_HALT, OP_JMP, OP_JZ, OP_LOAD, OP_STORE, OP_SUB
from smc import PageWatcher


class Block:
    """A translated basic block: its identity, guest range, and
    the instructions it executes."""

    __slots__ = ("block_id", "start_pc", "end_pc", "instructions")

    def __init__(self, block_id, start_pc, end_pc, instructions):
        self.block_id = block_id
        self.start_pc = start_pc
        self.end_pc = end_pc
        self.instructions = instructions

    def __repr__(self):
        return (f"Block#{self.block_id} "
                f"[0x{self.start_pc:02X}-0x{self.end_pc:02X}) "
                f"{len(self.instructions)} insns")


class Emulator:
    """Dynamic recompiler over the toy guest machine."""

    def __init__(self, guest, cache_capacity=8, cache=None, watcher=None):
        self.guest = guest
        if cache is None:
            cache = LRUCache(cache_capacity)
        self.cache = cache
        if watcher is None:
            watcher = PageWatcher(page_size=32)
        self.watcher = watcher
        self.stats = {"hits": 0, "misses": 0,
                      "evictions": 0, "invalidations": 0}
        self._next_block_id = 0
        self._by_id = {}  # block ID -> Block (for invalidation)

    # block dispatch
    def fetch_block(self, pc):
        """Return the translated block for pc, translating on a miss."""
        block = self.cache.get(pc)
        if block is not None:
            self.stats["hits"] += 1
            return block
        self.stats["misses"] += 1
        return self._translate(pc)

    def _translate(self, pc):
        """Decode pc's basic block and add it to the cache.

        Eviction of a victim also unregisters its pages from the
        watcher, so the watcher never points at dead blocks.
        """
        start, insts, end = self.guest.decode_block(pc)
        block = Block(self._next_block_id, start, end, insts)
        self._next_block_id += 1
        self.watcher.register(block.block_id, start, end)
        self._by_id[block.block_id] = block

        evicted = self.cache.put(pc, block)
        if evicted is not None:
            _, victim = evicted
            self.stats["evictions"] += 1
            self._drop_block(victim)
        return block

    def _drop_block(self, block):
        """Remove a block from the cache and the page watcher."""
        self.watcher.unregister(block.block_id, block.start_pc, block.end_pc)
        self._by_id.pop(block.block_id, None)
        self.cache.remove(block.start_pc)

    # self-modifying code
    def _store(self, addr, value):
        """Guest store. The written byte is masked to 8 bits. Writes
        into the code region go through the SMC watcher, so stale
        blocks are evicted before the new bytes are read by any
        future translation."""
        self.guest.mem[addr] = value % 256
        if addr < self.guest.code_end:
            for block_id in self.watcher.on_write(addr):
                block = self._by_id.get(block_id)
                if block is not None:
                    self._drop_block(block)
                    self.stats["invalidations"] += 1

    # execution
    def run(self, max_steps=100_000):
        """Execute the guest starting at pc = 0 until it HALTs.

        Returns the register file. Raises RuntimeError if the
        guest performs max_steps block dispatches without halting.
        """
        regs = self.guest.regs
        mem = self.guest.mem
        pc = 0
        for _ in range(max_steps):
            block = self.fetch_block(pc)
            for op, a, b in block.instructions:
                if op == OP_ADD:
                    regs[a] = (regs[a] + regs[b]) % 65536
                elif op == OP_LOAD:
                    regs[a] = mem[b]
                elif op == OP_STORE:
                    self._store(a, regs[b])
                elif op == OP_SUB:
                    regs[a] = (regs[a] - regs[b]) % 65536
                elif op == OP_JMP:
                    pc = a
                    break
                elif op == OP_JZ:
                    pc = b if regs[a] == 0 else block.end_pc
                    break
                elif op == OP_HALT:
                    return regs
                else:
                    raise ValueError(f"unknown opcode 0x{op:02X}")
        raise RuntimeError("guest did not halt within the step limit")
