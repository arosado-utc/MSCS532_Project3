"""
MSCS 532 - Project (Phase 1)
EmuCache: toy guest machine and block decoder.

A tiny register machine with a 3-byte instruction format
[op, a, b]. Memory is 256 bytes. By default addresses 0x00-0x3F
hold code and 0x40-0xFF hold data; the split is a parameter, so
a workload that needs more room for code can widen the code
region (the Phase 3 stress workloads use code_end = 0x80 with
data at 0x90 and up). Registers are 16 bits; LOAD/STORE move a
single byte between memory and a register (truncating on the
way in, masking on the way out), while ADD/SUB work on the full
16-bit register.

A BASIC BLOCK starts at a program counter and ends at, including,
the first instruction that transfers control (JMP, JZ, HALT).
Splitting guest code into basic blocks is the first step of a
dynamic recompiler's divide: each block is an independent unit
that can be translated and cached on its own.
"""

OP_ADD = 1    # r[a] = r[a] + r[b]
OP_LOAD = 2   # r[a] = mem[b]
OP_STORE = 3  # mem[a] = r[b]
OP_SUB = 4    # r[a] = r[a] - r[b]
OP_JMP = 5    # pc = a
OP_JZ = 6     # if r[a] == 0: pc = b
OP_HALT = 7   # stop

_BRANCH_OPS = {OP_JMP, OP_JZ, OP_HALT}
_KNOWN_OPS = {OP_ADD, OP_LOAD, OP_STORE, OP_SUB, OP_JMP, OP_JZ, OP_HALT}

CODE_END = 0x40  # first data address; code lives in [0, CODE_END)


class Guest:
    """Toy guest machine: 8 sixteen-bit registers, 256-byte memory."""

    def __init__(self, code, code_end=CODE_END):
        """code: iterable of (address, byte) pairs for the program.
        code_end: first data address; code lives in [0, code_end).
        Defaults to CODE_END, the Phase 1/2 split."""
        self.code_end = code_end
        self.mem = [0] * 256
        for addr, byte in code:
            if addr >= len(self.mem):
                raise ValueError(f"guest address 0x{addr:X} out of range")
            self.mem[addr] = byte
        self.regs = [0] * 8

    def decode_block(self, pc):
        """Decode a basic block starting at pc.

        Reads 3-byte instructions until (and including) the first
        branch, jump, or halt.

        Return: (start_pc, [(op, a, b), ...], end_pc) where end_pc
        is the first address past the block.
        """
        insts = []
        addr = pc
        while addr + 3 <= len(self.mem):
            op = self.mem[addr]
            if op not in _KNOWN_OPS:
                raise ValueError(
                    f"undecodable opcode 0x{op:02X} at guest address "
                    f"0x{addr:02X}")
            insts.append((op, self.mem[addr + 1], self.mem[addr + 2]))
            addr += 3
            if op in _BRANCH_OPS:
                break
        if not insts:
            raise ValueError(f"no code at guest address 0x{pc:X}")
        return pc, insts, addr
