"""
MSCS 532 - Project (Phase 3)
EmuCache: stress workloads for the Phase 3 structures.

Phase 2's workloads all fit in the original 64-byte code region.
Phase 3 widens it to [0x00, 0x80) (data moves to 0x90 and up) so
a workload can hold a hot core plus a long tail of one-shot
blocks, and two workloads are provided for each new structure:

  mixed_scan     - a four-block hot core interleaved with passes
                   of a one-shot stub chain; the workload the
                   history-based cache is built for.
  level_chain    - mixed_scan with a variable number of levels;
                   the scaling sweep.
  patch_storm    - a self-modifying loop that re-applies its
                   patch three times per phase; the SMC sweep
                   that compares page- and interval-granular
                   watching.

Each builder returns a fresh Guest, so runs are independent.
"""

from guest import (CODE_END, Guest, OP_ADD, OP_HALT, OP_JMP, OP_JZ,
                   OP_LOAD, OP_STORE, OP_SUB)

WIDE_CODE_END = 0x80   # Phase 3 code region; data starts at 0x90
_DATA_BASE = 0x90      # first data address for the wide workloads
_ONE = _DATA_BASE      # 0x90: holds 1
_FIVE = _DATA_BASE + 1  # 0x91: holds 5
_ROUNDS = _DATA_BASE + 2  # 0x92: countdown init (R)
_COUNT = _DATA_BASE + 3   # 0x93: live countdown (C)
_PASSES = _DATA_BASE + 4   # 0x94: remaining scan passes (P)


def mixed_scan(rounds=4, stubs=15, levels=3):
    """A hot core interleaved with one-shot level stubs.

    The core is a four-block countdown loop. Each round it
    scores +1 and +5, counting the C cell down to zero. When C
    hits zero the game walks a stub chain - one JMP block per
    stub, each visited once per pass - and the last stub reaches
    the RELOAD block, which resets C, decrements the pass
    counter, and either starts the next core round or halts.

    0x00: LOAD r1, 0x90   PRO  r1 = 1
    0x03: LOAD r5, 0x91        r5 = 5
    0x06: LOAD r2, 0x92        r2 = rounds
    0x09: STORE 0x93, r2       C = rounds
    0x0C: LOAD r3, 0x94        r3 = levels - 1
    0x0F: JMP  0x15
    0x15: LOAD r2, 0x93   D1   c = C
    0x18: JZ   r2, 0x30        c == 0 -> scan
    0x1B: ADD  r0, r1     P1   score += 1
    0x1E: JMP  0x21
    0x21: ADD  r0, r5     P2   score += 5
    0x24: JMP  0x27
    0x27: SUB  r2, r1     P3   c -= 1
    0x2A: STORE 0x93, r2
    0x2D: JMP  0x15
    0x30: JMP  0x33       S1   the stub chain, one JMP each
    ...
    S{stubs-1}: JMP RELOAD
    RELOAD: LOAD r3, 0x94 ; JZ r3, HALT ; LOAD r2, 0x92
            STORE 0x93, r2 ; SUB r3, r1 ; STORE 0x94, r3
            JMP 0x15
    HALT:   HALT

    RELOAD checks the pass counter before touching it: zero
    passes left halts the game, otherwise it resets C, decrements
    the counter, and re-enters the core.

    Block layout: the core is four blocks (D1, P1, P2, P3),
    the stubs are `stubs` one-shot blocks, and PRO, RELOAD, and
    HALT are single-visit blocks. RELOAD splits into two blocks
    at its pass-check branch: the check block and the reload
    body. The core runs `levels` times; each scan costs `stubs`
    stub fetches plus the RELOAD blocks it reaches (the final
    scan halts in the check block, so the count works out to
    `stubs + 2` either way). Fetches = 1 + levels*(4*rounds +
    stubs + 3), and score = levels * rounds * 6.
    """
    if rounds < 1 or rounds > 0xFF:
        raise ValueError("rounds must fit in one guest byte")
    if stubs < 1:
        raise ValueError("stubs must be >= 1")
    if levels < 2:
        raise ValueError("levels must be >= 2")
    if levels - 1 > 0xFF:
        raise ValueError("levels must fit in one guest byte + 1")
    code_end = WIDE_CODE_END
    if 0x30 + 3 * stubs + 24 > code_end:
        raise ValueError("stubs exceed the code region")
    code = [
        (0x00, OP_LOAD), (0x01, 1), (0x02, _ONE),
        (0x03, OP_LOAD), (0x04, 5), (0x05, _FIVE),
        (0x06, OP_LOAD), (0x07, 2), (0x08, _ROUNDS),
        (0x09, OP_STORE), (0x0A, _COUNT), (0x0B, 2),
        (0x0C, OP_LOAD), (0x0D, 3), (0x0E, _PASSES),
        (0x0F, OP_JMP), (0x10, 0x15), (0x11, 0),
        (0x15, OP_LOAD), (0x16, 2), (0x17, _COUNT),
        (0x18, OP_JZ), (0x19, 2), (0x1A, 0x30),
        (0x1B, OP_ADD), (0x1C, 0), (0x1D, 1),
        (0x1E, OP_JMP), (0x1F, 0x21), (0x20, 0),
        (0x21, OP_ADD), (0x22, 0), (0x23, 5),
        (0x24, OP_JMP), (0x25, 0x27), (0x26, 0),
        (0x27, OP_SUB), (0x28, 2), (0x29, 1),
        (0x2A, OP_STORE), (0x2B, _COUNT), (0x2C, 2),
        (0x2D, OP_JMP), (0x2E, 0x15), (0x2F, 0),
    ]
    for i in range(stubs):
        addr = 0x30 + 3 * i
        target = 0x30 + 3 * (i + 1) if i < stubs - 1 else 0x30 + 3 * stubs
        code += [(addr, OP_JMP), (addr + 1, target), (addr + 2, 0)]
    r0 = 0x30 + 3 * stubs
    code += [
        (r0 + 0, OP_LOAD), (r0 + 1, 3), (r0 + 2, _PASSES),
        (r0 + 3, OP_JZ), (r0 + 4, 3), (r0 + 5, r0 + 21),
        (r0 + 6, OP_LOAD), (r0 + 7, 2), (r0 + 8, _ROUNDS),
        (r0 + 9, OP_STORE), (r0 + 10, _COUNT), (r0 + 11, 2),
        (r0 + 12, OP_SUB), (r0 + 13, 3), (r0 + 14, 1),
        (r0 + 15, OP_STORE), (r0 + 16, _PASSES), (r0 + 17, 3),
        (r0 + 18, OP_JMP), (r0 + 19, 0x15), (r0 + 20, 0),
        (r0 + 21, OP_HALT), (r0 + 22, 0), (r0 + 23, 0),
    ]
    data = [(_ONE, 1), (_FIVE, 5), (_ROUNDS, rounds),
            (_COUNT, rounds), (_PASSES, levels - 1)]
    return Guest(code + data, code_end=code_end)


def level_chain(levels, rounds=4, stubs=15):
    """A chain of `levels` game levels.

    Same program as mixed_scan, but the number of levels is the
    parameter under study: each level is one core round pass
    plus one walk of the stub chain, so the one-shot stub
    fetches grow linearly with the chain length. The scaling
    sweep in benchmark3.py varies this and checks that fetches
    and wall time grow linearly."""
    return mixed_scan(rounds=rounds, stubs=stubs, levels=levels)


def patch_storm(cycles, reps):
    """One loop, many phases, and a patch that is applied three
    times per phase.

    The loop body always reads its increment register from byte
    0x17, the same self-modifying trick as Phase 2's
    patch_game. The difference is the patch routine: instead of
    one store it re-applies the patch three times in a row (a
    field hotfix re-asserting itself), so each phase boundary
    is three SMC writes. Page-granular watching invalidates
    every cached block on the written page for each of the
    three writes; interval watching invalidates exactly the
    one block that covers 0x17.

    0x00: LOAD r1, 0x50   r1 = 1
    0x03: LOAD r5, 0x51   r5 = 5
    0x06: LOAD r7, 0x52   r7 = 11
    0x09: LOAD r2, 0x53   r2 = reps
    0x0C: LOAD r3, 0x54   r3 = cycles - 1
    0x0F: JMP  0x15
    0x15: ADD  r0, r[0x17]  loop body; 0x17 is the register
    0x18: SUB  r2, r1       index of the increment
    0x1B: JZ   r2, 0x21   phase done -> patch check
    0x1E: JMP  0x15
    0x21: JZ   r3, 0x3F   no patches left -> halt
    0x24: SUB  r3, r1     patches remaining -= 1
    0x27: LOAD r4, 0x55   increment register index
    0x2A: STORE 0x17, r4  *** SMC 1 ***
    0x2D: STORE 0x17, r4  *** SMC 2, re-apply ***
    0x30: STORE 0x17, r4  *** SMC 3, re-apply ***
    0x33: LOAD r6, 0x56   index after next
    0x36: STORE 0x55, r6  update the "next" cell
    0x39: LOAD r2, 0x53   reload countdown
    0x3C: JMP  0x15
    0x3F: HALT

    The first patch switches the increment to r5 = 5 and every
    later patch to r7 = 11, so score = reps + (cycles >= 2 ?
    5*reps : 0) + (cycles >= 3 ? 11*reps*(cycles-2) : 0).

    Invalidation counts: each burst does 3 writes to the same
    byte, but only the first one of a burst finds the affected
    blocks still registered (the drop unregisters them), so the
    per-burst cost is the set of blocks that cover 0x17. The
    interval watcher pays one block per burst (the loop body)
    and the page watcher pays the body plus the loopback, which
    shares page 0 and is hot - it is re-missed after every
    burst. With cycles >= 2: interval = cycles - 1, page =
    3 + 2*(cycles - 2).
    """
    if cycles < 1:
        raise ValueError("cycles must be >= 1")
    if cycles - 1 > 0xFF:
        raise ValueError("cycles must fit in one guest byte + 1")
    if reps < 1 or reps > 0xFF:
        raise ValueError("reps must fit in one guest byte")
    code_end = 0x48   # code runs to 0x42; data stays at 0x50 and up
    code = [
        (0x00, OP_LOAD), (0x01, 1), (0x02, 0x50),
        (0x03, OP_LOAD), (0x04, 5), (0x05, 0x51),
        (0x06, OP_LOAD), (0x07, 7), (0x08, 0x52),
        (0x09, OP_LOAD), (0x0A, 2), (0x0B, 0x53),
        (0x0C, OP_LOAD), (0x0D, 3), (0x0E, 0x54),
        (0x0F, OP_JMP), (0x10, 0x15), (0x11, 0),
        (0x15, OP_ADD), (0x16, 0), (0x17, 1),
        (0x18, OP_SUB), (0x19, 2), (0x1A, 1),
        (0x1B, OP_JZ), (0x1C, 2), (0x1D, 0x21),
        (0x1E, OP_JMP), (0x1F, 0x15), (0x20, 0),
        (0x21, OP_JZ), (0x22, 3), (0x23, 0x3F),
        (0x24, OP_SUB), (0x25, 3), (0x26, 1),
        (0x27, OP_LOAD), (0x28, 4), (0x29, 0x55),
        (0x2A, OP_STORE), (0x2B, 0x17), (0x2C, 4),
        (0x2D, OP_STORE), (0x2E, 0x17), (0x2F, 4),
        (0x30, OP_STORE), (0x31, 0x17), (0x32, 4),
        (0x33, OP_LOAD), (0x34, 6), (0x35, 0x56),
        (0x36, OP_STORE), (0x37, 0x55), (0x38, 6),
        (0x39, OP_LOAD), (0x3A, 2), (0x3B, 0x53),
        (0x3C, OP_JMP), (0x3D, 0x15), (0x3E, 0),
        (0x3F, OP_HALT), (0x40, 0), (0x41, 0),
    ]
    data = [(0x50, 1), (0x51, 5), (0x52, 11), (0x53, reps),
            (0x54, cycles - 1), (0x55, 5), (0x56, 7)]
    return Guest(code + data, code_end=code_end)


def main():
    for name, g in (("mixed_scan", mixed_scan()),
                    ("level_chain(4)", level_chain(4)),
                    ("patch_storm(5, 20)", patch_storm(5, 20))):
        print(f"{name:18s} code_end=0x{g.code_end:02X}  "
              f"memory bytes set: "
              f"{sum(1 for b in g.mem if b)}")


if __name__ == "__main__":
    main()
