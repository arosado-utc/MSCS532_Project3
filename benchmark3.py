"""
MSCS 532 - Project (Phase 3)
EmuCache benchmark harness: the Phase 3 structures against the
Phase 1/2 baselines.

Run:  python3 benchmark3.py

Output:
    results3/capacity_compare.csv  one row per (capacity, cache)
    results3/levels_scaling.csv    one row per (levels, cache)
    results3/patch_storm.csv       one row per (cycles, watcher)
    plots3/history_vs_lru.png
    plots3/levels_scaling.png
    plots3/patch_storm_invalidations.png

Measurement notes:
    * Event counts (hits, misses, evictions, invalidations) are
      the primary metrics: they are exact and reproducible.
    * Wall time is secondary; the runs are small, so each point
      is timed once rather than averaged.
    * The capacity sweep runs to 24 blocks so it passes the
      point where the LRU cache can finally hold the whole
      working set (23 distinct blocks).
"""

import csv
import os
import time

from cache import LRUCache
from emulator import Emulator
from history import HistoryCache
from interval import IntervalWatcher
from smc import PageWatcher
from stress import level_chain, mixed_scan, patch_storm

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results3")
PLOTS = os.path.join(HERE, "plots3")

CAPACITIES = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24]
LEVELS = [2, 3, 4, 5, 6]
PATCH_CYCLES = [3, 5, 10, 20]
PATCH_REPS = 20
BASE_CAPACITY = 8

CACHE_CLASSES = [
    ("LRU", LRUCache),
    ("history", HistoryCache),
]


def run_case(builder, cache, watcher=None):
    """Run one workload once and return its statistics as a dict."""
    t0 = time.perf_counter()
    emu = Emulator(builder(), cache=cache, watcher=watcher)
    emu.run()
    wall_ms = (time.perf_counter() - t0) * 1000.0
    s = emu.stats
    fetches = s["hits"] + s["misses"]
    rate = 100.0 * s["hits"] / fetches if fetches else 0.0
    return {
        "fetches": fetches,
        "hits": s["hits"],
        "misses": s["misses"],
        "evictions": s["evictions"],
        "invalidations": s["invalidations"],
        "hit_rate_pct": round(rate, 2),
        "wall_ms": round(wall_ms, 3),
    }


def write_csv(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def sweep_capacity():
    """Hit rate vs cache capacity for LRU and the history cache."""
    rows = []
    for cap in CAPACITIES:
        for name, cls in CACHE_CLASSES:
            if name == "history" and cap < 2:
                continue  # HistoryCache needs capacity >= 2
            row = {"capacity": cap, "cache": name}
            row.update(run_case(mixed_scan, cls(cap)))
            rows.append(row)
    path = os.path.join(RESULTS, "capacity_compare.csv")
    write_csv(path,
              ["capacity", "cache", "fetches", "hits", "misses",
               "evictions", "invalidations", "hit_rate_pct", "wall_ms"],
              rows)
    return rows


def sweep_levels():
    """Fetches and hit rate as the level chain grows, at C = 8."""
    rows = []
    for lv in LEVELS:
        for name, cls in CACHE_CLASSES:
            row = {"levels": lv, "cache": name}
            row.update(run_case(lambda lv=lv: level_chain(lv),
                                cls(BASE_CAPACITY)))
            rows.append(row)
    path = os.path.join(RESULTS, "levels_scaling.csv")
    write_csv(path,
              ["levels", "cache", "fetches", "hits", "misses",
               "evictions", "invalidations", "hit_rate_pct", "wall_ms"],
              rows)
    return rows


def sweep_patch():
    """Invalidations vs patch frequency, C = 8, LRU cache."""
    rows = []
    watchers = [("page", lambda: PageWatcher(page_size=32)),
                ("interval", lambda: IntervalWatcher())]
    for cycles in PATCH_CYCLES:
        for name, build in watchers:
            row = {"cycles": cycles, "watcher": name}
            row.update(run_case(lambda c=cycles: patch_storm(c, PATCH_REPS),
                                LRUCache(BASE_CAPACITY), build()))
            rows.append(row)
    path = os.path.join(RESULTS, "patch_storm.csv")
    write_csv(path,
              ["cycles", "watcher", "fetches", "hits", "misses",
               "evictions", "invalidations", "hit_rate_pct", "wall_ms"],
              rows)
    return rows


def plot_capacity(rows):
    """Hit rate vs capacity: the two eviction policies."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_cache = {}
    for r in rows:
        by_cache.setdefault(r["cache"], []).append(r)
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    styles = {"LRU": ("tab:blue", "o"), "history": ("tab:red", "s")}
    for name in ("LRU", "history"):
        rs = sorted(by_cache[name], key=lambda r: r["capacity"])
        c, mk = styles[name]
        ax.plot([r["capacity"] for r in rs],
                [r["hit_rate_pct"] for r in rs],
                color=c, marker=mk, ms=4, label=name)
    ax.set_xlabel("Cache capacity (blocks)")
    ax.set_ylabel("Hit rate (%)")
    ax.set_title("mixed_scan: hit rate vs capacity")
    ax.set_xticks(CAPACITIES)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend()
    os.makedirs(PLOTS, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "history_vs_lru.png"))
    plt.close(fig)


def plot_levels(rows):
    """Fetches and wall time vs level count: linear scaling."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_cache = {}
    for r in rows:
        by_cache.setdefault(r["cache"], []).append(r)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.0), dpi=150)
    for name, c in (("LRU", "tab:blue"), ("history", "tab:red")):
        rs = sorted(by_cache[name], key=lambda r: r["levels"])
        xs = [r["levels"] for r in rs]
        a1.plot(xs, [r["fetches"] for r in rs], marker="o", ms=4,
                color=c, label=name)
        a2.plot(xs, [r["wall_ms"] for r in rs], marker="o", ms=4,
                color=c, label=name)
    a1.set_xlabel("Levels")
    a1.set_ylabel("Block fetches")
    a1.set_title("Fetches vs level count")
    a1.grid(alpha=0.3)
    a2.set_xlabel("Levels")
    a2.set_ylabel("Wall time (ms)")
    a2.set_title("Wall time vs level count")
    a2.grid(alpha=0.3)
    a1.legend()
    fig.tight_layout()
    os.makedirs(PLOTS, exist_ok=True)
    fig.savefig(os.path.join(PLOTS, "levels_scaling.png"))
    plt.close(fig)


def plot_patch(rows):
    """Invalidations vs patch frequency: page vs interval."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_watcher = {}
    for r in rows:
        by_watcher.setdefault(r["watcher"], []).append(r)
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    for name, c in (("page", "tab:orange"), ("interval", "tab:green")):
        rs = sorted(by_watcher[name], key=lambda r: r["cycles"])
        ax.plot([r["cycles"] for r in rs],
                [r["invalidations"] for r in rs],
                marker="o", ms=4, color=c, label=name)
    ax.set_xlabel("Patch cycles")
    ax.set_ylabel("Invalidations")
    ax.set_title("patch_storm: invalidations vs patch frequency")
    ax.grid(alpha=0.3)
    ax.legend()
    os.makedirs(PLOTS, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "patch_storm_invalidations.png"))
    plt.close(fig)


def main():
    print("== EmuCache Phase 3: benchmark harness ==\n")
    cap = sweep_capacity()
    lv = sweep_levels()
    patch = sweep_patch()

    print("--- capacity sweep (mixed_scan, C = 8) ---")
    for r in cap:
        if r["capacity"] == 8:
            print(f"  {r['cache']:8s} misses={r['misses']:3d} "
                  f"rate={r['hit_rate_pct']:6.2f}%")
    print()
    print("--- level chain (C = 8) ---")
    for r in lv:
        print(f"  levels={r['levels']} {r['cache']:8s} "
              f"fetches={r['fetches']:3d} misses={r['misses']:3d} "
              f"rate={r['hit_rate_pct']:6.2f}%")
    print()
    print("--- patch storm (C = 8) ---")
    for r in patch:
        print(f"  cycles={r['cycles']:2d} {r['watcher']:8s} "
              f"invalidations={r['invalidations']:3d} "
              f"misses={r['misses']:3d}")

    plot_capacity(cap)
    plot_levels(lv)
    plot_patch(patch)
    print()
    print(f"wrote {RESULTS}/capacity_compare.csv, levels_scaling.csv, "
          f"patch_storm.csv")
    print(f"wrote {PLOTS}/history_vs_lru.png, levels_scaling.png, "
          f"patch_storm_invalidations.png")


if __name__ == "__main__":
    main()
