import csv
import sys
from scheduler import (
    generate_conflict_graph, greedy_upper_bound,
    plain_backtracking, pruning_backtracking, validate_colouring
)

# Institution-size proxy: number of courses.
COURSE_COUNTS = [10, 15, 20, 25, 30, 40, 50, 65, 80]
DENSITIES = {"sparse": 0.10, "medium": 0.30, "dense": 0.55}
SEED_TRIALS = [1, 2, 3]   # repeat each config 3x with different seeds, average

TIME_LIMIT = 8.0  # seconds per run before declaring timeout

rows = []
run_id = 0
total = len(COURSE_COUNTS) * len(DENSITIES) * len(SEED_TRIALS) * 2

for density_name, p in DENSITIES.items():
    for n in COURSE_COUNTS:
        for seed in SEED_TRIALS:
            adj = generate_conflict_graph(n, p, seed=seed)
            k = greedy_upper_bound(adj)
            n_edges = sum(len(s) for s in adj.values()) // 2

            for algo_name, fn in [
                ("plain", plain_backtracking),
                ("pruning", pruning_backtracking),
            ]:
                run_id += 1
                res = fn(adj, k, time_limit=TIME_LIMIT)
                ok = validate_colouring(adj, res.colouring) if res.solved else None
                rows.append({
                    "run_id": run_id,
                    "density_class": density_name,
                    "edge_prob": p,
                    "n_courses": n,
                    "n_edges": n_edges,
                    "k_colours": k,
                    "seed": seed,
                    "algorithm": algo_name,
                    "solved": res.solved,
                    "valid": ok,
                    "steps": res.steps,
                    "conflicts": res.conflicts,
                    "time_sec": round(res.time_sec, 6),
                    "peak_memory_kb": round(res.peak_memory_kb, 3),
                })
                print(f"[{run_id}/{total}] {density_name:6s} n={n:3d} k={k:2d} "
                      f"seed={seed} {algo_name:8s} solved={res.solved} "
                      f"steps={res.steps:7d} time={res.time_sec:.5f}s "
                      f"mem={res.peak_memory_kb:.1f}KB", file=sys.stderr)

with open("results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"\nDone. {len(rows)} rows written to results.csv", file=sys.stderr)
