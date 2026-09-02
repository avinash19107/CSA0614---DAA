"""
Conflict-Free Exam Timetable Scheduling via Graph Colouring.

Implements:
  1. Plain Backtracking graph colouring.
  2. Pruning-enhanced Backtracking graph colouring using
     Most-Constrained-Vertex (MCV / degree) ordering + Forward Checking.

Each course = vertex. Edge (u, v) exists if courses u and v share
at least one common student (i.e. cannot be scheduled in the same slot).
A valid colouring with k colours = valid exam timetable with k time slots.
"""

import random
import time
import tracemalloc
from dataclasses import dataclass, field


# --------------------------------------------------------------------------
# 1. Conflict graph generation (synthetic, Erdos-Renyi style with a course/
#    student-set interpretation)
# --------------------------------------------------------------------------

def generate_conflict_graph(n_courses: int, edge_prob: float, seed: int = 42):
    """
    Generate a synthetic course-conflict graph.

    Each pair of courses is connected with probability `edge_prob`,
    modelling the chance that they share at least one common student.
    This is the standard method used when no real enrolment dataset
    is available (documented per assignment constraints).

    Returns: adjacency list (dict[int, set[int]])
    """
    rng = random.Random(seed)
    adj = {i: set() for i in range(n_courses)}
    for u in range(n_courses):
        for v in range(u + 1, n_courses):
            if rng.random() < edge_prob:
                adj[u].add(v)
                adj[v].add(u)
    return adj


def greedy_upper_bound(adj):
    """Welsh-Powell style greedy colouring to obtain a practical k (upper
    bound on chromatic number) to attempt backtracking with."""
    order = sorted(adj.keys(), key=lambda v: -len(adj[v]))
    colour = {}
    for v in order:
        used = {colour[nb] for nb in adj[v] if nb in colour}
        c = 0
        while c in used:
            c += 1
        colour[v] = c
    return max(colour.values()) + 1 if colour else 0


# --------------------------------------------------------------------------
# 2. Metrics container
# --------------------------------------------------------------------------

@dataclass
class RunResult:
    algorithm: str
    n_courses: int
    edge_prob: float
    n_edges: int
    k_colours: int
    solved: bool
    steps: int = 0          # number of assignment attempts / recursive calls
    conflicts: int = 0      # number of colour-conflicts detected & rejected
    time_sec: float = 0.0
    peak_memory_kb: float = 0.0
    colouring: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# 3. Plain Backtracking (tries colours 0..k-1 in order, backtracks on conflict)
# --------------------------------------------------------------------------

def plain_backtracking(adj, k, time_limit=15.0):
    n = len(adj)
    vertices = list(adj.keys())
    colour = {}
    stats = {"steps": 0, "conflicts": 0}
    start = time.perf_counter()
    timed_out = [False]

    def is_safe(v, c):
        for nb in adj[v]:
            if colour.get(nb) == c:
                return False
        return True

    def backtrack(idx):
        if timed_out[0]:
            return False
        if time.perf_counter() - start > time_limit:
            timed_out[0] = True
            return False
        if idx == n:
            return True
        v = vertices[idx]
        for c in range(k):
            stats["steps"] += 1
            if is_safe(v, c):
                colour[v] = c
                if backtrack(idx + 1):
                    return True
                del colour[v]
            else:
                stats["conflicts"] += 1
        return False

    tracemalloc.start()
    solved = backtrack(0)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    elapsed = time.perf_counter() - start

    return RunResult(
        algorithm="Plain Backtracking",
        n_courses=n, edge_prob=0.0,
        n_edges=sum(len(s) for s in adj.values()) // 2,
        k_colours=k, solved=solved and not timed_out[0],
        steps=stats["steps"], conflicts=stats["conflicts"],
        time_sec=elapsed, peak_memory_kb=peak / 1024,
        colouring=dict(colour) if solved else {},
    )


# --------------------------------------------------------------------------
# 4. Pruning-enhanced Backtracking: Most-Constrained-Vertex (MCV / highest
#    saturation degree, i.e. DSATUR-style) ordering + Forward Checking
# --------------------------------------------------------------------------

def pruning_backtracking(adj, k, time_limit=15.0):
    n = len(adj)
    colour = {}
    # domains[v] = set of colours still legal for v given current partial assignment
    domains = {v: set(range(k)) for v in adj}
    stats = {"steps": 0, "conflicts": 0}
    start = time.perf_counter()
    timed_out = [False]

    def select_most_constrained_vertex():
        # MCV: fewest remaining legal colours (most constrained / saturation degree).
        # Tie-break with highest degree (most constraining variable).
        uncoloured = [v for v in adj if v not in colour]
        return min(
            uncoloured,
            key=lambda v: (len(domains[v]), -len(adj[v]))
        )

    def forward_check(v, c):
        """After assigning v=c, remove c from neighbours' domains.
        Returns list of (neighbour, removed) for undo, or None if a
        neighbour's domain becomes empty (dead end -> prune)."""
        removed = []
        for nb in adj[v]:
            if nb not in colour and c in domains[nb]:
                domains[nb].discard(c)
                removed.append(nb)
                if not domains[nb]:
                    return removed, False
        return removed, True

    def undo_forward_check(c, removed):
        for nb in removed:
            domains[nb].add(c)

    def backtrack(count):
        if timed_out[0]:
            return False
        if time.perf_counter() - start > time_limit:
            timed_out[0] = True
            return False
        if count == n:
            return True
        v = select_most_constrained_vertex()
        for c in sorted(domains[v]):
            stats["steps"] += 1
            colour[v] = c
            removed, ok = forward_check(v, c)
            if ok:
                if backtrack(count + 1):
                    return True
            else:
                stats["conflicts"] += 1
            undo_forward_check(c, removed)
            del colour[v]
        return False

    tracemalloc.start()
    solved = backtrack(0)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    elapsed = time.perf_counter() - start

    return RunResult(
        algorithm="Pruning Backtracking (MCV + Forward Checking)",
        n_courses=n, edge_prob=0.0,
        n_edges=sum(len(s) for s in adj.values()) // 2,
        k_colours=k, solved=solved and not timed_out[0],
        steps=stats["steps"], conflicts=stats["conflicts"],
        time_sec=elapsed, peak_memory_kb=peak / 1024,
        colouring=dict(colour) if solved else {},
    )


def validate_colouring(adj, colouring):
    """Sanity check: no adjacent vertices share a colour."""
    for u in adj:
        for v in adj[u]:
            if colouring.get(u) == colouring.get(v):
                return False
    return True
