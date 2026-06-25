#!/usr/bin/env sage
from __future__ import annotations

import argparse
import re
import time
from collections import deque
from typing import List, Optional, Tuple

from sage.all import libgap  # type: ignore[import]
from sage.parallel.decorate import fork

from tom_store import TomStore
from build_cache import _marks_from_gap_group_or_string


# ── tunables ──────────────────────────────────────────────────────────
DEFAULT_MAX_RANK = 500  # skip groups with more subgroup classes than this
DESCRIPTION_TIMEOUT = 15  # seconds allowed per StructureDescription call


@fork(timeout=DESCRIPTION_TIMEOUT)
def _structure_description(rep) -> str:
    """Return GAP's StructureDescription string for a group representative."""
    return str(libgap.StructureDescription(rep))


# ─────────────────────────────────────────────────────────────────────
# Obtaining a GAP TOM object for a name already in the store
# ─────────────────────────────────────────────────────────────────────


def _gap_tom_from_name(name: str):
    """Attempt a direct TomLib lookup.  Returns a GAP TOM object or None."""
    tom = libgap.TableOfMarks(name)
    return None if str(tom) == "fail" else tom


_CpxCq_RE = re.compile(r"^C(\d+)xC(\d+)$")


def _reconstruct_gap_group(name: str):
    """
    Reconstruct a GAP group from a store key using the naming conventions
    established in build_cache.py.  Returns a libgap group object or None.
    """
    # Cyclic: C1, C12, …
    if m := re.fullmatch(r"C(\d+)", name):
        return libgap.CyclicGroup(int(m.group(1)))

    # Dihedral: D3, D12, …  (store key Dn  ↔  GAP DihedralGroup(2n))
    if m := re.fullmatch(r"D(\d+)", name):
        return libgap.DihedralGroup(2 * int(m.group(1)))

    # Symmetric: S3, S4, …
    if m := re.fullmatch(r"S(\d+)", name):
        return libgap.SymmetricGroup(int(m.group(1)))

    # Alternating: A3, A4, …
    if m := re.fullmatch(r"A(\d+)", name):
        return libgap.AlternatingGroup(int(m.group(1)))

    # Direct product of two cyclics: C2xC3, C5xC5, …
    if m := _CpxCq_RE.fullmatch(name):
        p, q = int(m.group(1)), int(m.group(2))
        return libgap.DirectProduct(libgap.CyclicGroup(p), libgap.CyclicGroup(q))

    # Special named aliases from build_cache.py
    if name == "V4":
        return libgap.DirectProduct(libgap.CyclicGroup(2), libgap.CyclicGroup(2))
    if name == "Q8":
        return libgap.QuaternionGroup(8)

    return None


def _gap_tom(name: str):
    """
    Best-effort retrieval of a GAP TOM object for a stored group name.
    Tries TomLib first, then falls back to group reconstruction.
    Returns None if neither approach succeeds.
    """
    tom = _gap_tom_from_name(name)
    if tom is not None:
        return tom

    grp = _reconstruct_gap_group(name)
    if grp is None:
        return None

    tom = libgap.TableOfMarks(grp)
    return None if str(tom) == "fail" else tom


# ─────────────────────────────────────────────────────────────────────
# Per-group processing
# ─────────────────────────────────────────────────────────────────────


def _process_group(
    name: str,
    store: TomStore,
    max_rank: int,
    dry_run: bool,
) -> List[str]:
    """
    For the group *name* already in the store, iterate over its conjugacy
    classes of subgroups and ensure each one is also in the store.

    Returns the list of names that were added (or would be added in dry-run
    mode).  Only names absent from the store at call time are returned, so
    the caller can enqueue them for further processing.
    """
    data = store.get(name)
    if data is None:
        return []

    if data.rank > max_rank:
        print(f"  [skip]  {name}  rank={data.rank} > max_rank={max_rank}")
        return []

    tom = _gap_tom(name)
    if tom is None:
        print(f"  [skip]  {name}  cannot obtain GAP TOM")
        return []

    print(f"  [scan]  {name}  (order={data.group_order}, rank={data.rank})")
    new_names: List[str] = []

    for i in range(1, data.rank + 1):
        # ── get a concrete representative for subgroup class i ──
        try:
            rep = libgap.RepresentativeTom(tom, i)
        except Exception as exc:
            print(f"    [warn]  subgroup {i}: RepresentativeTom failed — {exc}")
            continue

        # ── canonical name via StructureDescription ──
        desc = _structure_description(rep)
        if desc == "NO DATA (timed out)":
            print(f"    [warn]  subgroup {i}: StructureDescription timed out")
            continue

        sub_name = desc.replace(" ", "")  # "C2 x C3" → "C2xC3"

        if sub_name in store:
            continue  # already present — nothing to do

        if dry_run:
            print(f"    [miss]  {sub_name}")
            new_names.append(sub_name)
            continue

        # ── compute and store ──
        try:
            result = _marks_from_gap_group_or_string(rep, skip_descriptions=False)
            if result is None:
                print(f"    [fail]  {sub_name}: TableOfMarks returned fail")
                continue
            marks, orders, sub_sub_names, group_order = result
            store.put(sub_name, marks, orders, sub_sub_names, group_order)
            print(f"    [add]   {sub_name}  (order={group_order}, rank={len(marks)})")
            new_names.append(sub_name)
        except Exception as exc:
            print(f"    [fail]  {sub_name}: {exc}")

    if not dry_run:
        store.set_closed(name)
        print(f"  [done]  {name} marked closed")

    return new_names


# ─────────────────────────────────────────────────────────────────────
# Main closure loop  (BFS over the store)
# ─────────────────────────────────────────────────────────────────────


def build_closure(store: TomStore, max_rank: int, dry_run: bool) -> None:
    """
    BFS starting from every name currently in the store.  Newly added
    entries are enqueued so their own subgroups are checked in turn,
    guaranteeing full transitive closure.
    """
    visited: set[str] = set()
    queue: deque[str] = deque(
        name for name in store.names() if not store.is_closed(name)
    )
    added_total = 0
    pending_commit = 0
    COMMIT_INTERVAL = 1  # flush to disk every this many new entries

    while queue:
        name = queue.popleft()
        if name in visited:
            continue
        visited.add(name)

        new_names = _process_group(name, store, max_rank, dry_run)
        added_total += len(new_names)

        if new_names and not dry_run:
            pending_commit += len(new_names)
            queue.extend(new_names)
            if pending_commit >= COMMIT_INTERVAL:
                store.commit()
                pending_commit = 0

    if not dry_run:
        store.commit()

    action = "discovered (dry-run)" if dry_run else "added"
    print(f"\nSubgroup entries {action}: {added_total}")


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    libgap.LoadPackage("tomlib")

    parser = argparse.ArgumentParser(
        description=(
            "Extend TomStore so that it is closed under taking subgroups: "
            "for every group G in the store, every H ≤ G will also be in the store."
        )
    )
    parser.add_argument(
        "--db",
        default="tom_cache.sqlite",
        help="Path to the SQLite database (default: tom_cache.sqlite)",
    )
    parser.add_argument(
        "--max-rank",
        type=int,
        default=DEFAULT_MAX_RANK,
        help=f"Skip groups whose TOM rank exceeds this value (default: {DEFAULT_MAX_RANK})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report missing subgroups without modifying the database",
    )
    args = parser.parse_args()

    store = TomStore(args.db)
    t0 = time.time()

    banner = "Subgroup closure pass" + (
        "  [DRY RUN — no writes]" if args.dry_run else ""
    )
    print("=" * 60)
    print(banner)
    print("=" * 60)

    build_closure(store, args.max_rank, args.dry_run)

    elapsed = time.time() - t0
    print(f"Elapsed: {elapsed:.1f}s")

    if not args.dry_run:
        for row in store.stats():
            print(f"Entry count:      {row['entries']}")
            print(f"Compressed size:  {row['compressed_bytes'] / 1e6:.3f} MB")

    store.close()
