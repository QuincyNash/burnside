#!/usr/bin/env sage

from __future__ import annotations

import os
import time
import argparse
from typing import List, Tuple, Union, Optional
from tqdm import tqdm
from sympy import primerange

from sage.all import libgap  # type: ignore[import]
from sage.parallel.decorate import fork

from tom_store import TomStore


# Helper to add arguments to argparse
def _add_argument(parser: argparse.ArgumentParser, name: str, default: int) -> None:
    parser.add_argument(
        f"-{name}",
        type=int,
        default=default,
        help=f"Maximum n for {name} groups (default: {default})",
    )


@fork(timeout=5)
def _compute_description(tom, i: int) -> str:
    rep = libgap.RepresentativeTom(tom, i)
    return str(libgap.StructureDescription(rep))


# Compute table of marks using GAP
def _marks_from_gap_group_or_string(
    gap_group: Union[str, object], skip_descriptions=True
) -> Optional[Tuple[List[List[int]], List[int], List[str], int]]:
    tom = libgap.TableOfMarks(gap_group)
    if str(tom) == "fail":
        return None

    # Convert matrix to a list of lists
    gap_M = libgap.MatTom(tom)
    marks = [[int(x) for x in row] for row in gap_M]

    orders = [int(o) for o in libgap.OrdersTom(tom)]

    subgroup_names: List[str] = []
    if skip_descriptions:
        for i in range(1, len(orders) + 1):
            subgroup_names.append(f"(order: {orders[i - 1]})")
    else:
        # Get subgroup names using GAP's StructureDescription for each subgroup representative
        for i in tqdm(range(1, len(orders) + 1), desc="Getting subgroup names"):
            desc = _compute_description(tom, i)

            # If timeout occured, fallback to order of subgroup for description
            if desc == "NO DATA (timed out)":
                desc = f"(order: {orders[i - 1]})"
            else:
                # Remove spaces from GAP description for consistency
                desc = desc.replace(" ", "")

            subgroup_names.append(desc)

    # Search for duplicates and qualify with index if necessary
    name_counts = {}
    for i, name in enumerate(subgroup_names):
        if name in name_counts:
            name_counts[name] += 1
            subgroup_names[i] = f"{name} #{name_counts[name]}"
        else:
            name_counts[name] = 1

    return marks, orders, subgroup_names, orders[-1]  # Last order is the group order


# Store TOM in sqlite store
def try_store(store: TomStore, name: str, gap_group: object = None, source="computed"):
    if name in store:
        print(f"Skipping {name} (already in store)")
        return

    try:
        # Load group if defined, otherwise try to compute from name
        result = _marks_from_gap_group_or_string(
            gap_group if gap_group is not None else name, False
        )
        if result is None:
            print(f"Error: {name} (GAP failed to compute TOM)")
            return
        marks, orders, subgroup_names, order = result
        store.put(name, marks, orders, subgroup_names, order)
        print(f"Success: {name} (order {order}, rank {len(marks)})")
    except Exception as e:
        print(f"Error: {name} ({e})")


def build_cyclic(store: TomStore):
    print(f"Cyclic groups C_n for n = 1 to {MAX_CYCLIC}")
    for n in range(1, MAX_CYCLIC + 1):
        try_store(store, f"C{n}", libgap.CyclicGroup(n))


def build_dihedral(store: TomStore):
    print(f"Dihedral groups D_n for n = 2 to {MAX_DIHEDRAL}")
    for n in range(2, MAX_DIHEDRAL + 1):
        try_store(store, f"D{n}", libgap.DihedralGroup(2 * n))


def build_symmetric(store: TomStore):
    print(f"Symmetric groups S_n for n = 1 to {MAX_SYMMETRIC}")
    for n in range(1, MAX_SYMMETRIC + 1):
        try_store(store, f"S{n}", libgap.SymmetricGroup(n))


def build_alternating(store: TomStore):
    print(f"Alternating groups A_n for n = 3 to {MAX_ALTERNATING}")
    for n in range(3, MAX_ALTERNATING + 1):
        try_store(store, f"A{n}", libgap.AlternatingGroup(n))


def build_elementary_abelian(store: TomStore):
    primes = list(primerange(2, MAX_PRIME + 1))

    print(f"Elementary abelian C_p x C_p up to p = {primes[-1]}")

    for p in primes:
        G = libgap.DirectProduct(libgap.CyclicGroup(p), libgap.CyclicGroup(p))
        try_store(store, f"C{p}xC{p}", G)


# Special named groups: V4, Q8
def build_named(store: TomStore):
    print("Special named groups")
    try_store(
        store, "V4", libgap.DirectProduct(libgap.CyclicGroup(2), libgap.CyclicGroup(2))
    )
    try_store(store, "Q8", libgap.QuaternionGroup(8))
    try_store(
        store,
        "Q8xC2",
        libgap.DirectProduct(libgap.QuaternionGroup(8), libgap.CyclicGroup(2)),
    )
    try_store(
        store,
        "V4xC4",
        libgap.DirectProduct(
            libgap.DirectProduct(libgap.CyclicGroup(2), libgap.CyclicGroup(2)),
            libgap.CyclicGroup(4),
        ),
    )
    try_store(
        store,
        "D8xC2",
        libgap.DirectProduct(libgap.DihedralGroup(8), libgap.CyclicGroup(2)),
    )
    try_store(
        store,
        "Q16",
        libgap.DirectProduct(libgap.DihedralGroup(8), libgap.CyclicGroup(4)),
    )


def build_cyclic_products(store: TomStore):
    print(f"Products of cyclic groups C_m x C_n for m,n = 1 to {MAX_CYCLIC}")

    # Make sure m > n
    for n in range(1, MAX_CYCLIC + 1):
        for m in range(n, MAX_CYCLIC + 1):
            G = libgap.DirectProduct(libgap.CyclicGroup(m), libgap.CyclicGroup(n))
            try_store(store, f"C{m}xC{n}", G)


def build_tomlib(store: TomStore):
    print("Groups in TomLib")
    for name in libgap.AllLibTomNames():
        try:
            try_store(store, str(name), source="TomLib")
        except Exception as e:
            print(f"Error: {str(name)} ({e})")


if __name__ == "__main__":
    # Make sure to load TomLib
    libgap.LoadPackage("tomlib")

    parser = argparse.ArgumentParser(description="Build cache of tables of marks.")

    # Add arguments for each family of groups
    _add_argument(parser, "cyclic", 100)
    _add_argument(parser, "dihedral", 100)
    _add_argument(parser, "symmetric", 8)
    _add_argument(parser, "alternating", 8)
    _add_argument(parser, "prime", 100)

    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Clear existing cache and recompute everything",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=5,
        help="Timeout in seconds for GAP calls when computing subgroup names (default: 5)",
    )
    parser.add_argument(
        "--db",
        default="tom_cache.sqlite",
        help="Path to sqlite database (default: tom_cache.sqlite)",
    )

    # Load arguments
    args = parser.parse_args()
    MAX_CYCLIC: int = args.cyclic
    MAX_DIHEDRAL: int = args.dihedral
    MAX_SYMMETRIC: int = args.symmetric
    MAX_ALTERNATING: int = args.alternating
    MAX_PRIME: int = args.prime
    REBUILD: bool = args.rebuild
    TIMEOUT_SECONDS: int = args.timeout
    DB_FILE: str = args.db

    # Load TomLib globally
    libgap.LoadPackage("tomlib")

    if REBUILD and os.path.exists(DB_FILE):
        os.remove(DB_FILE)
        print("Cleared existing cache.")

    store = TomStore(DB_FILE)
    tomlib_names = set(str(n) for n in libgap.AllLibTomNames())

    t0 = time.time()

    build_cyclic(store)
    build_dihedral(store)
    build_symmetric(store)
    build_alternating(store)
    build_elementary_abelian(store)
    build_cyclic_products(store)
    build_named(store)
    build_tomlib(store)

    store.commit()
    elapsed = time.time() - t0

    print(f"\nDone in {elapsed:.1f}s.")
    for row in store.stats():
        print(f"Entry count: {row['entries']}")
        print(f"Compressed size: {row['compressed_bytes'] / 1e6:.3f} MB")
    store.close()
