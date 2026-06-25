#!/usr/bin/env python3

import json
import sqlite3
import zlib
from array import array
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

# Useful helper
Triple = Tuple[int, int, int]


# Helper to convert matrix to its nonzero (row, col, value) triples
# Row first following standard math convention
def triples_from_dense(rows: Sequence[Sequence[int]]) -> List[Triple]:
    return [
        (i, j, int(v))
        for i, row in enumerate(rows)
        for j, v in enumerate(row)
        if v != 0
    ]


# Stores all marks information for a group, including triples, subgroup orders, and group order
@dataclass
class MarksData:
    name: str
    group_order: int
    rank: int
    subgroup_orders: List[int]
    subgroup_names: List[str]
    triples: List[Triple]  # nonzero (row, col, value) entries only

    # Get dense matrix representation of the marks
    def dense(self) -> List[List[int]]:
        mat = [[0 for _ in range(self.rank)] for _ in range(self.rank)]
        for row, col, value in self.triples:
            mat[row][col] = value
        return mat

    # Get sparse row representation of the marks (each row is a dict of {col: value})
    def rows(self) -> List[Dict[int, int]]:
        rows: List[Dict[int, int]] = [dict() for _ in range(self.rank)]
        for row, col, value in self.triples:
            rows[row][col] = value
        return rows

    # Get sparse column representation of the marks (each column is a dict of {row: value}), equivalently rows of M^T
    def cols(self) -> List[Dict[int, int]]:
        cols: List[Dict[int, int]] = [dict() for _ in range(self.rank)]
        for row, col, value in self.triples:
            cols[col][row] = value
        return cols


# Schema for sqlite:
# blob stores compressed triples of (row, col, value)
# subgroup_orders is JSON-encoded list of ints
# subgroup_names also JSON-encoded list of strings
# group_order is stored as text to avoid 64-bit integer overflow
_SCHEMA = """
CREATE TABLE IF NOT EXISTS marks (
    name             TEXT PRIMARY KEY,
    group_order      TEXT NOT NULL,
    rank             INTEGER NOT NULL,
    subgroup_orders  TEXT NOT NULL,
    subgroup_names   TEXT NOT NULL,
    num_nonzero      INTEGER NOT NULL,
    closed           INTEGER NOT NULL DEFAULT 0,
    blob             BLOB NOT NULL
);
"""


# Handles sqlite storage and retrieval of TOM data, including caching in memory
class TomStore:
    def __init__(self, path: str = "tom_cache.sqlite"):
        self.path = path
        self._int_zlib_encoding = "q"  # 64-bit signed integers for zlib compression

        # Create sqlite database and table if it doesn't exist
        self._conn = sqlite3.connect(path)
        self._conn.execute(_SCHEMA)
        self._conn.commit()

        # Cache for TOM data for quick access
        self._mem: Dict[str, MarksData] = {}

    # Add entry to store, compressing triples to save space
    def put(
        self,
        name: str,
        marks: Sequence[Sequence[int]],
        subgroup_orders: Sequence[int],
        subgroup_names: Sequence[str],
        group_order: int,
        commit: bool = True,
    ) -> None:
        rank = len(marks)

        triples = triples_from_dense(marks)

        # Flatten triples and compress with zlib
        flat = [x for t in triples for x in t]
        blob = zlib.compress(array(self._int_zlib_encoding, flat).tobytes())

        self._conn.execute(
            "INSERT OR REPLACE INTO marks "
            "(name, group_order, rank, subgroup_orders, subgroup_names, num_nonzero, closed, blob) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                name,
                str(int(group_order)),
                rank,
                json.dumps(list(subgroup_orders)),  # Stored as JSON string
                json.dumps(list(subgroup_names or [])),  # Optional subgroup names
                len(triples),
                0,  # closed column, default to 0
                blob,
            ),
        )
        # Only commit to disk if requested
        if commit:
            self._conn.commit()

        # Refresh cache so proper version is loaded next time
        self._mem.pop(name, None)

    # Save changes to disk
    def commit(self) -> None:
        self._conn.commit()

    # Efficiently check if name exists in store, cache first
    def contains(self, name: str) -> bool:
        if name in self._mem:
            return True

        cur = self._conn.execute("SELECT 1 FROM marks WHERE name = ?", (name,))
        return cur.fetchone() is not None

    # Overload "in" operator
    def __contains__(self, name: str) -> bool:
        return self.contains(name)

    # Retrieve entry from store and cache in memory, return None if not found
    def get(self, name: str) -> Optional[MarksData]:
        # Check cache first
        if name in self._mem:
            return self._mem[name]

        # Fetch from db
        cur = self._conn.execute(
            "SELECT group_order, rank, subgroup_orders, subgroup_names, blob "
            "FROM marks WHERE name = ?",
            (name,),
        )
        row = cur.fetchone()
        if row is None:
            return None

        group_order, rank, subgroup_orders_json, subgroup_names_json, blob = row

        # Decompress blob to triples
        flat = array(self._int_zlib_encoding)
        flat.frombytes(zlib.decompress(blob))
        triples: List[Triple] = [
            (flat[k], flat[k + 1], flat[k + 2]) for k in range(0, len(flat), 3)
        ]

        # Create MarksData object and cache it
        data = MarksData(
            name=name,
            group_order=int(group_order),
            rank=rank,
            subgroup_orders=json.loads(subgroup_orders_json),
            subgroup_names=json.loads(subgroup_names_json),
            triples=triples,
        )
        self._mem[name] = data
        return data

    # Helper to set the closed status of a group in the store
    def set_closed(self, name: str, value: bool = True):
        self._conn.execute(
            "UPDATE marks SET closed = ? WHERE name = ?",
            (int(value), name),
        )
        self._conn.commit()
        self._mem.pop(name, None)

    # Helper to check if a group is marked as closed in the store
    def is_closed(self, name: str) -> bool:
        cur = self._conn.execute(
            "SELECT closed FROM marks WHERE name = ?",
            (name,),
        )
        row = cur.fetchone()
        return bool(row[0]) if row else False

    # Get all names in store (can be slow for large databases)
    def names(self) -> List[str]:
        cur = self._conn.execute("SELECT name FROM marks ORDER BY name")
        return [r[0] for r in cur.fetchall()]

    # Useful information about database contents
    def stats(self) -> List[dict]:
        cur = self._conn.execute(
            "SELECT COUNT(*), SUM(LENGTH(blob)), SUM(num_nonzero) FROM marks"
        )
        return [
            {
                "entries": n,
                "compressed_bytes": b or 0,  # Handle None case
                "nonzero_marks": z or 0,  # Handle None case
            }
            for n, b, z in cur.fetchall()
        ]

    def close(self) -> None:
        self._conn.close()

    # Automatic support for "with" blocks, ensuring close() called afterward
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
