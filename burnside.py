# Example usage:
# from burnside import BurnsideRing
# R = BurnsideRing("S3")
# R.show_marks()     Note: Use q to quit, w/a/s/d or arrow keys to scroll, mouse wheel if supported
# R.transitive(0) + R.transitive(1)
# R.transitive(1) * R.transitive(2)
# R.transitive(1) ** 2
# R.find("C3")
# R.one
# R.zero
# ...

from __future__ import annotations
import os
import re
import curses
import heapq
import difflib
from tqdm import tqdm
from sympy import divisors
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union, Type

try:
    from sage.all import libgap, Integer, matrix, ZZ  # type: ignore[import]
    from build_cache import _marks_from_gap_group_or_string
except ImportError:
    libgap, Integer, matrix, ZZ, _marks_from_gap_group_or_string = (
        None,
        None,
        None,
        None,
        None,
    )

from tom_store import TomStore, triples_from_dense


_DEFAULT_DB = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "tom_cache.sqlite"
)
_MATCH_NUM = 5  # Number of close matches to return for fuzzy search
_MATCH_CUTOFF = 0.6  # Minimum similarity ratio for fuzzy search (0 to 1)

# Global dict of stores for TOM data, shared across all BurnsideRing instances
_stores: Dict[str, TomStore] = {}

if libgap is not None:
    # Load TomLib globally
    libgap.LoadPackage("tomlib")


# Load store from disk and add to global dict
def _get_store(path: str = _DEFAULT_DB) -> TomStore:
    if path not in _stores:
        _stores[path] = TomStore(path)
    return _stores[path]


# Try to resolve a group name to its table of marks data
# In order, check store, TomLib, then SmallGroup(n,k)
def _resolve(
    name: str, store: TomStore
) -> Optional[Tuple[List[Tuple[int, int, int]], List[int], List[str], int, str]]:
    data = store.get(name)
    if data is not None:
        # Add source="database" for display purposes
        return (
            data.triples,
            data.subgroup_orders,
            data.subgroup_names,
            data.group_order,
            "database",
        )

    if libgap is None or _marks_from_gap_group_or_string is None:
        return None  # Cannot resolve without libgap

    # Try to get marks
    marks = _marks_from_gap_group_or_string(name)
    if marks is not None:
        # Convert to triples
        return triples_from_dense(marks[0]), marks[1], marks[2], marks[3], "tomlib"

    # Detects strings of the form "SmallGroup( n , k )" and extracts n,k
    regex = re.compile(r"^SmallGroup\(\s*(\d+)\s*,\s*(\d+)\s*\)$")
    m = regex.match(name)
    if m:
        n, k = int(m.group(1)), int(m.group(2))
        G = libgap.SmallGroup(n, k)
        marks = _marks_from_gap_group_or_string(G)
        if marks is not None:
            return (
                triples_from_dense(marks[0]),
                marks[1],
                marks[2],
                marks[3],
                "small_group",
            )

    return None  # Fail


class BurnsideRing:
    def __init__(
        self,
        group: Union[str, object],
        name=None,
        cache_result=False,
        store_path=_DEFAULT_DB,
        skip_descriptions=True,
    ):
        self._name = name or (group if isinstance(group, str) else "G")
        store = _get_store(store_path)

        # Try to find named group
        if isinstance(group, str):
            marks = _resolve(group, store)
            if marks is None:
                raise ValueError(
                    f"Could not resolve group '{group}' to a table of marks. Not found in cache, TomLib, or SmallGroup(n,k)."
                )
            self._source = marks[-1]
            triples = marks[0]

        # Compute marks directly if input is a GAP group object
        else:
            if libgap is None or _marks_from_gap_group_or_string is None:
                raise RuntimeError(
                    "libgap is required to compute table of marks from a GAP group object."
                )

            marks = _marks_from_gap_group_or_string(
                group, skip_descriptions=skip_descriptions
            )
            if marks is None:
                raise ValueError(
                    f"Failed to compute table of marks directly for group '{group}')."
                )

            if cache_result:
                if name is None:
                    raise ValueError("cache_result=True requires an explicit `name`.")
                store.put(name, *marks)
                store.commit()
            self._source = "computed"

            # Convert marks to triples
            triples = triples_from_dense(marks[0])

        # Unpack marks data
        self._subgroup_orders: List[int] = marks[1]
        self._subgroup_names: List[str] = marks[2]
        self._order: int = marks[3]
        self._rank: int = len(self._subgroup_orders)

        # Store marks as a list of sparse rows
        self._rows: List[Dict[int, int]] = [{} for _ in range(self._rank)]

        for row, col, value in triples:
            self._rows[row][col] = value

    # Formula: ghost[j] = Sum_i orbit[i] * M[i][j]
    # Or equivalently, ghost[j] = Sum_i orbit[i] * M^T[j][i] (more efficient to access)
    def _orbit_to_ghost(self, orbit_vec: Dict[int, int]) -> Dict[int, int]:
        ghost: Dict[int, int] = {}
        for orbit_index, orbit_value in orbit_vec.items():
            for ghost_index, mark_value in self._rows[orbit_index].items():
                new = ghost.get(ghost_index, 0) + orbit_value * mark_value
                if new != 0:  # Only store nonzero entries for sparsity
                    ghost[ghost_index] = new
                else:
                    ghost.pop(ghost_index, None)  # Remove zero entries for sparsity
        return ghost

    # Back-substitution to solve for orbit given ghost. Works since M^T is upper triangular (M is lower triangular)
    # Uses a max-heap of row indices to efficiently find the new row to eliminate
    def _ghost_to_orbit(self, ghost_vec: Dict[int, int]) -> Dict[int, int]:
        orbit: Dict[int, int] = {}
        ghost = dict(ghost_vec)  # Copy to avoid mutating input

        heap = [(-i, i) for i, v in ghost.items() if v != 0]
        heapq.heapify(heap)

        # Heap will be empty when all nonzero entries have been eliminated
        while heap:
            _, row = heapq.heappop(heap)
            ghost_value = ghost.get(row, 0)
            if ghost_value == 0:
                continue  # Only happens when heap is stale

            diag = self._rows[row][row]

            # Ensure ghost value is valid (ghost map not always surjective)
            if ghost_value % diag != 0:
                raise ValueError(
                    f"Ghost vector {ghost_vec} does not define an element of A({self._name}): non-integer orbit coefficient at index {row}"
                )

            q = ghost_value // diag
            orbit[row] = q

            # Super important fix that was corrupting ghost coordinates for virtual g-sets
            ghost.pop(row, None)  # Remove the diagonal entry since it's now eliminated

            # Subtract q * row j from ghost vector to eliminate j-th coordinate
            for col, mark_value in self._rows[row].items():
                # Skip diagonal since we already counted it
                if col == row:
                    continue

                prev = ghost.get(col, 0)
                new = prev - q * mark_value

                if new != 0:
                    ghost[col] = new
                    # Only push to heap if it was previously zero and now nonzero entry)
                    # Means heap isn't perfectly ordered, but efficiency is better overall, since it avoids reorganizing
                    if prev == 0:
                        heapq.heappush(heap, (-col, col))
                else:
                    ghost.pop(col, None)  # Remove zero entries for sparsity

        return orbit

    # Find integer roots of a polynomial with integer coefficients
    # First coefficient is constant, last is leading term
    # Also sorts roots in increasing order (no duplicates)
    def _integer_roots(self, coeffs: List[int]) -> List[int]:
        roots = set()

        if len(coeffs) == 0:
            raise ValueError("Polynomial has infinitely many solutions")

        first_nonzero = 0
        while first_nonzero < len(coeffs) and coeffs[first_nonzero] == 0:
            first_nonzero += 1
            roots.add(0)  # Poly divisible by x, so 0 is a root

        if first_nonzero == len(coeffs):
            raise ValueError("Polynomial has infinitely many solutions")

        # Remove leading zeros for the polynomial
        coeffs = coeffs[first_nonzero:]
        constant_term = coeffs[0]

        # Get all positive divisors of the constant term
        divs = divisors(constant_term)

        # Find integer roots using the Rational Root Theorem
        for r in divs:
            # Check positive divisor
            val = 0
            for a in reversed(coeffs):
                val = val * r + a
            if val == 0:
                roots.add(r)

            # Check negative divisor
            val = 0
            for a in reversed(coeffs):
                val = val * (-r) + a
            if val == 0:
                roots.add(-r)

        return sorted(roots)

    # Helper function to find all BurnsideElements whose ghost coordinates are drawn from the given integer root sets
    def _solve_from_root_sets(
        self, root_sets: List[List[int]]
    ) -> List[BurnsideElement]:
        diag = [self._rows[i][i] for i in range(self._rank)]
        orbit: Dict[int, int] = {}
        ghost: Dict[int, int] = {}
        solutions: List[BurnsideElement] = []

        def dfs(k):
            # DFS finished, so we have a valid solution
            if k < 0:
                solutions.append(
                    BurnsideElement._from_orbit_and_ghost(
                        self, dict(orbit), dict(ghost)
                    )
                )
                return

            # Compute contribution from previously chosen orbit coefficients
            contribution = 0
            for j in orbit:
                if j <= k:
                    continue
                contribution += self._rows[j].get(k, 0) * orbit[j]

            # Iterate over all integer roots for the k-th polynomial and try to extend the solution
            for root in root_sets[k]:
                residual = root - contribution

                if residual % diag[k] != 0:  # Not a valid orbit coefficient, skip
                    continue

                q = residual // diag[k]

                old_ghost = ghost.get(k)
                old_orbit = orbit.get(k)

                if root:
                    ghost[k] = root
                else:
                    ghost.pop(k, None)  # Remove zero entries for sparsity

                if q:
                    orbit[k] = q
                else:
                    orbit.pop(k, None)

                dfs(k - 1)

                if old_ghost is None:
                    ghost.pop(k, None)
                else:
                    ghost[k] = old_ghost

                if old_orbit is None:
                    orbit.pop(k, None)
                else:
                    orbit[k] = old_orbit

        dfs(self._rank - 1)
        return solutions

    # Solve polynomial with BurnsideElement coefficients, returns list of BurnsideElements that are roots
    # Coefficients are in increasing order (first is constant term)
    # Errors if the polynomial has infinitely many solutions (all coefficients zero)
    def solve_poly(
        self,
        coeffs: List[Union[BurnsideElement, int]],
    ) -> List[BurnsideElement]:
        if len(coeffs) == 0:
            raise ValueError("Polynomial has infinitely many solutions")

        # Convert coeffs to genuine Burnside elements if passed as integers
        one = self.one
        real_coeffs: List[BurnsideElement] = [
            coeff if isinstance(coeff, BurnsideElement) else one * int(coeff)
            for coeff in coeffs
        ]

        # Integer roots for each polynomial of ghost coordinates
        root_sets: List[List[int]] = []
        for j in range(self._rank):
            # Construct polynomial of integer coeffs given by ghost coordinates of Burnside coeffs
            poly = [coeff.mark(j) for coeff in real_coeffs]

            int_roots = self._integer_roots(poly)
            if len(int_roots) == 0:
                return []  # No integer roots, so no Burnside roots
            root_sets.append(int_roots)

        return self._solve_from_root_sets(root_sets)

    # Find all possible special lambda structures psi^p for some p
    # Returns a mapping from basis[i] to psi^p(basis[i]) for each solution
    def find_psi_p(self, p: int):
        p = int(p)

        basis: List[BurnsideElement] = [self.transitive(i) for i in range(self._rank)]
        basis_prod: List[List[BurnsideElement]] = [
            [a * b for b in basis] for a in basis
        ]
        basis_pow = [b**p for b in basis]

        # Precompute ghost coordinate tables once, outside the DFS.
        # Both avoid any BurnsideElement multiplication inside search().
        ghost_R: List[List[int]] = [
            [basis_pow[i].mark(coord) for coord in range(self._rank)]
            for i in range(self._rank)
        ]
        ghost_R_sq: List[List[int]] = [
            [ghost_R[i][coord] ** 2 for coord in range(self._rank)]
            for i in range(self._rank)
        ]

        # Dict of choices: chosen[k] = psi^p(basis[k])
        # psi^p([G/G]) = [G/G] is always true
        chosen: Dict[int, BurnsideElement] = {self._rank - 1: basis[-1]}
        ghost_chosen: Dict[int, List[int]] = {
            self._rank - 1: [basis[-1].mark(coord) for coord in range(self._rank)]
        }
        solutions: List[Dict[BurnsideElement, BurnsideElement]] = []

        # Recursive search for psi^p(basis[i]) from i = self._rank - 2 to 0
        def search(i):
            if i == -1:
                # Full multiplicativity check (works because A(G) is commutative and associative, so we only need to check pairs)
                for a in range(self._rank):
                    for b in range(a, self._rank):
                        prod_orbit = basis_prod[a][b]._orbit
                        for coord in range(self._rank):
                            # LHS is psi^p(basis[a]) * psi^p(basis[b])
                            lhs = ghost_chosen[a][coord] * ghost_chosen[b][coord]
                            # RHS is psi^p(basis[a] * basis[b])
                            rhs = sum(
                                coeff * ghost_chosen[k][coord]
                                for k, coeff in prod_orbit.items()
                            )
                            if lhs != rhs:
                                return

                # Convert chosen to a mapping from basis[i] to psi^p(basis[i])
                solutions.append({basis[k]: chosen[k] for k in range(self._rank)})
                return

            # Get orbit coefficients (n_i) of basis[i]^2
            n_i = basis_prod[i][i]._orbit
            m = n_i.get(i, 0)

            # Build quadratic polynomial:
            # (p^2)Q^2 + (2pR_i - mp)Q + (R_i^2 - mR_i - sum_{k in chosen}n_k chosen[k]) = 0
            root_sets = []
            for coord in range(self._rank):
                constant = ghost_R_sq[i][coord] - m * ghost_R[i][coord]
                for k, coeff in n_i.items():
                    if k in chosen:
                        constant -= coeff * ghost_chosen[k][coord]
                linear = 2 * p * ghost_R[i][coord] - m * p
                int_roots = self._integer_roots([constant, linear, p * p])
                if not int_roots:
                    return  # No integer roots, so no solutions
                root_sets.append(int_roots)

            # Construct BurnsideElements from root_sets
            Q_candidates = self._solve_from_root_sets(root_sets)

            # Check if each root satisfies multiplicative compatibility:
            # (R_i + pQ)chosen[j] = psi^p(basis[i] * basis[j]) for all j > i
            for Q in Q_candidates:
                ghost_candidate = [
                    ghost_R[i][coord] + p * Q.mark(coord) for coord in range(self._rank)
                ]
                candidate: BurnsideElement = basis_pow[i] + p * Q

                # Multiplicativity pruning for efficiency: check if candidate * chosen[j] = psi^p(basis[i] * basis[j]) for all j > i
                compatible = True
                for j, choice in chosen.items():
                    if j == i:
                        continue

                    prod_orbit = basis_prod[i][j]._orbit

                    # Check that all chosen[k] are defined for k in prod.orbit
                    # Otherwise we cannot yet ensure multiplicativity
                    if any(k not in chosen and k != i for k in prod_orbit):
                        continue

                    for coord in range(self._rank):
                        lhs = ghost_candidate[coord] * ghost_chosen[j][coord]
                        rhs = sum(
                            coeff
                            * (
                                ghost_candidate[coord]
                                if k == i
                                else ghost_chosen[k][coord]
                            )
                            for k, coeff in prod_orbit.items()
                        )
                        if lhs != rhs:
                            compatible = False
                            break

                    if not compatible:
                        break  # No need to check further if already incompatible

                # Failed multiplicativity check, so skip
                if not compatible:
                    continue

                # Add to chosen and recurse
                chosen[i] = candidate
                ghost_chosen[i] = ghost_candidate
                search(i - 1)
                del chosen[i]  # For backtracking
                del ghost_chosen[i]  # For backtracking

        # Start search from index self._rank - 2 since psi^p([G/G]) = [G/G] is fixed
        search(self._rank - 2)
        return solutions

    # Try to find transitive G-set by subgroup_name, used for exploratory purposes, not programmatically
    def find(self, name: str, print_results: bool = True) -> Optional[BurnsideElement]:
        name_lower = name.lower()

        for i, subgroup_name in enumerate(self._subgroup_names):
            # Exact match
            if subgroup_name == name:
                if print_results:
                    print(f"Found exact match: [{subgroup_name}] at index {i}")
                return self.transitive(i)

            # Case-insensitive match
            elif subgroup_name.lower() == name_lower:
                if print_results:
                    print(
                        f"Found case-insensitive match: [{subgroup_name}] at index {i}"
                    )
                return self.transitive(i)

        # No need to fuzzy match
        if not print_results:
            return None

        # Fuzzy match using difflib
        names_lower = [subgroup_name.lower() for subgroup_name in self._subgroup_names]
        close = difflib.get_close_matches(
            name_lower, names_lower, n=_MATCH_NUM, cutoff=_MATCH_CUTOFF
        )
        if close:
            print("Close matches:")
            for match in close:
                i = names_lower.index(match)  # Index into lowercased list
                print(
                    f"  [{self._subgroup_names[i]}] at index {i} (order {self._subgroup_orders[i]})"
                )
        else:
            print("No close matches found")

        return None

    # Construct from orbit coefficients (integers in the basis of transitive G-sets [G/H_i])
    def from_orbit(self, coeffs: Dict[int, int]):
        return BurnsideElement._from_orbit_and_ghost(
            self, coeffs, self._orbit_to_ghost(coeffs)
        )

    # Construct from ghost coordinates
    def from_ghost(self, ghost: Dict[int, int]):
        return BurnsideElement._from_orbit_and_ghost(
            self, self._ghost_to_orbit(ghost), ghost
        )

    # Access i-th transitive generator [G/H_i] (0-indexed from smallest to largest)
    def transitive(self, i: int):
        return BurnsideElement(self, i)

    # Name of group
    @property
    def name(self):
        return self._name

    # Zero element in orbit basis
    @property
    def zero(self):
        return BurnsideElement._from_orbit_and_ghost(self, {}, {})

    # Multiplicative identity [G/G] (last basis element)
    @property
    def one(self):
        return BurnsideElement(self, self._rank - 1)

    @property
    def source(self):
        return self._source

    # Rank of Burnside Ring = number of conjugacy classes of subgroups
    @property
    def rank(self):
        return self._rank

    # Order of the group
    @property
    def order(self):
        return self._order

    # Orders of subgroups (increasing order as usual)
    @property
    def subgroup_orders(self):
        return self._subgroup_orders

    # Names of subgroups
    @property
    def subgroup_names(self):
        return self._subgroup_names

    # Honest G-sets have non-negative orbit coefficients
    def is_honest(self, element):
        return all(c >= 0 for c in element.orbit)

    # Generate the full marks matrix as a Python list of lists
    # Warning: slow and built on demand (for inspection purposes only)
    def M(self) -> List[List[int]]:
        entries = [[0 for _ in range(self._rank)] for _ in range(self._rank)]

        for row in range(self._rank):
            for col, value in self._rows[row].items():
                entries[row][col] = int(value)

        return entries

    # Same for transpose (also slow and built on demand)
    def M_T(self) -> List[List[int]]:
        entries = [[0 for _ in range(self._rank)] for _ in range(self._rank)]

        for row in range(self._rank):
            for col, value in self._rows[row].items():
                entries[col][row] = int(value)

        return entries

    # Inverse of the table of marks as a sage matrix
    def M_inv(self):
        if matrix is None or ZZ is None:
            raise RuntimeError(
                "Sage is required to compute the inverse of the table of marks."
            )

        # Convert to a flattened coordinate dictionary for Sage matrix constructor
        coord_dict = {
            (row, col): val
            for row, data in enumerate(self._rows)
            for col, val in data.items()
        }

        M = matrix(ZZ, self._rank, self._rank, coord_dict)
        return M.inverse()

    # Display table of marks in human readable format with subgroup labels
    # Dynamic switches to an interactive display and is very useful for large ranks
    # Dynamic mode uses w/a/s/d or arrow keys, also support scrolling wheel (and shift + scroll for horizontal) if terminal supports it. Press q or esc to quit dynamic mode.
    def show_marks(self, dynamic=None):
        # If dynamic is None, choose automatically based on terminal size
        if dynamic is None:
            try:
                w, h = os.get_terminal_size()
                # Might modify these formulas later, because they're basically approximations
                max_rows = max(1, h - 2)  # Leave space for header and footer
                max_cols = max(1, (w - 12) // 12)  # Approximate cell width
                dynamic = self._rank > max_rows or self._rank > max_cols
            except OSError:
                dynamic = False  # Default to static

        # If large rank and dynamic
        if dynamic:
            curses.wrapper(lambda stdscr: draw(self, stdscr))
            return

        column_labels = [f"[{self._name}/{n}]" for n in self._subgroup_names]

        cell_w = max(len(l) for l in column_labels) + 2
        header = " " * cell_w + "".join(l.rjust(cell_w) for l in self._subgroup_names)
        print(f"\nTable of marks for {self._name}  (source: {self._source})")
        print("-" * len(header))
        print(header)
        for i in range(self._rank):
            row = self._rows[i]
            cells = "".join(str(row.get(j, 0)).rjust(cell_w) for j in range(self._rank))
            print(column_labels[i].ljust(cell_w) + cells)
        print()

    # Display helper
    def __repr__(self):
        return (
            f"BurnsideRing('{self._name}', rank={self._rank}, source={self._source!r})"
        )


# Helper to draw an interactive table of marks
def draw(ring: BurnsideRing, stdscr: curses.window):
    curses.curs_set(0)
    curses.mousemask(curses.ALL_MOUSE_EVENTS)
    stdscr.keypad(True)

    pos = [0, 0]

    # Might need to fix this up in the future, works for my current terminal but not universal
    SCROLL_UP = 65536
    SCROLL_DOWN = 2097152
    HSCROLL_LEFT = 67174400
    HSCROLL_RIGHT = 69206016

    while True:
        stdscr.clear()

        h, width = stdscr.getmaxyx()
        cell_w = max(len(l) for l in ring._subgroup_names) + 2

        view_rows = max(1, h - 2)
        view_cols = max(1, (width - cell_w) // cell_w)

        max_row = max(0, ring._rank - view_rows)
        max_col = max(0, ring._rank - view_cols)

        pos[0] = min(max(0, pos[0]), max_row)
        pos[1] = min(max(0, pos[1]), max_col)

        header = " " * cell_w + "".join(
            l.rjust(cell_w) for l in ring._subgroup_names[pos[1] : pos[1] + view_cols]
        )
        stdscr.addstr(0, 0, header[: width - 1])

        for i in range(pos[0], min(pos[0] + view_rows, ring._rank)):
            row = ring._rows[i]

            cells = "".join(
                str(row.get(j, 0)).rjust(cell_w)
                for j in range(pos[1], min(pos[1] + view_cols, ring._rank))
            )

            line = ring._subgroup_names[i].ljust(cell_w) + cells
            stdscr.addstr(i - pos[0] + 1, 0, line[: width - 1])

        stdscr.refresh()

        key = stdscr.getch()

        if key in (ord("q"), ord("Q"), 27):  # q or esc to quit
            break
        elif key in (ord("w"), curses.KEY_UP):
            pos[0] = max(0, pos[0] - 1)
        elif key in (ord("s"), curses.KEY_DOWN):
            pos[0] = min(max_row, pos[0] + 1)
        elif key in (ord("a"), curses.KEY_LEFT):
            pos[1] = max(0, pos[1] - 1)
        elif key in (ord("d"), curses.KEY_RIGHT):
            pos[1] = min(max_col, pos[1] + 1)
        elif key == curses.KEY_MOUSE:
            try:
                _, _, _, _, bstate = curses.getmouse()

                if bstate == SCROLL_UP:
                    pos[0] = max(0, pos[0] - 1)
                elif bstate == SCROLL_DOWN:
                    pos[0] = min(max_row, pos[0] + 1)
                elif bstate == HSCROLL_LEFT:
                    pos[1] = max(0, pos[1] - 1)
                elif bstate == HSCROLL_RIGHT:
                    pos[1] = min(max_col, pos[1] + 1)
            except curses.error:
                pass


# Create burnside ring and cache in global dict to avoid recomputation
@lru_cache(maxsize=64)
def get_ring(name: str) -> BurnsideRing:
    return BurnsideRing(name)


# Orbits and ghost coordinates are stored as sparse dicts
# Ghost coordinates are lazily computed from orbit coefficients (which always exist)
class BurnsideElement:
    # Pass in either orbit coefficients or a single int
    # Int indicates that vector is a basis element with 1 in that position and 0 elsewhere
    # Makes ghost coordinate calculations extremely fast for transitive generators
    def __init__(
        self,
        ring: BurnsideRing,
        orbit_coeffs: Union[Dict[int, int], int],
    ):
        self.ring: BurnsideRing = ring

        if isinstance(orbit_coeffs, (int, Integer if Integer is not None else int)):
            orbit_coeffs = int(orbit_coeffs)  # type: ignore
            if orbit_coeffs < 0 or orbit_coeffs >= ring.rank:
                raise ValueError(
                    f"Basis index {orbit_coeffs} out of range for BurnsideRing of rank {ring.rank}"
                )
            self._orbit: Dict[int, int] = {orbit_coeffs: 1}  # type: ignore

            # Ghost coordinates of basis vectors are simply the corresponding column of the table of marks
            self._ghost: Optional[Dict[int, int],] = dict(ring._rows[orbit_coeffs])  # type: ignore
        else:
            self._orbit: Dict[int, int] = dict(orbit_coeffs)  # type: ignore
            self._ghost: Optional[Dict[int, int]] = None  # Defer for now

    # Internal helper to efficiently construct from orbit and ghost coordinates
    # DOES NOT CREATE COPIES, so use carefully
    @classmethod
    def _from_orbit_and_ghost(
        cls, ring: BurnsideRing, orbit: Dict[int, int], ghost: Optional[Dict[int, int]]
    ):
        obj = cls.__new__(cls)
        obj.ring = ring
        obj._orbit = orbit
        obj._ghost = ghost
        return obj

    # Orbit coefficients of element
    @property
    def orbit(self):
        return self._orbit

    # Ghost coordinates of element
    @property
    def ghost(self):
        if self._ghost is None:
            self._ghost = self.ring._orbit_to_ghost(self._orbit)
        return self._ghost

    # Return a copy of this element
    def copy(self):
        ghost_copy = dict(self.ghost) if self._ghost is not None else None
        return BurnsideElement._from_orbit_and_ghost(
            self.ring, dict(self._orbit), ghost_copy
        )

    # Helper for accessing individual ghost coordinates (0-indexed)
    def mark(self, j: int) -> int:
        return self.ghost.get(int(j), 0)

    # Addition is component-wise in orbit basis
    def __add__(self, other: Union[BurnsideElement, int]):
        if isinstance(other, (int, Integer if Integer is not None else int)):
            other = self.ring.one * int(other)  # type: ignore

        if not isinstance(other, BurnsideElement):
            other = self.ring.one * other
        if self.ring is not other.ring:
            raise ValueError("Cannot add BurnsideElements from different BurnsideRings")

        # Order orbits by size to minimize number of iterations in the addition loop
        smaller_orbit, larger_orbit = self._orbit, other._orbit
        if len(self._orbit) > len(other._orbit):
            smaller_orbit, larger_orbit = other._orbit, self._orbit

        orb_result = dict(larger_orbit)  # Start with larger orbit
        for k, v in smaller_orbit.items():
            new = orb_result.get(k, 0) + v
            if new != 0:
                orb_result[k] = new
            else:
                orb_result.pop(k, None)

        # Repeat for ghost coordinates if both ghost coordinates exist, otherwise defer
        if self._ghost is not None and other._ghost is not None:
            smaller_ghost, larger_ghost = self._ghost, other._ghost
            if len(self._ghost) > len(other._ghost):
                smaller_ghost, larger_ghost = other._ghost, self._ghost

            ghost_result = dict(larger_ghost)  # Start with larger ghost
            for k, v in smaller_ghost.items():
                new = ghost_result.get(k, 0) + v
                if new != 0:
                    ghost_result[k] = new
                else:
                    ghost_result.pop(k, None)
        else:
            ghost_result = None  # Defer

        return BurnsideElement._from_orbit_and_ghost(
            self.ring, orb_result, ghost_result
        )

    def __radd__(self, other):
        return self.__add__(other)

    # Negation is component-wise in orbit basis
    def __neg__(self):
        if self._ghost is not None:
            ghost_neg = {k: -v for k, v in self._ghost.items()}
        else:
            ghost_neg = None  # Defer

        return BurnsideElement._from_orbit_and_ghost(
            self.ring, {k: -v for k, v in self._orbit.items()}, ghost_neg
        )

    # Subtraction is just addition of negation
    def __sub__(self, other: Union[BurnsideElement, int]):
        if isinstance(other, (int, Integer if Integer is not None else int)):
            other = self.ring.one * int(other)  # type: ignore

        if not isinstance(other, BurnsideElement):
            raise TypeError(
                f"Unsupported operand type(s) for -: 'BurnsideElement' and '{type(other).__name__}'"
            )

        orb_keys = self._orbit.keys() | other._orbit.keys()
        result = {
            k: v
            for k in orb_keys
            if (v := self._orbit.get(k, 0) - other._orbit.get(k, 0)) != 0
        }

        # Repeat for ghost coordinates if both exist
        if self._ghost is not None and other._ghost is not None:
            ghost_keys = self._ghost.keys() | other._ghost.keys()
            ghost_result = {
                k: v
                for k in ghost_keys
                if (v := self._ghost.get(k, 0) - other._ghost.get(k, 0)) != 0
            }
        else:
            ghost_result = None  # Defer

        return BurnsideElement._from_orbit_and_ghost(self.ring, result, ghost_result)

    # Supports scalar multipliction by integers and BurnsideRing multiplication
    def __mul__(self, other: Union[int, BurnsideElement]):
        if isinstance(other, (int, Integer if Integer is not None else int)):
            other = int(other)  # type: ignore

            if self._ghost is not None:
                ghost = {k: v * other for k, v in self._ghost.items()}
            else:
                ghost = None  # Defer

            return BurnsideElement._from_orbit_and_ghost(
                self.ring,
                {k: v * other for k, v in self._orbit.items()},
                ghost,
            )

        if not isinstance(other, BurnsideElement):
            raise TypeError(
                f"Unsupported operand type(s) for *: 'BurnsideElement' and '{type(other).__name__}'"
            )
        if self.ring is not other.ring:
            raise ValueError(
                "Cannot multiply BurnsideElements from different BurnsideRings"
            )

        # For multiplication, need to compute ghost coordinates now
        # Multiply ghost coordinates coordinate-wise, then convert back to orbit coefficients
        self_ghost, other_ghost = self.ghost, other.ghost
        # Only multiply where both are nonzero
        ghost_prod = {
            k: self_ghost.get(k, 0) * other_ghost.get(k, 0)
            for k in self_ghost.keys() & other_ghost.keys()
        }
        return BurnsideElement._from_orbit_and_ghost(
            self.ring, self.ring._ghost_to_orbit(ghost_prod), ghost_prod
        )

    def __rmul__(self, scalar: int):
        return self.__mul__(scalar)

    # Rather than implementing repeated multiplication, we can use the fact that ghost map is a ring homomorphism
    # So, we can just exponentiate ghost coordinates and then convert back
    def __pow__(self, n: int):
        n = int(n)  # Convert to Python int immediately
        if n < 0:
            raise ValueError("Exponent must be a non-negative integer.")

        # Special case to avoid potential issues
        if n == 0:
            return self.ring.one

        # Compute ghost coordinates now
        self_ghost = self.ghost
        ghost_prod = {k: v**n for k, v in self_ghost.items() if v != 0}
        orbit = self.ring._ghost_to_orbit(ghost_prod)
        return BurnsideElement._from_orbit_and_ghost(self.ring, orbit, ghost_prod)

    def __eq__(self, other):
        return (
            isinstance(other, BurnsideElement)
            and self.ring
            is other.ring  # Ring must be the same object (not copies), for efficiency
            and self._orbit == other._orbit
        )

    # Display element as a linear combination of transitive generators
    def __repr__(self):
        labels = self.ring._subgroup_names
        # Only include nonzero terms
        terms = [
            f"{v}[{self.ring._name}/{labels[k]}]"
            for k, v in self._orbit.items()
            if v != 0
        ]
        return " + ".join(terms) if terms else "0"

    # Hash based on ring identity and orbit coefficients
    def __hash__(self):
        return hash((id(self.ring), frozenset(self._orbit.items())))

    # Detailed readable data about an element (useful for debugging and verification)
    def show(self):
        labels = self.ring._subgroup_names
        w = max(len(l) for l in labels) + 2
        print(f"\nElement of A({self.ring.name})")
        print(f"  {'Subgroup'.ljust(w)}  {'Orbit':>10}  {'Ghost':>10}")
        print("  " + "-" * (w + 29))

        # Compute ghost coordinates once for efficiency
        ghost = self.ghost
        for k, v in self._orbit.items():
            subgroup = f"[{labels[k]}]"
            print(f"  {subgroup.ljust(w)}  {str(v):>10}  {str(ghost.get(k, 0)):>10}")
        print()


def verify_c2():
    R = BurnsideRing("C2", name="C2")
    R.show_marks()

    print("C2 verification")
    for i in range(R.rank):
        print(f"Transitive generator {i}:")
        R.transitive(i).show()

    pt = R.one
    free = R.transitive(0)

    # Verify multiplicative identity and [G/e]^2 = 2[G/e]
    assert pt * pt == pt, "Multiplicative identity check failed!"
    assert (free * free).ghost == {
        k: 2 * v for k, v in free.ghost.items()
    }, "Multiplication check failed for [G/e]^2!"


def verify_s3():
    R = BurnsideRing("S3", name="S3")
    R.show_marks()

    print("S3: multiplicative table on basis elements")
    for i in range(R.rank):
        for j in range(i, R.rank):
            prod = R.transitive(i) * R.transitive(j)
            print(f"  e_{i} * e_{j} = {prod}")
    print()

    pt = R.one
    assert pt * pt == pt, "Multiplicative identity check failed!"
    print("Multiplicative identity [pt]^2 = [pt] passed")
    print()


def run_verification():
    verify_c2()
    verify_s3()
    print("All checks passed (see output above for manual inspection).")


if __name__ == "__main__":
    run_verification()
