# Hexagonal Tic-Tac-Toe Terminology

This document serves as a glossary of the core concepts, data structures, and rules used in the Hexagonal Tic-Tac-Toe engine.

## Game Mechanics

- **Turn Pattern (1-2-2-2...)**: The unique move sequence built into the game. Player 1 places `1` mark on the first turn. After the first turn, players alternate, each placing `2` marks per turn.
- **Win Length**: The number of contiguous marks a player must align in any of the three directional axes to win the game. In this engine, the default win length is `6`.

## Board & Coordinates

- **Hexagonal Grid**: The playing area, shaped as a large regular hexagon made up of smaller hexagonal cells.
- **Radius (`R`)**: The size of the board, defined as the number of cells from the center cell to the outermost edge, inclusive. The total number of cells across the middle axis is `2R - 1`. 
- **Axial Coordinates (`q`, `r`)**: A 2D coordinate system used for hexagonal grids.
  - `q` represents the primary axis (often thought of as the horizontal or diagonal column).
  - `r` represents the secondary axis (often thought of as the row).
  - A cell is valid on a board of radius `R` if it satisfies the bounds: `abs(q) < R`, `abs(r) < R`, and `abs(q + r) < R`.
- **Cube Coordinates (`q`, `r`, `s`)**: An extension of axial coordinates that makes hexagonal symmetry more apparent. The completely implicit third axis `s` is defined by the constraint `q + r + s = 0`.
  - **r-axis**, **q-axis**, and **s-axis** are the three lines of symmetry along which players can form winning lines.

## Engine & Data Structures

- **Bitboard**: A highly optimized data structure where the entire game state for a player is stored within a single integer (a sequence of bits). A bit is set to `1` if the player has a mark on the corresponding cell. This allows the engine to compute game states, valid moves, and wins using extremely fast bitwise operations.
- **`W` (Padded Width)**: An internal constant representing the width of a row in the 1D bitboard memory layout. It includes "padding" (equal to the win length) to ensure that bit-shifting operations used for win-checking do not erroneously wrap around from the end of one row to the beginning of the next.
- **Offset**: An internal shift applied to `(q, r)` coordinates before storing them in the 1D bitboard array. Because axial coordinates can be negative, the offset ensures all indices in the bitboard map to positive integers.
