"""Centralized constants for Hexagonal Tic-Tac-Toe."""

from typing import Final, Literal

# Player Type
type Player = Literal[1, 2]

# Game Rules
RADIUS: Final[int] = 50
WIN_LENGTH: Final[int] = 6

# Player Identifiers
PLAYER_1: Final[Player] = 1
PLAYER_2: Final[Player] = 2

# Derived Constants
# The width of the padded internal bitboard representation.
PADDED_WIDTH: Final[int] = 2 * RADIUS + WIN_LENGTH
GRID_SIZE: Final[int] = PADDED_WIDTH * PADDED_WIDTH

# Max number of pieces that can fit on the board (the valid cells)
MAX_PIECES: Final[int] = 3 * RADIUS * RADIUS - 3 * RADIUS + 1
