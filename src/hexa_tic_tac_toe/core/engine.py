"""Hexagonal Tic-Tac-Toe game logic.

This module contains the HexGame class which handles the rules, win conditions,
and board state for a hexagonal tic-tac-toe game played on a grid of bitboards.
"""

from collections.abc import Iterable
from typing import Final, Literal

# Type alias for coordinates
type Coord = tuple[int, int]
# Player type (1 or 2)
type Player = Literal[1, 2]


class HexGame[T]:
    """A hexagonal tic-tac-toe game engine.

    The game is played on a hexagonal grid where players take turns placing
    marks. Player 1 plays once on the first turn, and then players alternate
    taking two moves each. The first player to get 6 in a row wins.

    Attributes:
        radius: The radius of the hexagonal board.
        win_length: Final number of marks in a row required to win (6).
        current_player: The index of the player whose turn it is (1 or 2).
        moves_this_turn: The number of moves already made in the current turn.
        turn_number: The current turn number (starts at 1).
        move_history: A list of all moves made in the game.
        winner: The player index (1 or 2) if the game is won, else None.
    """

    radius: int
    win_length: Final[int] = 6
    # A list containing bitboards for player 1 and player 2.
    _boards: list[int]
    current_player: Player
    moves_this_turn: int
    turn_number: int
    move_history: list[Coord]
    winner: Player | None
    # Internal offset used for mapping axial coordinates to bitboard indices.
    _offset: int
    # The width of the padded internal bitboard representation.
    _padded_width: int

    def __init__(self, radius: int = 50) -> None:
        """Initializes a new HexGame.

        Args:
            radius: The radius of the hexagonal board. Defaults to 50.
        """
        self.radius = radius
        # Bitboards for player 1 and 2
        self._boards = [0, 0]
        self.current_player = 1
        self.moves_this_turn = 0
        self.turn_number = 1
        self.move_history = []
        self.winner = None
        # Mapping parameters: (q, r) -> (q+_offset) * _padded_width + (r+_offset)
        self._offset = radius - 1
        self._padded_width = 2 * radius + self.win_length  # Padding for shift-win check

    def _coord_to_index(self, q: int, r: int) -> int:
        """Converts axial coordinates to a bitboard index.

        Args:
            q: The axial q-coordinate.
            r: The axial r-coordinate.

        Returns:
            The integer index for the bitboard.
        """
        return (q + self._offset) * self._padded_width + (r + self._offset)

    def get_all_coordinates(self) -> Iterable[Coord]:
        """Provides an iterator over all valid axial coordinates on the board.

        Returns:
            An iterable of (q, r) tuples representing all valid cells.
        """
        for q in range(-(self.radius - 1), self.radius):
            for r in range(-(self.radius - 1), self.radius):
                if abs(q + r) < self.radius:
                    yield (q, r)

    def is_valid_coordinate(self, q: int, r: int) -> bool:
        """Checks if a (q, r) coordinate is within the board radius.

        Args:
            q: Axial q-coordinate.
            r: Axial r-coordinate.

        Returns:
            True if the coordinate is on the board.
        """
        return (
            abs(q) < self.radius and abs(r) < self.radius and abs(q + r) < self.radius
        )

    def is_valid_move(self, q: int, r: int) -> bool:
        """Checks if a coordinate is a valid move.

        A move is valid if it is within the hexagonal radius and the cell
        is not already occupied by either player.

        Args:
            q: The axial q-coordinate.
            r: The axial r-coordinate.

        Returns:
            True if the move is valid, False otherwise.
        """
        if not self.is_valid_coordinate(q, r):
            return False

        index = self._coord_to_index(int(q), int(r))
        # Check if either board has the bit set
        return not (((self._boards[0] | self._boards[1]) >> index) & 1)

    def get_player_at(self, q: int, r: int) -> Player | None:
        """Returns the player occupying the given cell, or None if empty.

        Args:
            q: The axial q-coordinate.
            r: The axial r-coordinate.

        Returns:
            1 if player 1 occupies the cell, 2 if player 2, None otherwise.
        """
        if not self.is_valid_coordinate(q, r):
            return None

        index = self._coord_to_index(int(q), int(r))
        if (self._boards[0] >> index) & 1:
            return 1
        if (self._boards[1] >> index) & 1:
            return 2
        return None

    def make_move(self, q: int, r: int) -> Player | None:
        """Updates the board with a player's move.

        Args:
            q: The axial q-coordinate.
            r: The axial r-coordinate.

        Returns:
            The player number (1 or 2) if the move ends the game with a win,
            None otherwise.

        Raises:
            ValueError: If the move is invalid or the cell is already occupied.
        """
        if not self.is_valid_move(q, r):
            if self.is_valid_coordinate(q, r):
                raise ValueError(f"Cell ({q}, {r}) is already occupied.")
            raise ValueError(f"Cell ({q}, {r}) is outside the radius {self.radius}.")

        index = self._coord_to_index(int(q), int(r))
        player_index = self.current_player - 1
        self._boards[player_index] |= 1 << index
        self.move_history.append((int(q), int(r)))

        if self._check_win(self.current_player):
            self.winner = self.current_player
            return self.winner

        # Turn logic: 1 move for P1 on turn 1, 2 moves for every turn after.
        self.moves_this_turn += 1
        moves_needed = 1 if self.turn_number == 1 else 2

        if self.moves_this_turn >= moves_needed:
            # Swap 1 and 2
            self.current_player = 1 if self.current_player == 2 else 2
            self.moves_this_turn = 0
            self.turn_number += 1

        return None

    def get_winner(self) -> Player | None:
        """Checks if either player has won the game.

        Returns:
            The player number (1 or 2) who has won, or None if no winner yet.
        """
        if self._check_win(1):
            return 1
        if self._check_win(2):
            return 2
        return None

    def undo_move(self) -> None:
        """Reverts the last move made in the game.

        This method updates the bitboards, current player, turn number,
        and moves this turn to exactly what they were before the last move.
        """
        if not self.move_history:
            return

        # Get last move
        q, r = self.move_history.pop()
        index = self._coord_to_index(q, r)

        # Clear bits on both boards to be robust
        self._boards[0] &= ~(1 << index)
        self._boards[1] &= ~(1 << index)
        self.winner = None  # Reset winner on undo

        # Revert turn logic
        if self.moves_this_turn > 0:
            self.moves_this_turn -= 1
        else:
            # We were at the start of a turn, so we go back to the previous turn
            self.turn_number -= 1
            self.current_player = 1 if self.current_player == 2 else 2
            moves_needed = 1 if self.turn_number == 1 else 2
            self.moves_this_turn = moves_needed - 1

    def reset(self) -> None:
        """Resets the game to the initial empty state."""
        self._boards = [0, 0]
        self.current_player = 1
        self.moves_this_turn = 0
        self.turn_number = 1
        self.move_history = []
        self.winner = None

    def _check_win(self, player: int) -> bool:
        """Checks if the specified player has won the game.

        Uses bitwise shifts to detect `win_length` in a row in all 3 axes.
        The bitboard layout ensures that:
        - Direction 1 is the r-axis (0, 1)
        - Direction _padded_width is the q-axis (1, 0)
        - Direction _padded_width-1 is the s-axis (1, -1)

        Args:
            player: The player number (1 or 2) to check.

        Returns:
            True if the player has `win_length` marks in a row, False otherwise.
        """
        bits = self._boards[player - 1]
        for direction in (1, self._padded_width, self._padded_width - 1):
            # Check for win_length in a row: bits & bits<<d & bits<<2d ... & bits<<(N-1)d
            mask = bits
            for shift in range(1, self.win_length):
                mask &= bits << (shift * direction)
            if mask:
                return True
        return False

    @classmethod
    def from_string(cls: type[T], text_grid: str) -> T:
        """Creates a HexGame instance from a string representation.

        The input should be a staggered grid of '.', 'X', 'O'.
        Indentation is used to represent the hexagonal structure.

        Args:
            text_grid: A multi-line string representing the hexagonal board.

        Returns:
            A HexGame instance initialized with the state from the string.

        Raises:
            ValueError: If the grid string is empty, has an even number of rows,
                or has inconsistent row lengths.
        """
        lines = [line.strip() for line in text_grid.strip().split("\n") if line.strip()]
        if not lines:
            raise ValueError("Empty grid string.")

        num_rows = len(lines)
        if num_rows % 2 == 0:
            raise ValueError("Hexagonal grid must have an odd number of rows (2R-1).")

        radius = (num_rows + 1) // 2
        game = cls(radius=radius)

        total_pieces: int = 0
        for row_index, line in enumerate(lines):
            r = row_index - (radius - 1)
            # Validate allowed characters (ignoring spaces)
            cleaned_line = line.replace(" ", "")
            if not all(c in "XO." for c in cleaned_line):
                invalid_chars = "".join(
                    sorted(set(c for c in cleaned_line if c not in "XO."))
                )
                raise ValueError(
                    f"Invalid characters '{invalid_chars}' in row {row_index}."
                )

            q_start = max(-(radius - 1), -(radius - 1) - r)
            expected_len = 2 * radius - 1 - abs(r)

            if len(cleaned_line) != expected_len:
                raise ValueError(
                    f"Row {row_index} (r={r}) has {len(cleaned_line)} cells, "
                    f"expected {expected_len}."
                )

            for q_idx, char in enumerate(cleaned_line):
                if char == ".":
                    continue

                q = q_start + q_idx
                index = game._coord_to_index(q, r)
                if char == "X":
                    game._boards[0] |= 1 << index
                elif char == "O":
                    game._boards[1] |= 1 << index
                total_pieces += 1

        # Reconstruct turn state from the number of pieces
        if total_pieces > 0:
            game.turn_number = (total_pieces + 1) // 2 + 1
            game.current_player = 2 if game.turn_number % 2 == 0 else 1
            game.moves_this_turn = (total_pieces - 1) % 2
        else:
            game.moves_this_turn = 0

        game.winner = game.get_winner()
        return game

    def __str__(self) -> str:
        """Returns a string representation of the hexagonal board.

        The board is rendered as a staggered ASCII grid with '.' for empty cells,
        'X' for player 1, and 'O' for player 2.

        Returns:
            A string representing the board state.
        """
        lines: list[str] = []
        for r in range(-(self.radius - 1), self.radius):
            # Indent proportional to distance from center
            indent = " " * abs(r)
            row_chars: list[str] = []

            q_start = max(-(self.radius - 1), -(self.radius - 1) - r)
            q_end = min((self.radius - 1), (self.radius - 1) - r)

            for q in range(q_start, q_end + 1):
                player = self.get_player_at(q, r)
                if player == 1:
                    row_chars.append("X")
                elif player == 2:
                    row_chars.append("O")
                else:
                    row_chars.append(".")

            lines.append(indent + " ".join(row_chars))

        return "\n".join(lines)
