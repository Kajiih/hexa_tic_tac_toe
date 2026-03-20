"""Hexagonal Tic Tac Toe game logic.

This module contains the HexGame class which handles the rules, win conditions,
and board state for a hexagonal tic-tac-toe game played on a grid of bitboards.
"""

# Type alias for coordinates
type Coord = tuple[int, int]


class HexGame:
    """A hexagonal tic-tac-toe game engine.

    The game is played on a hexagonal grid where players take turns placing
    marks. Player 1 plays once on the first turn, and then players alternate
    taking two moves each. The first player to get 6 in a row wins.

    Attributes:
        radius: The radius of the hexagonal board.
        win_length: The number of marks in a row required to win.
        current_player: The index of the player whose turn it is (1 or 2).
        moves_this_turn: The number of moves already made in the current turn.
        turn_number: The current turn number (starts at 1).
    """

    radius: int
    win_length: int
    # A list containing bitboards for player 1 and player 2.
    _boards: list[int]
    current_player: int
    moves_this_turn: int
    turn_number: int
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
        self.win_length = 6
        # Bitboards for player 1 and 2
        self._boards = [0, 0]
        self.current_player = 1
        self.moves_this_turn = 0
        self.turn_number = 1
        # Mapping parameters: (q, r) -> (q+_offset) * _padded_width + (r+_offset)
        self._offset = radius - 1
        self._padded_width = 2 * radius + self.win_length  # Padding for shift-win check

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
        # Axial coordinates constraint for a hexagon:
        # abs(q) < R, abs(r) < R, abs(q + r) < R
        if abs(q) >= self.radius or abs(r) >= self.radius or abs(q + r) >= self.radius:
            return False

        index = (q + self._offset) * self._padded_width + (r + self._offset)
        # Check if either board has the bit set
        if (self._boards[0] | self._boards[1]) & (1 << index):
            return False

        return True

    def make_move(self, q: int, r: int) -> int | None:
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
            if (
                abs(q) < self.radius
                and abs(r) < self.radius
                and abs(q + r) < self.radius
            ):
                # Occupied
                raise ValueError(f"Cell ({q}, {r}) is already occupied.")
            else:
                raise ValueError(
                    f"Cell ({q}, {r}) is outside the radius {self.radius}."
                )

        index = (q + self._offset) * self._padded_width + (r + self._offset)
        player_index = self.current_player - 1
        self._boards[player_index] |= 1 << index

        if self._check_win(self.current_player):
            return self.current_player

        # Turn logic: 1 move for P1 on turn 1, 2 moves for every turn after.
        self.moves_this_turn += 1
        moves_needed = 1 if self.turn_number == 1 else 2

        if self.moves_this_turn >= moves_needed:
            self.current_player = 3 - self.current_player  # Swap 1 and 2
            self.moves_this_turn = 0
            self.turn_number += 1

        return None

    def _check_win(self, player: int) -> bool:
        """Checks if the specified player has won the game.

        Uses extremely fast bitwise shifts to detect 6 in a row in all 3 axes.

        Args:
            player: The player number (1 or 2) to check.

        Returns:
            True if the player has 6 marks in a row, False otherwise.
        """
        bits = self._boards[player - 1]
        # Directions in axial grid bitboard:
        # 1: r-axis (0, 1)
        # _padded_width: q-axis (1, 0)
        # _padded_width-1: s-axis (1, -1)
        for direction in (1, self._padded_width, self._padded_width - 1):
            # Check for 6 in a row: bits & bits<<direction & bits<<2d & bits<<3d & bits<<4d & bits<<5d
            two_in_a_row = bits & (bits << direction)
            four_in_a_row = two_in_a_row & (two_in_a_row << (2 * direction))
            six_in_a_row = four_in_a_row & (two_in_a_row << (4 * direction))
            if six_in_a_row:
                return True
        return False

    @classmethod
    def from_string[T: HexGame](cls: type[T], text_grid: str) -> T:
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

        number_of_rows = len(lines)
        if number_of_rows % 2 == 0:
            raise ValueError("Hexagonal grid must have an odd number of rows (2R-1).")

        radius = (number_of_rows + 1) // 2
        game = cls(radius=radius)

        total_pieces: int = 0
        for row_index, line in enumerate(lines):
            r = row_index - (radius - 1)
            # Extract only the relevant symbols, ignoring spaces
            characters = "".join(c for c in line if c in "XO.")

            q_start = max(-(radius - 1), -(radius - 1) - r)
            expected_len = 2 * radius - 1 - abs(r)

            if len(characters) != expected_len:
                raise ValueError(
                    f"Row {row_index} (r={r}) has {len(characters)} cells, expected {expected_len}."
                )

            for q_index, character in enumerate(characters):
                if character == ".":
                    continue

                q = q_start + q_index
                index = (q + game._offset) * game._padded_width + (r + game._offset)
                if character == "X":
                    game._boards[0] |= 1 << index
                elif character == "O":
                    game._boards[1] |= 1 << index
                total_pieces = total_pieces + 1

        # Reconstruct turn state
        if total_pieces > 0:
            game.turn_number = (total_pieces + 1) // 2 + 1
            game.current_player = 2 if game.turn_number % 2 == 0 else 1
            game.moves_this_turn = (total_pieces - 1) % 2
        else:
            game.turn_number = 1
            game.current_player = 1
            game.moves_this_turn = 0

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
            # Indent proportional to distance from center to align as a hexagon
            indent = " " * abs(r)
            row_characters: list[str] = []

            q_start = max(-(self.radius - 1), -(self.radius - 1) - r)
            q_end = min((self.radius - 1), (self.radius - 1) - r)

            for q in range(q_start, q_end + 1):
                index = (q + self._offset) * self._padded_width + (r + self._offset)
                if self._boards[0] & (1 << index):
                    row_characters.append("X")
                elif self._boards[1] & (1 << index):
                    row_characters.append("O")
                else:
                    row_characters.append(".")

            lines.append(indent + " ".join(row_characters))

        return "\n".join(lines)
