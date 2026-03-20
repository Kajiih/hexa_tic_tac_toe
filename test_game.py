"""Unit tests for Hexagonal Tic Tac Toe."""

import pytest
from game import HexGame


@pytest.fixture
def game() -> HexGame:
    """Fixture to provide a standard HexGame instance for testing."""
    return HexGame(radius=50)


# --- Radius Validation ---
@pytest.mark.parametrize(
    "q, r, expected",
    [
        pytest.param(0, 0, True, id="center"),
        pytest.param(49, 0, True, id="q_edge"),
        pytest.param(0, 49, True, id="r_edge"),
        pytest.param(-49, 49, True, id="s_edge"),
        pytest.param(50, 0, False, id="q_outside_bounds"),
        pytest.param(0, 50, False, id="r_outside_bounds"),
        pytest.param(25, 25, False, id="s_outside_bounds_positive"),
        pytest.param(-25, -25, False, id="s_outside_bounds_negative"),
        pytest.param(-49, -1, False, id="q_r_negative"),
        pytest.param(49, 1, False, id="q_r_positive"),
        pytest.param(100, 100, False, id="distinctly_outside"),
    ],
)
def test_radius_validation(game: HexGame, q: int, r: int, expected: bool) -> None:
    """Tests that coordinates are correctly validated against the board radius."""
    assert game.is_valid_move(q, r) is expected


# --- Invalid Move Validations ---
@pytest.mark.parametrize(
    "q, r, match_text",
    [
        pytest.param(0, 0, "already occupied", id="already_occupied"),
        pytest.param(50, 0, "outside the radius", id="q_outside_radius"),
        pytest.param(0, 50, "outside the radius", id="r_outside_radius"),
        pytest.param(25, 25, "outside the radius", id="s_outside_radius_positive"),
        pytest.param(-25, -25, "outside the radius", id="s_outside_radius_negative"),
    ],
)
def test_invalid_make_move(game: HexGame, q: int, r: int, match_text: str) -> None:
    """Tests that invalid moves correctly raise appropriate ValueErrors."""
    game.make_move(0, 0)  # Occupy center before testing
    with pytest.raises(ValueError, match=match_text):
        game.make_move(q, r)


# --- Turn Pattern Logic ---
def test_turn_pattern(game: HexGame) -> None:
    """Tests the 1-2-2-2... turn pattern of the hexagonal tic-tac-toe game."""
    # Turn 1: P1 plays 1
    assert game.current_player == 1
    game.make_move(0, 0)

    # Turn 2: P2 plays 2
    assert game.current_player == 2
    game.make_move(1, 0)
    assert game.current_player == 2
    game.make_move(2, 0)

    # Turn 3: P1 plays 2
    assert game.current_player == 1
    game.make_move(0, 1)
    assert game.current_player == 1
    game.make_move(0, 2)


# --- False Win Fallbacks ---
@pytest.mark.parametrize(
    "move_sequence",
    [
        pytest.param([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)], id="5_in_a_row_q_axis"),
        pytest.param([(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)], id="5_in_a_row_r_axis"),
        pytest.param(
            [(0, 0), (1, -1), (2, -2), (3, -3), (4, -4)], id="5_in_a_row_s_axis"
        ),
        pytest.param(
            [(0, 1), (1, 1), (2, 1), (4, 1), (5, 1), (6, 1)], id="split_line_gap_of_1"
        ),
    ],
)
def test_no_false_win(move_sequence: list[tuple[int, int]]) -> None:
    """Tests that 5-in-a-row or split lines do not falsely trigger wins."""
    game = HexGame(radius=10)
    for q, r in move_sequence:
        assert game.make_move(q, r) is None
        # Force P1 to keep playing consecutive pieces for testing
        game.current_player = 1


# --- Real Win Validations ---
WIN_R_GRID = """
      . . . . . .
     . . . . . . .
    . . . . . . . .
   . . . . . . . . .
  . . . . . . . . . .
 . . . . . X . . . . .
  . . . . O X O . . .
   . . . . O X . . .
    . . . . O X . .
     . . . . O X .
      . . . . O .
"""

WIN_Q_GRID = """
      . . . . . .
     . . . . . . .
    . . . . . . . .
   . . . . . . . . .
  . . . . . . . . . .
 . . . . . X X X X X .
  . . . . O O O O O O
   . . . . . . . . .
    . . . . . . . .
     . . . . . . .
      . . . . . .
"""

WIN_S_GRID = """
      . . . . O .
     . . . . O X .
    . . . . O X . .
   . . . . O X . . .
  . . . . O X . . . .
 . . . . O X . . . . .
  . . . . . . . . . .
   . . . . . . . . .
    . . . . . . . .
     . . . . . . .
      . . . . . .
"""


@pytest.mark.parametrize(
    "grid, move_q, move_r, expected_winner",
    [
        pytest.param(WIN_R_GRID, 0, 5, 1, id="win_along_r_axis"),
        pytest.param(WIN_Q_GRID, 5, 0, 1, id="win_along_q_axis"),
        pytest.param(WIN_S_GRID, 5, -5, 1, id="win_along_s_axis"),
    ],
)
def test_win_conditions(
    grid: str, move_q: int, move_r: int, expected_winner: int
) -> None:
    """Tests win conditions along all three hexagonal axes."""
    game = HexGame.from_string(grid)
    winner = game.make_move(move_q, move_r)
    assert winner == expected_winner


# --- Parsing Exceptions ---
@pytest.mark.parametrize(
    "grid_str, match_text",
    [
        pytest.param("", "Empty grid string", id="empty_string"),
        pytest.param("   \n   ", "Empty grid string", id="whitespace_string"),
        pytest.param(" . . \n  . ", "odd number of rows", id="even_number_of_rows"),
        pytest.param(
            " . . \n . . \n . . ", "has 2 cells, expected 3", id="incorrect_row_length"
        ),
    ],
)
def test_from_string_invalid(grid_str: str, match_text: str) -> None:
    """Tests that invalid strings raise the correct ValueErrors during parsing."""
    with pytest.raises(ValueError, match=match_text):
        HexGame.from_string(grid_str)


# --- String Roundtrip Operations ---
@pytest.mark.parametrize(
    "move_sequence",
    [
        pytest.param([], id="empty_board"),
        pytest.param([(0, 0)], id="single_piece"),
        pytest.param([(0, 0), (1, 0)], id="two_pieces"),
        pytest.param(
            [(0, 0), (1, 0), (-1, -1), (0, 1), (-1, 0), (2, -1), (0, 2)],
            id="mid_game_sequence",
        ),
    ],
)
def test_string_roundtrip(move_sequence: list[tuple[int, int]]) -> None:
    """Tests that str(game) and from_string(game_string) are consistent."""
    game = HexGame(radius=4)
    for q, r in move_sequence:
        game.make_move(q, r)

    game_string = str(game)
    game2 = HexGame.from_string(game_string)
    assert str(game) == str(game2)
    assert game._boards == game2._boards
    assert game.current_player == game2.current_player
    assert game.turn_number == game2.turn_number
    assert game.moves_this_turn == game2.moves_this_turn


# --- String vs Manual Setup Equivalencies ---
@pytest.mark.parametrize(
    "grid, move_sequence",
    [
        pytest.param("  . .\n . . .\n  . .", [], id="empty_board"),
        pytest.param("  X .\n . . .\n  . .", [(0, -1)], id="single_piece_p1"),
        pytest.param(
            "  O X\n O X X\n  O O",
            [(1, -1), (0, -1), (-1, 0), (0, 0), (1, 0), (0, 1), (-1, 1)],
            id="mid_game_7_pieces",
        ),
    ],
)
def test_from_string_vs_manual(grid: str, move_sequence: list[tuple[int, int]]) -> None:
    """Tests that creating a game from string matches a manually created game."""
    game_str = HexGame.from_string(grid)

    game_manual = HexGame(radius=2)
    for q, r in move_sequence:
        game_manual.make_move(q, r)

    if len(move_sequence) == 7:
        game_manual.current_player = 1

    assert game_str._boards == game_manual._boards
    assert game_str.current_player == game_manual.current_player
    assert game_str.turn_number == game_manual.turn_number
    assert game_str.moves_this_turn == game_manual.moves_this_turn
    assert str(game_str) == str(game_manual)
