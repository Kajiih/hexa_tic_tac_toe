# Hexagonal Tic-Tac-Toe

A high-performance hexagonal tic-tac-toe game engine implemented in Python, featuring a React-based visualizer.

## 🕹️ Game Rules

- **Board**: A hexagonal grid with a configurable radius (default: 50).
- **Turn Pattern**: To mitigate first-player advantage, the game follows a **1-2-2-2...** pattern:
    - Player 1 makes **1** move on the first turn.
    - Every subsequent turn, the active player makes **2** moves.
- **Winning Condition**: The first player to align **6** marks in a straight line (along any of the 3 hexagonal axes) wins.

## 📐 Coordinate System

The board uses the **Axial Coordinate System** $(q, r)$. 
- $q$ is the column index.
- $r$ is the row index.
- The third coordinate $s = -(q + r)$ is implicitly handled for win detection.
- A cell is valid if $|q| < R$, $|r| < R$, and $|q + r| < R$.

## 🚀 Performance

The engine uses **Bitboards** for extremely fast state management and win detection. Win checking is performed using bitwise shifts, allowing thousands of games to be simulated per second.

## 🛠️ Setup & Usage

### Prerequisites
- Python 3.13+
- [uv](https://github.com/astral-sh/uv) (recommended for dependency management)

### Installation
```bash
# Install dependencies
uv sync
```

### Running Tests
```bash
pytest
```

### Benchmarking
To measure the engine's performance:
```bash
python benchmark.py
```

### Visualizer
The project includes a React visualizer located in the `visualizer/` directory.
```bash
cd visualizer
npm install
npm run dev
```

## 📂 Project Structure

- `game.py`: Core bitboard-based game engine.
- `test_game.py` / `test_history.py`: Comprehensive test suites.
- `benchmark.py`: Performance measurement script.
- `visualizer/`: React/Vite/SVG frontend for game playback.
- `export_game.py`: Utility to export game data for the visualizer.
