"""PettingZoo AEC Environment for Hexagonal Tic-Tac-Toe."""

import functools

import gymnasium
import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils import AgentSelector, wrappers

from hexa_tic_tac_toe.core.engine import HexGame, Player


def env(radius: int = 50, render_mode: str | None = None) -> AECEnv:
    """Creates a HexTicTacToeEnv with standard PettingZoo wrappers.

    Args:
        radius: The radius of the hexagonal board.
        render_mode: Optional rendering mode.

    Returns:
        The wrapped environment.
    """
    environment = HexTicTacToeEnv(radius=radius, render_mode=render_mode)
    environment = wrappers.TerminateIllegalWrapper(environment, illegal_reward=-1)
    environment = wrappers.AssertOutOfBoundsWrapper(environment)
    environment = wrappers.OrderEnforcingWrapper(environment)
    return environment


class HexTicTacToeEnv(AECEnv):
    """Hexagonal Tic-Tac-Toe environment for PettingZoo ending in AEC API."""

    metadata = {
        "render_modes": ["ansi", "human"],
        "name": "hexa_tic_tac_toe_v0",
    }

    def __init__(self, radius: int = 50, render_mode: str | None = None) -> None:
        """Initializes the environment.

        Args:
            radius: Radius of the board (defaults to 50 for large board).
            render_mode: Rendering mode String.
        """
        super().__init__()
        self.radius = radius
        self.game = HexGame(radius=radius)
        self.render_mode = render_mode

        self.agents = ["player_1", "player_2"]
        self.possible_agents = self.agents[:]

        # Padded width of the 2D matrix used for observation
        self.grid_size = 2 * radius - 1
        
        # Action space: A flat index of the 2D matrix grid_size x grid_size
        self._action_spaces = {
            agent: gymnasium.spaces.Discrete(self.grid_size * self.grid_size)
            for agent in self.agents
        }

        # Observation space: Standard spatial grid box (3 channels, H, W)
        # Channel 0: active player stones, Channel 1: opponent stones, Channel 2: valid mask
        self._observation_spaces = {
            agent: gymnasium.spaces.Box(
                low=0, high=1, shape=(3, self.grid_size, self.grid_size), dtype=np.int8
            )
            for agent in self.agents
        }

        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.agent_selection = ""

        # Precompute the board valid coordinates mask
        self._valid_mask = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        for q in range(-(self.radius - 1), self.radius):
            for r in range(-(self.radius - 1), self.radius):
                if abs(q + r) < self.radius:
                    matrix_q, matrix_r = self._axial_to_matrix(q, r)
                    self._valid_mask[matrix_q, matrix_r] = 1

    def _axial_to_matrix(self, q: int, r: int) -> tuple[int, int]:
        """Converts axial coordinates `(q, r)` to 0-indexed matrix coordinates."""
        return q + self.radius - 1, r + self.radius - 1

    def _matrix_to_axial(self, mq: int, mr: int) -> tuple[int, int]:
        """Converts 0-indexed matrix coordinates back to axial coordinates."""
        return mq - (self.radius - 1), mr - (self.radius - 1)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> gymnasium.spaces.Space:
        return self._observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> gymnasium.spaces.Space:
        return self._action_spaces[agent]

    def render(self) -> str | None:
        """Renders the environment."""
        if self.render_mode is None:
            gymnasium.logger.warn("You are calling render method without specifying any render mode.")
            return None
        return str(self.game)

    def observe(self, agent: str) -> np.ndarray:
        """Returns the spatial observation for the specified agent.

        Args:
            agent: The name of the agent.

        Returns:
            A 3D NumPy array (3, H, W).
        """
        agent_idx = self.agents.index(agent)
        active_player: Player = 1 if agent_idx == 0 else 2
        opponent_player: Player = 2 if active_player == 1 else 1

        obs = np.zeros((3, self.grid_size, self.grid_size), dtype=np.int8)
        action_mask = np.zeros(self.grid_size * self.grid_size, dtype=np.int8)

        # Build channels
        for q in range(-(self.radius - 1), self.radius):
            for r in range(-(self.radius - 1), self.radius):
                if abs(q + r) < self.radius:
                    mq, mr = self._axial_to_matrix(q, r)
                    player_at_cell = self.game.get_player_at(q, r)
                    
                    if player_at_cell == active_player:
                        obs[0, mq, mr] = 1
                    elif player_at_cell == opponent_player:
                        obs[1, mq, mr] = 1
                    elif player_at_cell is None:
                        # If empty, it's a valid action
                        action_index = mq * self.grid_size + mr
                        action_mask[action_index] = 1

        # Channel 2 is always the constant valid board mask
        obs[2] = self._valid_mask

        # Update action mask in the info dictionary for this agent
        self.infos[agent]["action_mask"] = action_mask

        return obs

    def close(self) -> None:
        """Closes the environment."""
        pass

    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        """Resets the environment to the starting state."""
        self.game.reset()
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        # Player 1 starts
        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.reset()

    def step(self, action: int | np.integer) -> None:
        """Executes a single step for the active agent."""
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            # Handles stepping after agent is dead
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        action = int(action)
        
        # Convert flat action index to matrix coordinates, then to axial
        mq = action // self.grid_size
        mr = action % self.grid_size
        q, r = self._matrix_to_axial(mq, mr)

        # Enforce valid move
        if not self.game.is_valid_move(q, r):
            # Should be prevented by action masking, but if not:
            self.terminations = {a: True for a in self.agents}
            self.rewards[agent] = -1
            self._accumulate_rewards()
            return

        # Make the move
        winner = self.game.make_move(q, r)

        if winner:
            self.terminations = {a: True for a in self.agents}
            winner_agent = f"player_{winner}"
            loser_agent = self.agents[1] if winner == 1 else self.agents[0]
            self.rewards[winner_agent] = +1
            self.rewards[loser_agent] = -1
        elif len(self.game.move_history) == np.sum(self._valid_mask):
            # Draw
            self.terminations = {a: True for a in self.agents}
            self.rewards = {a: 0 for a in self.agents}
        else:
            # Continuing game
            # Update agent selection based on game.current_player
            next_player_idx = self.game.current_player - 1
            self._agent_selector.reinit(self.agents)
            while self._agent_selector.next() != self.agents[next_player_idx]:
                pass
            self.agent_selection = self.agents[next_player_idx]

        self._accumulate_rewards()
